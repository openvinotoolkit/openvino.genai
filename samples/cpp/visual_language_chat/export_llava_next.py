from pathlib import Path

MODEL_DIR = Path("llava_next")
IMAGE_ENCODER_PATH = MODEL_DIR / "image_encoder.xml"
INPUT_EMBEDDING_PATH = MODEL_DIR / "input_embeddings.xml"
LANGUAGE_MODEL_PATH = MODEL_DIR / "language_model.xml"

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import gc

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
image_encoder_model, input_embedding_model, language_model = None, None, None


class ImageEncoder(torch.nn.Module):
    def __init__(self, config, vision_tower, multi_modal_projector):
        super().__init__()
        self.config = config
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(self, pixel_values):
        batch_size, num_patches, num_channels, height, width = pixel_values.shape
        reshaped_pixel_values = pixel_values.view(batch_size * num_patches, num_channels, height, width)
        image_features = self.vision_tower(reshaped_pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.config.vision_feature_layer]
        if self.config.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.config.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features


model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", low_cpu_mem_usage=True)
model.config.save_pretrained(MODEL_DIR)
image_encoder_model = ImageEncoder(model.config, model.vision_tower, model.multi_modal_projector)
input_embedding_model = input_embedding_model = model.get_input_embeddings()
language_model = model.language_model
del model
gc.collect()

import torch
import openvino as ov
import gc


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


if not IMAGE_ENCODER_PATH.exists():
    ov_image_encoder = ov.convert_model(image_encoder_model, example_input=torch.zeros((1, 5, 3, 336, 336)))
    ov.save_model(ov_image_encoder, IMAGE_ENCODER_PATH)
    del ov_image_encoder
    cleanup_torchscript_cache()

del image_encoder_model
gc.collect()

llm_input = None

if not LANGUAGE_MODEL_PATH.exists():
    llm_input = input_embedding_model(torch.ones((2, 2), dtype=torch.int64))

if not INPUT_EMBEDDING_PATH.exists():
    ov_input_embeddings_model = ov.convert_model(input_embedding_model, example_input=torch.ones((2, 2), dtype=torch.int64))
    ov.save_model(ov_input_embeddings_model, INPUT_EMBEDDING_PATH)
    del ov_input_embeddings_model
    cleanup_torchscript_cache()

del input_embedding_model
gc.collect();

from typing import Optional, Tuple, List
from openvino.runtime import opset13
import numpy as np


def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[1:]]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )

make_stateful_model = True
core = ov.Core()

if not LANGUAGE_MODEL_PATH.exists():
    pkv = language_model(inputs_embeds=llm_input, attention_mask=torch.ones((2, 2), dtype=torch.int64))[1]
    model_inputs = ["attention_mask", "position_ids"]
    model_outputs = ["logits"]
    for idx in range(len(pkv)):
        model_inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        model_outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])
    model_inputs.append("inputs_embeds")
    language_model.config.torchscript = True
    position_ids = torch.tensor([[2, 3], [2, 3]])
    ov_model = ov.convert_model(
        language_model,
        example_input={
            "inputs_embeds": llm_input,
            "attention_mask": torch.ones((2, 4)),
            "past_key_values": pkv,
            "position_ids": position_ids,
        },
    )

    for input, input_name in zip(ov_model.inputs, model_inputs):
        input.get_tensor().set_names({input_name})

    for output, output_name in zip(ov_model.outputs, model_outputs):
        output.get_tensor().set_names({output_name})
    if make_stateful_model:
        patch_stateful(ov_model)
    ov.save_model(ov_model, LANGUAGE_MODEL_PATH)
    del ov_model
    cleanup_torchscript_cache()
    del language_model
    gc.collect()

# import nncf

# compression_configuration = {
#     "mode": nncf.CompressWeightsMode.INT4_SYM,
#     "group_size": 64,
#     "ratio": 0.6,
# }

# LANGUAGE_MODEL_PATH_INT4 = LANGUAGE_MODEL_PATH.parent / LANGUAGE_MODEL_PATH.name.replace(".xml", "-int4.xml")
# if to_compress_weights.value and not LANGUAGE_MODEL_PATH_INT4.exists():
#     ov_model = core.read_model(LANGUAGE_MODEL_PATH)
#     ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
#     ov.save_model(ov_compressed_model, LANGUAGE_MODEL_PATH_INT4)
#     del ov_compressed_model
#     del ov_model
#     gc.collect()

# IMAGE_ENCODER_PATH_INT8 = IMAGE_ENCODER_PATH.parent / IMAGE_ENCODER_PATH.name.replace(".xml", "-int8.xml")


# import requests

# r = requests.get(
#     url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
# )
# open("skip_kernel_extension.py", "w").write(r.text)

# %load_ext skip_kernel_extension

# %%skip not $to_quantize.value

# import requests
# import requests
# from io import BytesIO
# import numpy as np
# from PIL import Image
# from requests.packages.urllib3.exceptions import InsecureRequestWarning
# requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


# def get_pil_from_url(url):
#     """
#     Downloads and converts an image from a URL to a PIL Image object.
#     """
#     response = requests.get(url, verify=False, timeout=20)
#     image = Image.open(BytesIO(response.content))
#     return image.convert("RGB")

# def collate_fn(example, image_column="image_url"):
#     """
#     Preprocesses an example by loading and transforming image and text data.
#     Checks if the text data in the example is valid by calling the `check_text_data` function.
#     Downloads the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
#     If there is any error during the download process, returns None.
#     Returns the preprocessed inputs with transformed image and text data.
#     """
#     assert len(example) == 1
#     example = example[0]
#     url = example[image_column]
#     try:
#         image = get_pil_from_url(url)
#         h, w = image.size
#         if h == 1 or w == 1:
#             return None
#     except Exception:
#         return None

#     inputs = processor.image_processor(images=[image], return_tensors="pt")
#     return inputs

# %%skip not $to_quantize.value

# import torch
# from datasets import load_dataset
# from tqdm.notebook import tqdm

# def prepare_calibration_data(dataloader, init_steps):
#     """
#     This function prepares calibration data from a dataloader for a specified number of initialization steps.
#     It iterates over the dataloader, fetching batches and storing the relevant data.
#     """
#     data = []
#     print(f"Fetching {init_steps} samples for the initialization...")
#     with tqdm(total=init_steps) as pbar:
#         for batch in dataloader:
#             if len(data) == init_steps:
#                 break
#             if batch:
#                 pbar.update(1)
#                 with torch.no_grad():
#                     data.append(
#                         {
#                             "pixel_values": batch["pixel_values"].to("cpu")
#                         }
#                     )
#     return data


# def prepare_dataset(opt_init_steps=50, max_train_samples=1000):
#     """
#     Prepares a vision-text dataset for quantization.
#     """
#     dataset = load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True)
#     train_dataset = dataset["train"].shuffle(seed=42)
#     dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1)
#     calibration_data = prepare_calibration_data(dataloader, opt_init_steps)
#     return calibration_data

# %%skip not $to_quantize.value

# vcalibration_data = []
# if not IMAGE_ENCODER_PATH_INT8.exists():
#     calibration_data = prepare_dataset()

# %%skip not $to_quantize.value


# if not IMAGE_ENCODER_PATH_INT8.exists():
#     if len(calibration_data) == 0:
#         raise RuntimeError(
#             'Calibration dataset is empty. Please check internet connection and try to download images manually.'
#         )

#     ov_model = core.read_model(IMAGE_ENCODER_PATH)
#     calibration_dataset = nncf.Dataset(calibration_data)
#     quantized_model = nncf.quantize(
#         model=ov_model,
#         calibration_dataset=calibration_dataset,
#         model_type=nncf.ModelType.TRANSFORMER,
#         subset_size=len(calibration_data),
#         # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
#         advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6)
#     )
#     ov.save_model(quantized_model, IMAGE_ENCODER_PATH_INT8)
#     del ov_model
#     del quantized_model
#     gc.collect()
