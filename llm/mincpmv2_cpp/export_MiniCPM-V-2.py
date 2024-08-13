import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import openvino as ov
from openvino_tokenizers import convert_tokenizer

import logging as log
from openvino.runtime import opset13
import numpy as np
from typing import  List
import nncf

from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
from openvino.runtime import opset10 as ops

def save_tokenizer(tokenizer, out_dir):
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        log.error(f'tokenizer loading failed with {e}')

class EmbeddingModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_id):
        return self.model.model.embed_tokens(inputs_id)

class VisionModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_pixel):
        return self.model.forward_features(image_pixel)

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

class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            print("root: ", root)
            if root is None:
                return False
            root_output = matcher.get_match_value()
            print("root_output", root_output)
            root_name = root.get_friendly_name()
            if (len(root.get_output_partial_shape(0)) == 3):
                print(f"Find target root node name: {root_name}")
                parent = root.input_value(0).get_node()
                print(f"Find target parent node name: {parent.get_friendly_name()}")
                grand_parent = parent.input_value(0).get_node()
                print(f"Find grandparent node name: {grand_parent.get_friendly_name()}")
                grand_parent_output = parent.input(0).get_source_output()
                print("grand_parent_output: ", grand_parent_output)
                consumers = grand_parent_output.get_target_inputs()

                print(f"consumers: {consumers}")
                print("Original reshape node output shape:", grand_parent_output.get_partial_shape())
                dims = grand_parent_output.get_partial_shape().get_min_shape()[2]
                print("grand_parent_output dims : ", dims)
                start = np.array([0, -1, 0], dtype=np.int32)
                stop = np.array([1, -2, dims], dtype=np.int32)
                step = np.array([1, -1, 1], dtype=np.int32)
                axes = np.array([0, 1, 2], dtype=np.int32)
                slice = ops.slice(grand_parent, start, stop, step, axes, name="inserted_slice")
                print("After insert slice node, output shape:", slice.output(0).get_partial_shape())

                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)

                return True

        self.register_matcher(Matcher(param,"InsertSlice"), callback)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Export minicpm-v2 Model to IR", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")

    args = parser.parse_args()
    model_path = args.output_dir

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    model = model.to(device='cpu')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # set path to save openvino IR
    VISION_MODEL_OV = Path(f"{model_path}/openvino_vision.xml")
    RESAMPLER_MODEL_OV = Path(f"{model_path}/openvino_resampler.xml")
    TOKENIZER_MODEL_OV = Path(f"{model_path}/openvino_tokenizer.xml")
    DE_TOKENIZER_MODEL_OV = Path(f"{model_path}/openvino_detokenizer.xml")
    EMBEDDING_MODEL_OV = Path(f"{model_path}/openvino_embedding.xml")
    LLM_MODEL_OV = Path(f"{model_path}/openvino_model.xml")
    LLM_MODEL_OV_INT4 = Path(f"{model_path}/openvino_model_int4.xml")
    LLM_MODEL_OV_INT4_REDUCE_LOGITS = Path(f"{model_path}/openvino_model_int4.xml")

    # convert vision model to openvino IR
    if not VISION_MODEL_OV.exists():
        #vision_model = model.vpm
        vision_model = VisionModel(model.vpm)
        vision_model.eval()
        image_pixel = torch.randn(1, 3, 448, 448, dtype=torch.float32)
        ov_vision_model = ov.convert_model(vision_model, example_input=image_pixel)
        ov.save_model(ov_vision_model, str(VISION_MODEL_OV), compress_to_fp16=True)

    # convert resampler model to openvino IR
    if not RESAMPLER_MODEL_OV.exists():
       resampler_model = model.resampler
       resampler_model.eval()
       vision_embedding = torch.randn(1, 1024, 1152, dtype=torch.float32)
       tgt_size = torch.tensor([[32], [32]])
       inputs = (vision_embedding, tgt_size)
       ov_resampler = ov.convert_model(resampler_model, example_input=inputs)
       ov.save_model(ov_resampler, str(RESAMPLER_MODEL_OV), compress_to_fp16=True)

    if not TOKENIZER_MODEL_OV.exists():
        ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True, handle_special_tokens_with_re=True)
        ov.save_model(ov_tokenizer, str(TOKENIZER_MODEL_OV))
        ov.save_model(ov_detokenizer, str(DE_TOKENIZER_MODEL_OV))

    if not EMBEDDING_MODEL_OV.exists():
        embdedding_model = EmbeddingModel(model.llm)
        ov_embedding_model = ov.convert_model(embdedding_model, example_input=torch.ones((1, 10), dtype=torch.long))
        ov.save_model(ov_embedding_model, str(EMBEDDING_MODEL_OV))

    make_stateful_model = True
    if not LLM_MODEL_OV.exists():
        language_model = model.llm
        embdedding_model = EmbeddingModel(model.llm)
        llm_input = embdedding_model(torch.ones((2, 2), dtype=torch.int64))
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
        ov.save_model(ov_model, LLM_MODEL_OV)
        save_tokenizer(tokenizer, model_path)
        model.config.save_pretrained(model_path)

    if not LLM_MODEL_OV_INT4.exists() and LLM_MODEL_OV.exists():
        compression_configuration = {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 128,
            "ratio": 1,
            }
        core = ov.Core()
        print("LLM model_ov", LLM_MODEL_OV)
        ov_model = core.read_model(LLM_MODEL_OV)
        ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
        ov.save_model(ov_compressed_model, LLM_MODEL_OV_INT4)

    if LLM_MODEL_OV_INT4.exists():
        core = ov.Core()
        ov_model = core.read_model(LLM_MODEL_OV_INT4)
        manager = Manager()
        manager.register_pass(InsertSlice())
        manager.run_passes(ov_model)
        ov.save_model(ov_model, LLM_MODEL_OV_INT4_REDUCE_LOGITS)
