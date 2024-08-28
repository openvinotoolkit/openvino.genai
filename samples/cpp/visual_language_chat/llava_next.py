import torch
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
    unpad_image,
)
import openvino as ov
from pathlib import Path
from PIL import Image
import requests
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import numpy as np


class OVLlavaForCausalLM(GenerationMixin):
    def __init__(
        self,
        core,
        image_encoder_path,
        input_embedding_path,
        language_model_path,
        device,
    ):
        self.image_encoder = core.compile_model(core.read_model(image_encoder_path), device)
        self.input_embeddings = core.compile_model(core.read_model(input_embedding_path), device)
        self.model = core.read_model(language_model_path)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {idx: key for idx, key in enumerate(self.model.outputs)}
        self.key_value_input_names = [key for key in list(self.input_names) if key not in ["beam_idx", "inputs_embeds", "attention_mask", "position_ids"]]
        self.key_value_output_names = [key for key in list(self.output_names)[1:]]
        self.stateful = len(self.key_value_input_names) == 0
        compiled_model = core.compile_model(self.model, device)
        self.request = compiled_model.create_infer_request()
        self.config = AutoConfig.from_pretrained(Path(language_model_path).parent)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self.next_beam_idx = None
        self.image_newline = torch.zeros(self.config.text_config.hidden_size, dtype=torch.float32)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.past_len = 0
        self._supports_cache_class = False

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask,
        past_key_values,
        position_ids,
        image_sizes=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids,
            pixel_values,
            attention_mask,
            past_key_values,
            position_ids,
            image_sizes,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        attention_mask,
        past_key_values,
        position_ids,
        image_sizes=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""
        inputs = {}
        if past_key_values is not None:
            inputs = {}
            if not self.stateful:
                past_key_values = tuple(past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer)
                # Add the past_key_values to the decoder inputs
                inputs = dict(zip(self.key_value_input_names, past_key_values))
            # input_ids = np.array(input_ids)[:, -1:]
            inputs_embeds = self.input_embeddings(input_ids)[0]
            inputs["inputs_embeds"] = inputs_embeds
            # inputs["attention_mask"] = attention_mask
            if "beam_idx" in self.input_names:
                inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)

            if not self.stateful:
                first_layer_past_key_value = torch.from_numpy(past_key_values[0][0][:, :, :, 0])
            else:
                first_layer_past_key_value = torch.from_numpy(self.request.query_state()[0].state.data[:, :, :, 0])

            # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            # Get the target length
            target_length = input_ids.shape[1]
            past_length = first_layer_past_key_value.shape[-1]

            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            # Filter out only the tokens that can be un-attended, this can happen
            # if one uses Llava + Fused modules where the cache on the
            # first iteration is already big enough, or if one passes custom cache
            valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            new_batch_index = batch_index[valid_indices]
            new_non_attended_tokens = non_attended_tokens[valid_indices]

            # Zero-out the places where we don't need to attend
            extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            inputs["attention_mask"] = attention_mask
            inputs["position_ids"] = position_ids

        else:
            inputs = self.prepare_multimodal_input(input_ids, pixel_values, attention_mask, position_ids, image_sizes)

        # Run inference
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()

        logits = torch.from_numpy(self.request.get_tensor(self.output_names[0]).data)

        if not self.stateful:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))
        else:
            past_key_values = ((),)
        self.past_len += inputs["inputs_embeds"].shape[1]
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def prepare_multimodal_input(self, input_ids, pixel_values, attention_mask, position_ids, image_sizes=None):
        """Preprocessing function for embedding multimodal data"""
        inputs = {}
        inputs_embeds = torch.from_numpy(self.input_embeddings(input_ids)[0])
        batch_size = input_ids.shape[0]
        if not self.stateful:
            for input_name in self.key_value_input_names:
                model_inputs = self.modeget_anyres_image_grid_shapel.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = batch_size
                if shape[2].is_dynamic:
                    shape[2] = 0
                else:
                    shape[1] = 0
                inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())
        else:
            self.past_len = 0
            self.request.reset_state()
            # Set initial value for the next beam_idx input that will be used at the current iteration
            # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
            self.next_beam_idx = np.arange(batch_size, dtype=int)

        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
        if pixel_values is None:
            inputs["inputs_embeds"] = inputs_embeds
            inputs["attention_mask"] = attention_mask
            if position_ids is None:
                position_ids = torch.cumsum(attention_mask, axis=1) - 1
                position_ids[attention_mask == 0] = 1
            inputs["position_ids"] = position_ids
        res = self.image_encoder(pixel_values)
        image_features = torch.from_numpy(res[0])
        split_sizes = [image.shape[0] for image in pixel_values]
        image_features = torch.split(image_features, split_sizes, dim=0)

        # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

        new_image_features = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]

                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                image_feature = torch.cat(
                    (
                        image_feature,
                        self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1),
                    ),
                    dim=-1,
                )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
            new_image_features.append(image_feature)
        image_features = torch.stack(new_image_features, dim=0)

        (
            inputs_embeds,
            attention_mask,
            position_ids,
        ) = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, None)
        inputs["inputs_embeds"] = inputs_embeds
        inputs["attention_mask"] = attention_mask
        inputs["position_ids"] = position_ids

        return inputs

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        return final_embedding, final_attention_mask, position_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if not self.stateful:
                cache_length = past_length = past_key_values[0][0].shape[2]
            else:
                cache_length = past_length = self.past_len

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.llava
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch gllavaenerationsubset_siz
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
            }
        )
        return model_inputs

core = ov.Core()

MODEL_DIR = Path("llava_next")
IMAGE_ENCODER_PATH = MODEL_DIR / "image_encoder.xml"
INPUT_EMBEDDING_PATH = MODEL_DIR / "input_embeddings.xml"
LANGUAGE_MODEL_PATH = MODEL_DIR / "language_model.xml"

lang_model_path = LANGUAGE_MODEL_PATH
image_encoder_path = IMAGE_ENCODER_PATH

ov_llava_model = OVLlavaForCausalLM(core, image_encoder_path, INPUT_EMBEDDING_PATH, lang_model_path, "CPU")


from transformers import TextStreamer

url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
image = Image.open(requests.get(url, stream=True).raw)
question = "What is unusual on this image?"
prompt = f"[INST] <image>\n{question}[/INST]"
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
streamer = TextStreamer(processor, skip_special_tokens=True, skip_prompt=True)

# inputs = processor(prompt, image, return_tensors="pt")
# print(f"Question:\n{question}")
# image

# print("Answer:")
# streamer = TextStreamer(processor, skip_special_tokens=True, skip_prompt=True)
# output = ov_llava_model.generate(**inputs, max_new_tokens=49, streamer=streamer)


import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
from PIL import Image
import torch

example_image_urls = [
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1d6a0188-5613-418d-a1fd-4560aae1d907",
        "bee.jpg",
    ),
    (
        "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/6cc7feeb-0721-4b5d-8791-2576ed9d2863",
        "baklava.png",
    ),
]
for url, file_name in example_image_urls:
    Image.open(requests.get(url, stream=True).raw).save(file_name)


def bot_streaming(message, history):
    """
    message={'text': 'How to make this pastry?', 'files': [{'path': 'C:\\Users\\vzlobin\\AppData\\Local\\Temp\\gradio\\1e9b0138039150676a50089cce17f34abca38cb683e7200c58bf85bee6eb233c\\baklava.png', 'url': 'http://127.0.0.1:7860/file=C:\\Users\\vzlobin\\AppData\\Local\\Temp\\gradio\\1e9b0138039150676a50089cce17f34abca38cb683e7200c58bf85bee6eb233c\\baklava.png', 'size': None, 'orig_name': 'baklava.png', 'mime_type': 'image/png', 'is_stream': False, 'meta': {'_type': 'gradio.FileData'}}, {'path': 'C:\\Users\\vzlobin\\AppData\\Local\\Temp\\gradio\\cd36b2bdd95705a92bc803865cceae7fb7407c75946df3c0cb3ed45fb7bc30dd\\319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg', 'url': 'http://127.0.0.1:7860/file=C:\\Users\\vzlobin\\AppData\\Local\\Temp\\gradio\\cd36b2bdd95705a92bc803865cceae7fb7407c75946df3c0cb3ed45fb7bc30dd\\319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg', 'size': 404080, 'orig_name': '319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg', 'mime_type': 'image/jpeg', 'is_stream': False, 'meta': {'_type': 'gradio.FileData'}}]}
history=[[('C:\\Users\\vzlobin\\AppData\\Local\\Temp\\gradio\\486bb35919bc6394296e6e6d737d1e3b349247710d22dc021cba8fbd5b5d5fbe\\bee.jpg',), None], ['What is on the flower?', 'The flower in the image has a bee on it. The bee appears to be a bumblebee, which is a type of bee known for its large size and fuzzy body. The flower itself is a pink daisy-like flower with a yellow center. ']]

    The responce ignored the first image:
    > The image you've provided shows a cat lying in a cardboard box...
    """
    print(f"{message=}")
    print(f"{history=}")
    if message["files"]:
        image = message["files"][-1]["path"] if isinstance(message["files"][-1], dict) else message["files"][-1]
    else:
        # if there's no image uploaded for this turn, look for images in the past turns
        # kept inside tuples, take the last one
        for hist in history:
            if isinstance(hist[0], tuple):
                image = hist[0][0]

    if image is None:
        gr.Error("You need to upload an image for LLaVA to work.")
    prompt = f"[INST] <image>\n{message['text']} [/INST]"
    image = Image.open(image).convert("RGB")
    inputs = processor(prompt, image, return_tensors="pt")

    streamer = TextIteratorStreamer(processor, **{"skip_special_tokens": True})
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)

    thread = Thread(target=ov_llava_model.generate, kwargs=generation_kwargs)
    thread.start()

    text_prompt = f"[INST]  \n{message['text']} [/INST]"

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        generated_text_without_prompt = buffer[len(text_prompt) :]
        yield generated_text_without_prompt


demo = gr.ChatInterface(
    fn=bot_streaming,
    title="LLaVA NeXT",
    examples=[
        {"text": "What is on the flower?", "files": ["./bee.jpg"]},
        {"text": "How to make this pastry?", "files": ["./baklava.png"]},
    ],
    description="Try [LLaVA NeXT](https://huggingface.co/docs/transformers/main/en/model_doc/llava_next) in this demo using OpenVINO. Upload an image and start chatting about it, or simply try one of the examples below. If you don't upload an image, you will receive an error.",
    stop_btn="Stop Generation",
    multimodal=True,
)

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(debug=True, share=True)
