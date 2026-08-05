import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union, Any
import torch

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class Gemma3InputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_index", 262144)
        else:
            self.def_image_token_id = 262144

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def preprocess_inputs(
        self,
        text: str,
        image: Optional[Union["Image", list["Image"]]] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional[Union["VideoInput", list["VideoInput"]]] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        self.update_images(image)
        content = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            content.extend([{"type": "image"}] * len(image))

        content.append({"type": "text", "text": text})

        if self.chat_mode:
            self.chat_history.append({"role": "user", "content": content})
            conversation = self.chat_history
        else:
            conversation = [{"role": "user", "content": content}]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        # switch off add_bos_token if chat template already includes it
        orig_add_bos_token = processor.tokenizer.add_bos_token
        if getattr(processor.tokenizer, "chat_template", None) and "bos_token" in processor.tokenizer.chat_template:
            processor.tokenizer.add_bos_token = False

        inputs = processor(images=self.images, text=text_prompt, return_tensors="pt")

        # recover add_bos_token flag in tokenizer
        processor.tokenizer.add_bos_token = orig_add_bos_token

        return inputs

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        if "pixel_values" not in inputs:
            return inputs

        image_token_id = getattr(model.config, "image_token_id", self.def_image_token_id)

        full_tokenized_chat_list = full_tokenized_chat[0].tolist()

        total_image_num = inputs["pixel_values"].shape[0]
        total_image_tokens = full_tokenized_chat_list.count(image_token_id)
        img_token_per_image = total_image_tokens // total_image_num if total_image_num > 0 else 0

        new_inputs_ids = full_tokenized_chat_list[prefix_len:]
        new_image_tokens = new_inputs_ids.count(image_token_id)
        new_image_num = new_image_tokens // img_token_per_image if img_token_per_image > 0 else 0
        if new_image_num < total_image_num:
            if new_image_num == 0:
                del inputs["pixel_values"]
            else:
                cached_image_num = total_image_num - new_image_num
                inputs["pixel_values"] = inputs["pixel_values"][cached_image_num:]

        return inputs
