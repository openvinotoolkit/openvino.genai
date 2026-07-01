import torch
import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from .gemma3 import Gemma3InputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union, Any

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class Gemma4UnifiedInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        if chat_mode:
            raise ValueError("gemma4_unified does not currently support chat mode.")
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        pass

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

        if image is not None:
            image_token = getattr(processor, "image_token", "<|image|>")
            text = f"{image_token}{text}"

        return processor(images=image, text=text, return_tensors="pt")


class Gemma4InputsPreprocessor(Gemma3InputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_index", 258880)
        else:
            self.def_image_token_id = 258880

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        if "pixel_values" not in inputs:
            return inputs

        full_tokenized_chat_list = full_tokenized_chat[0].tolist()

        total_image_num = inputs["pixel_values"].shape[0]
        total_image_tokens = full_tokenized_chat_list.count(self.def_image_token_id)
        img_token_per_image = total_image_tokens // total_image_num if total_image_num > 0 else 0

        new_inputs_ids = full_tokenized_chat_list[prefix_len:]
        new_image_tokens = new_inputs_ids.count(self.def_image_token_id)
        new_image_num = new_image_tokens // img_token_per_image if img_token_per_image > 0 else 0
        if new_image_num < total_image_num:
            if new_image_num == 0:
                del inputs["pixel_values"]
                inputs.pop("image_position_ids", None)
            else:
                cached_image_num = total_image_num - new_image_num
                inputs["pixel_values"] = inputs["pixel_values"][cached_image_num:]
                if "image_position_ids" in inputs:
                    inputs["image_position_ids"] = inputs["image_position_ids"][cached_image_num:]

        return inputs
