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


class Phi3MMInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        self.image_offset = 1
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": answer})

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
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            for i, _ in enumerate(image):
                image_token = getattr(processor.tokenizer, "image_token", f"<|image_{i + self.image_offset}|>\n")
                if image_token not in text:
                    text = image_token + text
            self.image_offset += len(image)

        if getattr(processor.tokenizer, "chat_template", None) is None:
            if self.chat_mode:
                raise ValueError("Chat mode is not supported when there is no chat_template defined.")
        else:
            new_message = {"role": "user", "content": text}
            if self.chat_mode:
                self.chat_history.append(new_message)
                chat_prompt = self.chat_history
            else:
                chat_prompt = [new_message]

            text = processor.tokenizer.apply_chat_template(chat_prompt, add_generation_prompt=True, tokenize=False)

        inputs = processor(images=self.images, text=text, return_tensors="pt")
        return inputs

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        if "input_image_embeds" not in inputs:
            return inputs

        # phi3 uses negative image tokens to indicate, which start with -1 and increment each new image
        # but if we would like to pass only new inputs, we should start count image with -1
        new_inputs_tokens = full_tokenized_chat[0, prefix_len:]
        neg_mask = (new_inputs_tokens < 0) & (new_inputs_tokens > -int(1e9))
        new_image_indices = sorted({-int(v) for v in new_inputs_tokens[neg_mask].tolist()})

        if new_image_indices:
            keep = [k - 1 for k in new_image_indices]
            inputs["pixel_values"] = inputs["pixel_values"][keep]
            inputs["image_sizes"] = inputs["image_sizes"][keep]
        else:
            inputs.pop("pixel_values", None)
            inputs.pop("image_sizes", None)

        return inputs

    def is_image_token(self, tokenized_input: list, idx: int) -> bool:
        return tokenized_input[idx] < 0 and tokenized_input[idx] > -int(1e9)
