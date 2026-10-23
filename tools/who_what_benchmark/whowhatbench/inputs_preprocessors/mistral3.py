# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from ..utils import no_double_bos
from typing import TYPE_CHECKING, Optional, Union, Any
import torch

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class Mistral3InputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        self.def_image_token_id = getattr(model.config, "image_token_index", None) if model is not None else None
        self.image_end_token_id = None

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

        self.image_end_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_end_token)

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

        with no_double_bos(processor):
            inputs = processor(images=self.images, text=text_prompt, return_tensors="pt")

        return inputs

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        if "pixel_values" not in inputs:
            return inputs

        # Pixtral image token count depends on image size, so count images by their single [IMG_END] marker.
        cached_image_num = full_tokenized_chat[0, :prefix_len].tolist().count(self.image_end_token_id)
        if cached_image_num >= inputs["pixel_values"].shape[0]:
            del inputs["pixel_values"]
            inputs.pop("image_sizes", None)
        elif cached_image_num > 0:
            inputs["pixel_values"] = inputs["pixel_values"][cached_image_num:]
            if "image_sizes" in inputs:
                inputs["image_sizes"] = inputs["image_sizes"][cached_image_num:]

        return inputs
