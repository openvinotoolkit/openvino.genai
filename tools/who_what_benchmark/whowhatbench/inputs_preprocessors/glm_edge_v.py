import torch

import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from typing import TYPE_CHECKING, Optional, Union, Any

from .vlm_inputs_preprocessor import VLMInputsPreprocessor

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class GlmEdgeVInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for GLM-Edge-V family (config.model_type == "glm").

    The GLM-Edge-V VLM ships a remote-code ``GlmForCausalLM`` whose
    ``AutoProcessor`` resolves to a bare tokenizer, while the image side is a
    standalone ``AutoImageProcessor`` (an ``MllamaImageProcessor`` in the
    reference weights). The chat template itself inserts the
    ``<|begin_of_image|>`` (``boi_token_id``) placeholders, so text is tokenized
    via ``tokenizer.apply_chat_template`` and images are encoded separately into
    the ``pixel_values`` tensor consumed by ``model.forward``.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "boi_token_id", None)
        else:
            self.def_image_token_id = None

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )

    @staticmethod
    def _resolve_image_processor(processor, config):
        # The GLM-Edge-V "processor" is a bare tokenizer; the image side is a
        # separate AutoImageProcessor. Prefer an already-attached image
        # processor, otherwise load one from the model directory referenced by
        # the config.
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            return image_processor
        if callable(getattr(processor, "preprocess", None)):
            # processor is itself an image processor
            return processor
        model_path = getattr(config, "_name_or_path", None)
        if model_path is None:
            raise ValueError(
                "Cannot resolve an image processor for GLM-Edge-V: config has no "
                "'_name_or_path'."
            )
        return AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

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
        if tokenizer is None:
            raise ValueError("Tokenizer is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        if image is not None and not isinstance(image, list):
            image = [image]
        self.update_images(image)

        content = []
        if image is not None:
            content.extend([{"type": "image"}] * len(image))
        content.append({"type": "text", "text": text})

        new_message = {"role": "user", "content": content}
        if self.chat_mode:
            self.chat_history.append(new_message)
            messages = self.chat_history
        else:
            messages = [new_message]

        if getattr(tokenizer, "chat_template", None) is None:
            raise ValueError(
                "GLM-Edge-V requires a chat template to place image tokens, "
                "but none is defined on the tokenizer."
            )

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

        result = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.images is not None:
            image_processor = self._resolve_image_processor(processor, config)
            pixel_values = image_processor(images=self.images, return_tensors="pt")[
                "pixel_values"
            ]
            # GLM-Edge-V forward expects pixel_values of shape
            # (batch, num_concurrent_media, num_tiles, channels, height, width).
            result["pixel_values"] = pixel_values

        return result
