import numpy as np
import torch
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


class GLMEdgeVInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for the GLM-Edge-V family of VLMs (config.model_type == "glm").

    GLM-Edge-V does not expose a combined multimodal ``AutoProcessor``: loading it
    returns the text tokenizer only, and images are handled by a separate
    ``MllamaImageProcessor``. The chat template expands every ``{"type": "image"}``
    entry into a fixed run of ``<|begin_of_image|>`` (``boi_token_id``) placeholder
    tokens which the model replaces with vision features during ``forward``.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        self._image_processor = None
        if model is not None:
            self.def_image_token_id = getattr(model.config, "boi_token_id", None)
        else:
            self.def_image_token_id = None

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def _get_image_processor(self, processor, config):
        if self._image_processor is not None:
            return self._image_processor

        # GLM-Edge-V's AutoProcessor resolves to the text tokenizer, so the image
        # processor must be loaded separately from the same model directory.
        model_path = getattr(processor, "name_or_path", None)
        if model_path is None and config is not None:
            model_path = getattr(config, "_name_or_path", None)
        if model_path is None:
            raise ValueError(
                "Unable to determine model path to load the GLM-Edge-V image processor."
            )
        self._image_processor = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        return self._image_processor

    def preprocess_inputs(
        self,
        text: str,
        image: Optional[Union["Image", list["Image"]]] = None,
        processor: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional[Union["VideoInput", list["VideoInput"]]] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None and tokenizer is None:
            raise ValueError("Tokenizer/processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        # For GLM-Edge-V the AutoProcessor loaded by WWB is the text tokenizer.
        text_tokenizer = tokenizer if tokenizer is not None else processor

        if image is not None and not isinstance(image, list):
            image = [image]
        self.update_images(image)

        content = []
        if image is not None:
            content.extend([{"type": "image"}] * len(image))
        content.append({"type": "text", "text": text})

        if self.chat_mode:
            self.chat_history.append({"role": "user", "content": content})
            conversation = self.chat_history
        else:
            conversation = [{"role": "user", "content": content}]

        prompt = text_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = text_tokenizer(prompt, return_tensors="pt")

        if self.images is not None:
            image_processor = self._get_image_processor(processor, config)
            image_inputs = image_processor(images=self.images, return_tensors="pt")
            inputs["pixel_values"] = image_inputs["pixel_values"]

        return inputs

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        if "pixel_values" not in inputs:
            return inputs

        boi_token_id = getattr(model.config, "boi_token_id", self.def_image_token_id)
        if boi_token_id is None:
            return inputs

        full_tokenized_chat_list = full_tokenized_chat[0].tolist()

        total_image_num = inputs["pixel_values"].shape[0]
        total_image_tokens = full_tokenized_chat_list.count(boi_token_id)
        boi_per_image = total_image_tokens // total_image_num if total_image_num > 0 else 0

        new_input_ids = full_tokenized_chat_list[prefix_len:]
        new_image_tokens = new_input_ids.count(boi_token_id)
        new_image_num = new_image_tokens // boi_per_image if boi_per_image > 0 else 0
        if new_image_num < total_image_num:
            if new_image_num == 0:
                del inputs["pixel_values"]
            else:
                cached_image_num = total_image_num - new_image_num
                inputs["pixel_values"] = inputs["pixel_values"][cached_image_num:]

        return inputs

    def is_image_token(self, tokenized_input: list, idx: int) -> bool:
        if self.def_image_token_id is None:
            return True
        return tokenized_input[idx] == self.def_image_token_id
