import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from typing import TYPE_CHECKING, Optional, Union

from .vlm_inputs_preprocessor import VLMInputsPreprocessor

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class GLMEdgeVInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for the GLM-Edge-V vision-text family.

    GLM-Edge-V ships a text tokenizer (with chat template) plus a separate
    Mllama image processor; there is no unified HF processor. The chat template
    expands ``{"type": "image"}`` into the model's begin_of_image tokens, and
    the model expects ``pixel_values`` from the image processor.
    """

    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)
        self._image_processor = None

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def _get_image_processor(self, tokenizer, config):
        if self._image_processor is None:
            model_path = getattr(config, "_name_or_path", None)
            self._image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        return self._image_processor

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
        message = {"role": "user", "content": content}

        if self.chat_mode:
            self.chat_history.append(message)
            messages = self.chat_history
        else:
            messages = [message]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        if self.images is not None:
            image_processor = self._get_image_processor(tokenizer, config)
            inputs["pixel_values"] = image_processor(images=self.images, return_tensors="pt")["pixel_values"]

        return inputs
