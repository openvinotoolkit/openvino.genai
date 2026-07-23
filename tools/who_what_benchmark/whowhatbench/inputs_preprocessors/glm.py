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


class GlmEdgeVInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for the GLM-Edge-V family (config.model_type == "glm"
    with a ``vision_config``).

    Unlike most VLMs, GLM-Edge-V does not ship a combined ``AutoProcessor``:
    ``AutoProcessor.from_pretrained`` resolves to the text tokenizer only, and
    the image side is a separate ``AutoImageProcessor`` (an ``MllamaImageProcessor``).
    So both ``processor`` and ``tokenizer`` passed here are the text tokenizer;
    the image processor is loaded here from ``config._name_or_path``.

    The documented inference contract is:
      * tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        return_dict=True, tokenize=True, return_tensors="pt") produces
        ``input_ids``/``attention_mask`` with ``config.vision_config`` worth of
        ``<|begin_of_image|>`` (``boi_token_id``) placeholder tokens per image;
      * ``AutoImageProcessor(image).pixel_values`` produces a
        ``[batch, num_media, num_tiles, 3, H, W]`` tensor;
      * ``model.generate(input_ids=..., attention_mask=..., pixel_values=...)``.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        self._image_processor = None
        if model is not None:
            self.def_image_token_id = getattr(model.config, "boi_token_id", None)
        else:
            self.def_image_token_id = None

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )

    def _get_image_processor(self, processor, config):
        # If the object handed to us already behaves like an image processor
        # (i.e. exposes pixel_values), reuse it directly.
        if processor is not None and hasattr(processor, "image_mean"):
            return processor
        if self._image_processor is None:
            if config is None or getattr(config, "_name_or_path", None) is None:
                raise ValueError(
                    "Cannot resolve the GLM-Edge-V image processor: config with "
                    "'_name_or_path' is required."
                )
            self._image_processor = AutoImageProcessor.from_pretrained(
                config._name_or_path, trust_remote_code=True
            )
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
        if tokenizer.chat_template is None:
            raise ValueError("Chat template is not set for the GLM-Edge-V tokenizer.")

        if image is not None and not isinstance(image, list):
            image = [image]
        self.update_images(image)

        content = []
        if image is not None:
            content.extend([{"type": "image"}] * len(image))
        content.append({"type": "text", "text": text})

        if self.chat_mode:
            self.chat_history.append({"role": "user", "content": content})
            messages = self.chat_history
        else:
            messages = [{"role": "user", "content": content}]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            tokenize=True,
            return_tensors="pt",
        )
        inputs = dict(inputs)

        if self.images:
            image_processor = self._get_image_processor(processor, config)
            pixel_values = image_processor(self.images).pixel_values
            inputs["pixel_values"] = torch.as_tensor(pixel_values)

        return inputs
