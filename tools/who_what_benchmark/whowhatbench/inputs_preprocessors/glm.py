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
    """Inputs preprocessor for THUDM/GLM-Edge-V models (config.model_type == "glm").

    GLM-Edge-V does not expose a combined ``AutoProcessor``: the text side is a
    plain tokenizer whose chat template expands the image placeholder into a run
    of ``boi`` image tokens, while the vision side is a separate
    ``AutoImageProcessor`` that returns 6D ``pixel_values``. This class mirrors
    the model's documented multimodal generation path: chat-templated
    ``input_ids`` from the tokenizer plus ``pixel_values`` from the image
    processor.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        # boi_token_id marks image placeholders in the tokenized prompt.
        if model is not None:
            self.def_image_token_id = getattr(model.config, "boi_token_id", None)
        else:
            self.def_image_token_id = None

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def _resolve_image_processor(self, processor, config):
        # WWB may pass the text-only AutoProcessor (a bare tokenizer) here.
        # Fall back to loading the real image processor when needed.
        if processor is not None and (
            hasattr(processor, "image_mean") or hasattr(processor, "image_processor")
        ):
            return processor
        model_id = getattr(config, "_name_or_path", None)
        if model_id is None:
            raise ValueError(
                "An image processor is required for GLM-Edge-V but none was provided "
                "and config._name_or_path is unavailable to load one."
            )
        return AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)

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
        if self.chat_mode and getattr(tokenizer, "chat_template", None) is None:
            raise ValueError("Chat template is not set, but pipeline was run in chat mode.")

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

        # The GLM-Edge-V chat template inserts the image token run for each image.
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = dict(inputs)

        if self.images is not None:
            image_processor = self._resolve_image_processor(processor, config)
            pixel_values = image_processor(images=self.images, return_tensors="pt")["pixel_values"]
            inputs["pixel_values"] = pixel_values.to(torch.float32)

        return inputs
