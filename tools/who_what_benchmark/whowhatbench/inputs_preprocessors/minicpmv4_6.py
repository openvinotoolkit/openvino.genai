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


class MiniCPMV4_6InputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for openbmb/MiniCPM-V-4.6 (model_type="minicpmv4_6").

    Unlike the legacy MiniCPM-V/-o models, MiniCPM-V-4.6 ships a modern
    transformers processor (``MiniCPMV4_6Processor``) that follows the standard
    image-text-to-text convention: images are declared in the chat content as
    ``{"type": "image"}`` items (expanded to the ``<|image_pad|>`` image token by
    the chat template) and ``processor(images=..., text=...)`` returns
    ``input_ids``, ``attention_mask``, ``pixel_values`` and ``target_sizes``.
    This differs from the legacy ``MiniCPMVInputsPreprocessor`` which relies on
    the ``(<image>./</image>)`` placeholder and ``image_bound``/``tgt_sizes``
    outputs, so a dedicated preprocessor is required.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", 248056)
        else:
            self.def_image_token_id = 248056

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

        inputs = processor(images=self.images, text=text_prompt, return_tensors="pt")

        return inputs

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        # Only native torch/HF models need image inputs shifted with respect to
        # the shared prefix cache; Optimum OpenVINO handles this internally.
        if "transformers" not in str(type(model)):
            return inputs

        if "pixel_values" not in inputs:
            return inputs

        image_token_id = getattr(model.config, "image_token_id", self.def_image_token_id)

        full_tokenized_chat_list = full_tokenized_chat[0].tolist()
        total_image_tokens = full_tokenized_chat_list.count(image_token_id)
        if total_image_tokens == 0:
            return inputs

        new_input_ids = full_tokenized_chat_list[prefix_len:]
        new_image_tokens = new_input_ids.count(image_token_id)

        # If none of the current-turn image tokens are outside the cached prefix,
        # the images are fully covered by the cache and must not be re-fed.
        if new_image_tokens == 0:
            inputs.pop("pixel_values", None)
            inputs.pop("target_sizes", None)

        return inputs
