import torch

import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
    __version__,
)
from packaging.version import Version
from typing import TYPE_CHECKING, Optional, Union

from .vlm_inputs_preprocessor import VLMInputsPreprocessor

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


TRANSFORMERS_VERSION = Version(__version__)


class LLAVAInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
        super().__init__(chat_mode)

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
        if self.chat_mode and getattr(processor, "chat_template", None) is None:
            raise ValueError("Chat template is not set, but pipeline was run in chat mode.")

        if image is not None and not isinstance(image, list):
            image = [image]

        self.update_images(image)

        if getattr(processor, "chat_template", None) is not None:
            templated_prompt = {"role": "user", "content": [{"type": "text", "text": text}]}

            if image is not None:
                templated_prompt["content"].extend([{"type": "image"}] * len(image))

            if self.chat_mode:
                self.chat_history.append(templated_prompt)
                templated_input = self.chat_history
            else:
                templated_input = [templated_prompt]

            prompt = processor.apply_chat_template(templated_input, add_generation_prompt=True, tokenize=False)
        else:
            if image is not None and "<image>" not in text:
                prompt = ("<image>\n") * len(image) + text
            else:
                prompt = text

        if TRANSFORMERS_VERSION > Version("4.47.99") and getattr(processor, "patch_size", None) is None:
            if (
                getattr(config, "vision_config", None) is not None
                and getattr(config.vision_config, "patch_size", None) is not None
            ):
                processor.patch_size = config.vision_config.patch_size
            else:
                raise ValueError(
                    "Processor does not have `patch_size` attribute. Please fix the processor or provide `patch_size` in the config."
                )

        inputs = processor(images=self.images, text=prompt, return_tensors="pt")
        return inputs


class NanoLlavaInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
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
        if tokenizer is None:
            raise ValueError("Tokenizer is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")
        if image is not None and processor is None:
            raise ValueError("Processor is required.")

        if not isinstance(image, list):
            image = [image]

        self.update_images(image)
        if len(image) > 0 and image[0] is not None:
            text = "<image>\n" * len(image) + text

        new_message = {"role": "user", "content": text}
        if self.chat_mode:
            self.chat_history.append(new_message)
            messages = self.chat_history
        else:
            messages = [new_message]

        if tokenizer.chat_template is None:
            if self.chat_mode:
                raise ValueError("Chat mode is not supported when chat_template is not defined.")
        else:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_token_id = getattr(config, "image_token_index", None)
        if image_token_id is None:
            image_token_id = getattr(config, "image_token_id", -200)

        if "<image>" in text:
            text_chunks = text.split("<image>")
            input_ids = []

            for idx, chunk in enumerate(text_chunks):
                if chunk.strip() != "":
                    chunk_ids = tokenizer(chunk).input_ids
                    input_ids.extend(chunk_ids)
                if idx < len(text_chunks) - 1:
                    input_ids.append(image_token_id)

            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        else:
            input_ids = tokenizer(text, return_tensors="pt").input_ids

        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        result = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.images is not None:
            result["images"] = processor(images=self.images, return_tensors="pt")["pixel_values"]

        return result
