import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class Phi3MMInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False):
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
