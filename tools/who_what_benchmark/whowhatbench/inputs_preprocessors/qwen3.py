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


class Qwen3VLInputsPreprocessor(VLMInputsPreprocessor):
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
        if audio is not None:
            raise ValueError("Audio input is not supported")

        media = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            media += [{"type": "image", "image": img} for img in image]
        if video is not None:
            if not isinstance(video, list):
                video = [video]
            media += [{"type": "video", "video": v} for v in video]

        new_message = {"role": "user", "content": media + [{"type": "text", "text": text}]}
        if self.chat_mode:
            self.chat_history.append(new_message)
            conversation = self.chat_history
        else:
            conversation = [new_message]

        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs
