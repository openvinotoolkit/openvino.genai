import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class DeepseekOCR2InputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        if chat_mode:
            raise ValueError("DeepSeek-OCR-2 is not supported in chat mode")
        super().__init__(chat_mode)

    def update_chat_history_with_answer(self, answer):
        pass

    def preprocess_inputs(
        self,
        text: Optional[str] = None,
        image: Optional["Image"] = None,
        processor: Optional[AutoImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        config: Optional[PretrainedConfig] = None,
        video: Optional["VideoInput"] = None,
        audio: Optional[np.ndarray] = None,
    ):
        if processor is None:
            raise ValueError("processor is required")
        if video is not None or audio is not None:
            raise ValueError("Video/audio inputs are not supported")
        if image is None:
            raise ValueError("Image input is required for DeepSeek-OCR-2")

        if text is None:
            text = "Free OCR."
        if "<image>" not in text:
            text = "<image>\n" + text
        return processor(images=image, text=text, return_tensors="pt")
