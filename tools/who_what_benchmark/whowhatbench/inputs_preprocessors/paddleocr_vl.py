import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union, Any

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class PaddleOCRVLInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for PaddleOCR-VL (model_type "paddleocr_vl").

    The processor chat template places image placeholders before the text of a user turn and
    the processor expands them into ``image_grid_thw``-sized image token sequences.
    Only image inputs are supported by the model.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode, model)

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
            raise ValueError("Video input is not supported by PaddleOCR-VL")
        if audio is not None:
            raise ValueError("Audio input is not supported by PaddleOCR-VL")

        self.update_images(image)
        content = []
        if image is not None:
            images = image if isinstance(image, list) else [image]
            content += [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text})

        new_message = {"role": "user", "content": content}
        if self.chat_mode:
            self.chat_history.append(new_message)
            conversation = self.chat_history
        else:
            conversation = [new_message]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        images = self.images
        if images is not None and not isinstance(images, list):
            images = [images]
        if not images:
            images = None

        return processor(images=images, text=text_prompt, return_tensors="pt")
