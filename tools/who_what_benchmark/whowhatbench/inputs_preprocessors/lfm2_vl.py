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


class Lfm2VlInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for LiquidAI LFM2-VL models (Lfm2VlForConditionalGeneration).

    The Lfm2VlProcessor expands ``<image>`` placeholders inside the chat template
    into the appropriate number of image tokens using the vision encoder metadata,
    so we build the conversation with ``{"type": "image"}`` items and let the
    processor apply its chat template and tokenize together with the images.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", 124907)
        else:
            self.def_image_token_id = 124907

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

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(images=self.images, text=inputs, return_tensors="pt")

        return inputs
