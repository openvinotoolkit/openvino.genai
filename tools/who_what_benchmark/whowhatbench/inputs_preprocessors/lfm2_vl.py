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


# Default image placeholder token id for LFM2-VL (LiquidAI/LFM2.5-VL-*).
# Kept as a fallback; the actual value is read from the model config when available.
DEFAULT_LFM2_VL_IMAGE_TOKEN_ID = 124907


class Lfm2VlInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for LFM2-VL (model_type == 'lfm2_vl').

    LFM2-VL uses a HuggingFace chat-template driven processor (Lfm2VlProcessor)
    that accepts message content of the form {"type": "image"/"text", ...},
    mirroring the Qwen2-VL style interface. This class builds the multimodal
    conversation, applies the chat template and calls the processor to produce
    the model inputs (input_ids, attention_mask, pixel_values, ...).
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(
                model.config, "image_token_id", DEFAULT_LFM2_VL_IMAGE_TOKEN_ID
            )
        else:
            self.def_image_token_id = DEFAULT_LFM2_VL_IMAGE_TOKEN_ID

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )

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
        if video is not None:
            raise ValueError("Video input is not supported for LFM2-VL")

        # Read the image placeholder token id from config when available.
        if config is not None:
            self.def_image_token_id = getattr(
                config, "image_token_id", self.def_image_token_id
            )

        self.update_images(image)

        media = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            media += [{"type": "image", "image": img} for img in image]

        new_message = {"role": "user", "content": media + [{"type": "text", "text": text}]}
        if self.chat_mode:
            self.chat_history.append(new_message)
            conversation = self.chat_history
        else:
            conversation = [new_message]

        text_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=self.images,
            text=text_prompt,
            return_tensors="pt",
        )

        return inputs
