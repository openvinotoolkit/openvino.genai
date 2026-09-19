import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from ..utils import no_double_bos
from typing import TYPE_CHECKING, Optional, Union, Any

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class Lfm2VlInputsPreprocessor(VLMInputsPreprocessor):
    """Input preprocessor for the LiquidAI LFM2-VL family (model_type == "lfm2_vl").

    LFM2-VL (Lfm2VlForConditionalGeneration) pairs a SigLIP2 vision tower with an
    LFM2 text backbone and exposes a standard HF ``Lfm2VlProcessor`` whose chat
    template consumes a message ``content`` list of ``{"type": "image"}`` /
    ``{"type": "text"}`` items and emits a single ``<image>`` placeholder per image
    (``image_token_id`` 124907). The processor produces ``pixel_values`` together
    with ``pixel_attention_mask`` and ``spatial_shapes`` for the packed SigLIP2
    vision inputs.
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

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        # The LFM2-VL chat template already emits the bos token; avoid duplicating it
        # when the processor's tokenizer would also prepend one.
        with no_double_bos(processor):
            inputs = processor(images=self.images, text=text_prompt, return_tensors="pt")

        return inputs
