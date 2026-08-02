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


class JinaVLMInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for the JinaVLM family (model_type='jvlm',
    JinaVLMForConditionalGeneration).

    JinaVLM ships a dedicated `JinaVLMProcessor` that wraps an image processor
    and a tokenizer. It expects the image placeholder token (`<|image|>`) to be
    present in the text; this is produced by the model's chat template for
    `content` entries of type `image`. The processor then builds
    `input_ids`, `image_patches`, `image_input_idx`, `image_masks` and
    `attention_mask`, which the model's `generate()` consumes directly.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        # JinaVLM interleaves image features by an image placeholder token id.
        # Prefer the processor-derived id when available; fall back to config.
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", None)
        else:
            self.def_image_token_id = None

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
        if getattr(processor, "chat_template", None) is None:
            raise ValueError("JinaVLM requires a chat template to build image-text inputs.")

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
            conversation = self.chat_history
        else:
            conversation = [new_message]

        prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=self.images,
            text=prompt,
            return_tensors="pt",
        )

        return inputs
