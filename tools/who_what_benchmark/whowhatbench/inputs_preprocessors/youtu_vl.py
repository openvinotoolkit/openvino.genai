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


class YoutuVLInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for the YoutuVL (Tencent Youtu-VL) model family.

    The YoutuVL processor (YoutuVLProcessor) follows the same interface as the
    Qwen2-VL processor: it accepts a list of message dicts, applies the model's
    chat template to produce a text string that already contains
    ``<|vision_start|><|image_pad|><|vision_end|>`` markers, and then calls
    the processor with ``(text=..., images=..., return_tensors="pt")`` to
    expand the image tokens and build the final model inputs.
    """

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
            raise ValueError("Processor is required for YoutuVL.")
        if audio is not None:
            raise ValueError("Audio input is not supported for YoutuVL.")

        self.update_images(image)

        media = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            media += [{"type": "image", "image": img} for img in image]

        if video is not None:
            if not isinstance(video, list):
                video = [video]
            media += [{"type": "video", "video": v} for v in video]
            if self.chat_mode:
                if self.videos is None:
                    self.videos = []
                self.videos.extend(video)
            else:
                self.videos = video
        elif not self.chat_mode:
            self.videos = None

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
