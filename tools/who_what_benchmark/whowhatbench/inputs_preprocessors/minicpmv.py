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


class MiniCPMVInputsPreprocessor(VLMInputsPreprocessor):
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
        if processor is None:
            raise ValueError("Processor is required.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        self.update_images(image)
        im_suffix = ""
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            im_suffix = "(<image>./</image>)" * len(image) + "\n"

        apply_chat_template_func = None
        if getattr(processor, "chat_template", None) is not None:
            apply_chat_template_func = processor.apply_chat_template
        elif getattr(processor.tokenizer, "chat_template", None) is not None:
            apply_chat_template_func = processor.tokenizer.apply_chat_template

        if apply_chat_template_func is not None:
            new_message = {"role": "user", "content": im_suffix + text}
            if self.chat_mode:
                self.chat_history.append(new_message)
                messages = self.chat_history
            else:
                messages = [new_message]

            prompt = apply_chat_template_func(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = ""
            if self.chat_mode:
                raise ValueError("Chat mode is not supported when there is no chat_template in processor or tokenizer.")
            else:
                prompt = (
                    f"<|im_start|>user\n(<image>./</image>)\n{text}<|im_end|>\n<|im_start|>assistant\n"
                    if image is not None
                    else text
                )

        inputs = processor(prompt, [self.images] if self.images is None else self.images, return_tensors="pt")
        inputs.pop("image_sizes", None)
        return inputs
