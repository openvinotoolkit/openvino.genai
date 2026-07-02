import numpy as np
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union, Any
import torch

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class MiniCPMVInputsPreprocessor(VLMInputsPreprocessor):
    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_token_id", 128244)
        else:
            self.def_image_token_id = 128244

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

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        if "transformers" in str(type(model)):
            return inputs

        if prefix_len <= 0 or "pixel_values" not in inputs:
            return inputs

        image_bound = inputs.get("image_bound")
        pixel_values = inputs.get("pixel_values")
        tgt_sizes = inputs.get("tgt_sizes")
        if not image_bound:
            return inputs

        # batch == 1: image_bound = [Tensor[num_images, 2]], pixel_values = [[img0, img1, ...]]
        bounds = image_bound[0]
        if not torch.is_tensor(bounds):
            bounds = torch.as_tensor(bounds)

        keep_image_idx = []
        shifted_bounds = []
        for img_idx in range(bounds.shape[0]):
            start = int(bounds[img_idx][0])
            end = int(bounds[img_idx][1])
            if start >= prefix_len:
                keep_image_idx.append(img_idx)
                shifted_bounds.append([start - prefix_len, end - prefix_len])

        if len(keep_image_idx) == 0:
            inputs.pop("pixel_values", None)
            inputs.pop("tgt_sizes", None)
            inputs.pop("image_bound", None)
            return inputs

        inputs["image_bound"] = [torch.tensor(shifted_bounds, dtype=bounds.dtype)]
        inputs["pixel_values"] = [[pixel_values[0][i] for i in keep_image_idx]]
        if tgt_sizes is not None:
            inputs["tgt_sizes"] = [[tgt_sizes[0][i] for i in keep_image_idx]]

        return inputs
