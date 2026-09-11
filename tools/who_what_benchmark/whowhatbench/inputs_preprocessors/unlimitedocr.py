import importlib
import math

import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from typing import TYPE_CHECKING, Optional, Union, Any

from .vlm_inputs_preprocessor import VLMInputsPreprocessor

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


# Constants defined by the UnlimitedOCR / DeepSeek-OCR remote-code `infer` method.
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
BOS_ID = 0
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
DEFAULT_BASE_SIZE = 1024
DEFAULT_IMAGE_SIZE = 640


class UnlimitedOCRInputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for baidu/Unlimited-OCR (model_type == "unlimited-ocr").

    The model ships a custom remote-code ``UnlimitedOCRForCausalLM`` (SAM ViT-B +
    CLIP-L -> linear projector -> DeepSeek-V2 MoE). Generation does not go through
    a HF processor: the reference ``infer`` method builds ``input_ids`` interleaved
    with image placeholder tokens plus ``images``, ``images_seq_mask`` and
    ``images_spatial_crop`` tensors that the custom ``forward`` consumes.

    This class reproduces that preprocessing (crop mode) generically, reusing the
    model's own remote-code helpers (image transform, dynamic crop, text encode)
    resolved from the live config module, so no dependency on a Hub id is needed.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        self.def_image_token_id = IMAGE_TOKEN_ID

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "<|Assistant|>", "content": answer})

    def _resolve_remote_helpers(self, config: PretrainedConfig):
        # The custom config class is defined in the model's remote-code module,
        # which also exposes BasicImageTransform / dynamic_preprocess / text_encode.
        module = importlib.import_module(type(config).__module__)
        return (
            module.BasicImageTransform,
            module.dynamic_preprocess,
            module.text_encode,
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
        if tokenizer is None:
            raise ValueError("Tokenizer is required for unlimited-ocr preprocessing.")
        if config is None:
            raise ValueError("Model config is required for unlimited-ocr preprocessing.")
        if video is not None:
            raise ValueError("Video input is not supported")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        from PIL import ImageOps

        BasicImageTransform, dynamic_preprocess, text_encode = self._resolve_remote_helpers(config)

        base_size = DEFAULT_BASE_SIZE
        image_size = DEFAULT_IMAGE_SIZE

        images = []
        if image is not None:
            if not isinstance(image, list):
                image = [image]
            images = [im.convert("RGB") for im in image if im is not None]
        self.update_images(images if images else None)

        # Interleave one <image> placeholder per image if the prompt has none.
        if images and IMAGE_TOKEN not in text:
            prompt = (IMAGE_TOKEN + "\n") * len(images) + text
        else:
            prompt = text

        image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        # Model is loaded in float32 by WWB (no torch_dtype override); keep inputs float32.
        target_dtype = torch.float32

        text_splits = prompt.split(IMAGE_TOKEN)

        images_list, images_crop_list = [], []
        tokenized_str, images_seq_mask, images_spatial_crop = [], [], []

        num_queries = math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
        num_queries_base = math.ceil((base_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)

        for text_sep, img in zip(text_splits, images):
            tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            # crop mode (matches infer default crop_mode=True)
            if img.size[0] <= 640 and img.size[1] <= 640:
                images_crop_raw, crop_ratio = [], [1, 1]
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(img)

            global_view = ImageOps.pad(
                img,
                (base_size, base_size),
                color=tuple(int(x * 255) for x in image_transform.mean),
            )
            images_list.append(image_transform(global_view).to(target_dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for crop in images_crop_raw:
                    images_crop_list.append(image_transform(crop).to(target_dtype))

            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
            tokenized_image += [IMAGE_TOKEN_ID]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]) * (
                    num_queries * height_crop_num
                )
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

        # trailing text after the last <image> placeholder
        tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # prepend bos
        tokenized_str = [BOS_ID] + tokenized_str
        images_seq_mask = [False] + images_seq_mask

        input_ids = torch.LongTensor(tokenized_str).unsqueeze(0)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, image_size, image_size), dtype=target_dtype)
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, base_size, base_size), dtype=target_dtype)
        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, base_size, base_size), dtype=target_dtype)

        # The DeepSeek-V2 backbone uses a ring-buffer sliding-window KV cache during
        # decode; the reference `infer` disables `config.sliding_window` before
        # generate so DynamicCache does not truncate the (long) image prefill.
        # Mirror that here on the live config object.
        if getattr(config, "sliding_window", None) is not None:
            config.sliding_window = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": [(images_crop, images_ori)],
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }
