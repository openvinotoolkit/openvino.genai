import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
    SiglipImageProcessor,
)
from .vlm_inputs_preprocessor import VLMInputsPreprocessor
from typing import TYPE_CHECKING, Optional, Union, Any

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


def _load_image(image):
    """Load a PIL Image from a path string or return as-is if already a PIL Image."""
    if isinstance(image, str):
        from PIL import Image as PILImage
        return PILImage.open(image).convert("RGB")
    return image


class GlmInputsPreprocessor(VLMInputsPreprocessor):
    """
    Input preprocessor for GLM-Edge-V (THUDM/glm-edge-v-*) models.

    The GLM model encodes each image as exactly 578 <|begin_of_image|> placeholder
    tokens embedded through a SigLIP vision encoder. Images are resized to
    672x672 pixels and the pixel_values tensor is shaped
    [batch, num_concurrent_media, num_tiles, channels, height, width].

    The ``processor`` argument is expected to be the AutoTokenizer / AutoProcessor
    loaded from the model directory (it resolves to a PreTrainedTokenizerFast
    with a chat_template that injects the 578 BOI tokens automatically).
    """

    _IMAGE_SIZE = 672

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode)
        if model is not None:
            self.def_image_token_id = getattr(model.config, "boi_token_id", 59256)
        else:
            self.def_image_token_id = 59256  # <|begin_of_image|>

    def update_chat_history_with_answer(self, answer: str):
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
        # Treat None, empty string, and NaN as "no video"
        import math
        _has_video = (
            video is not None
            and video != ""
            and not (isinstance(video, float) and math.isnan(video))
        )
        if _has_video:
            raise ValueError("Video input is not supported for GLM-Edge-V.")
        if audio is not None:
            raise ValueError("Audio input is not supported for GLM-Edge-V.")

        # For GLM, AutoProcessor resolves to the tokenizer; it carries the
        # chat_template that inserts 578 BOI placeholder tokens per image.
        tok = processor if processor is not None else tokenizer
        if tok is None:
            raise ValueError("Either processor or tokenizer must be provided.")

        # Normalize image input: load from path if string, convert to list
        if image is not None:
            if isinstance(image, (str, np.ndarray)):
                image = [image]
            if not isinstance(image, list):
                image = [image]
            # Filter out None/empty string values (empty video fields from CSV)
            image = [img for img in image if img is not None and img != ""]
            if len(image) == 0:
                image = None
            else:
                image = [_load_image(img) for img in image]

        self.update_images(image)

        # Build the chat message with content items
        content = []
        if image is not None:
            for _ in image:
                content.append({"type": "image"})
        content.append({"type": "text", "text": text})

        new_message = {"role": "user", "content": content}
        if self.chat_mode:
            self.chat_history.append(new_message)
            messages = self.chat_history
        else:
            messages = [new_message]

        prompt = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        input_ids = tok(prompt, return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.images is not None and len(self.images) > 0:
            image_size = self._IMAGE_SIZE
            if config is not None:
                image_size = getattr(
                    getattr(config, "vision_config", None), "image_size", image_size
                )

            img_proc = SiglipImageProcessor(
                size={"height": image_size, "width": image_size}
            )
            pixel_values_list = []
            for img in self.images:
                pv = img_proc(images=img, return_tensors="pt").pixel_values
                # pv: [1, C, H, W] -> [1, 1, C, H, W]
                pixel_values_list.append(pv.unsqueeze(0))

            # Concatenate along num_concurrent_media dim: [batch, N, 1, C, H, W]
            # GLM forward expects [batch, num_concurrent_media, num_tiles, C, H, W]
            pixel_values = torch.cat(pixel_values_list, dim=0)  # [N_imgs, 1, C, H, W]
            # Wrap in batch dim: [1, N_imgs, 1, C, H, W]
            pixel_values = pixel_values.unsqueeze(0)
            inputs["pixel_values"] = pixel_values
        else:
            # No image: pass a dummy all-zeros tensor so the model's default arg is used
            if config is not None:
                image_size = getattr(
                    getattr(config, "vision_config", None), "image_size", self._IMAGE_SIZE
                )
            else:
                image_size = self._IMAGE_SIZE
            inputs["pixel_values"] = torch.zeros(
                [1, 1, 1, 3, image_size, image_size], dtype=torch.float32
            )

        return inputs

