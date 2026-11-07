from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from .vlm_inputs_preprocessor import VLMInputsPreprocessor

if TYPE_CHECKING:
    from PIL.Image import Image
    from transformers.image_utils import VideoInput


class Molmo2InputsPreprocessor(VLMInputsPreprocessor):
    """Inputs preprocessor for Molmo2 (Molmo2ForConditionalGeneration).

    Molmo2 ships its own ``Molmo2Processor`` (a ``ProcessorMixin`` bundling an
    image processor and a tokenizer) and a chat template that inserts the
    ``<|image|>`` placeholder for each image in the conversation. The processor
    then expands that placeholder into the model-specific image token layout and
    produces ``pixel_values`` together with the auxiliary image tensors
    (``image_grids``, ``image_num_crops``, ...). We therefore rely on the
    processor's own ``apply_chat_template`` to build a fully consistent set of
    inputs instead of assembling image tokens manually.
    """

    def __init__(self, chat_mode: bool = False, model: Any | None = None):
        super().__init__(chat_mode)
        # Token id emitted for image patches in the tokenized sequence.
        default_patch_id = 151938
        if model is not None:
            self.def_image_token_id = getattr(model.config, "image_patch_id", default_patch_id)
        else:
            self.def_image_token_id = default_patch_id

    def update_chat_history_with_answer(self, answer):
        self.chat_history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    def preprocess_inputs(
        self,
        text: str,
        image: Union["Image", list["Image"]] | None = None,
        processor: AutoImageProcessor | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        config: PretrainedConfig | None = None,
        video: Union["VideoInput", list["VideoInput"]] | None = None,
        audio: np.ndarray | None = None,
    ):
        if processor is None:
            raise ValueError("Processor is required.")
        if audio is not None:
            raise ValueError("Audio input is not supported")

        self.update_images(image)

        content = []
        if image is not None:
            images = image if isinstance(image, list) else [image]
            content.extend([{"type": "image", "image": img} for img in images])
        if video is not None:
            content.append({"type": "video", "video": "placeholder"})
        content.append({"type": "text", "text": text})

        if video is not None:
            if self.chat_mode:
                raise ValueError("Video input is not supported in chat mode")
            # The video comes as already decoded frames without metadata. Like OpenVINO GenAI, treat the frames as
            # sampled at the model's sampling fps, then let the processor apply Molmo2 frame sampling and timestamps.
            from transformers.video_utils import VideoMetadata

            frames = np.asarray(video)
            fps = float(getattr(processor.video_processor, "sampling_fps", 2) or 2)
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            return processor(
                text=[prompt],
                images=image,
                videos=[frames],
                video_metadata=[VideoMetadata(total_num_frames=len(frames), fps=fps)],
                return_tensors="pt",
            )

        if self.chat_mode:
            self.chat_history.append({"role": "user", "content": content})
            conversation = self.chat_history
        else:
            conversation = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs

    def align_inputs_with_cache(self, model: Any, inputs: dict, full_tokenized_chat: torch.Tensor, prefix_len: int):
        if "pixel_values" not in inputs:
            return inputs

        image_token_id = getattr(model.config, "image_patch_id", self.def_image_token_id)
        new_image_tokens = full_tokenized_chat[0].tolist()[prefix_len:].count(image_token_id)

        # Every image contributes image_num_crops[i] crops and one "<im_patch>" token per pooled patch: the
        # low-resolution grid plus the high-resolution grid of image_grids[i] = (resized_h, resized_w, h, w).
        grids = inputs["image_grids"]
        num_crops = inputs["image_num_crops"]
        pooled = (grids[:, 0] * grids[:, 1] + grids[:, 2] * grids[:, 3]).tolist()
        total_images = len(pooled)

        # The images whose tokens are not cached yet are the last ones of the conversation.
        new_images, tokens = 0, 0
        while new_images < total_images and tokens < new_image_tokens:
            new_images += 1
            tokens += pooled[total_images - new_images]
        if tokens != new_image_tokens:
            raise ValueError(
                f"Cannot align Molmo2 image inputs with the cached prefix: {new_image_tokens} new image tokens do not "
                f"correspond to whole images (image token counts: {pooled})"
            )
        if new_images == total_images:
            return inputs
        if new_images == 0:
            for key in ("pixel_values", "image_token_pooling", "image_grids", "image_num_crops"):
                inputs.pop(key, None)
            return inputs

        new_crops = int(num_crops[total_images - new_images :].sum())
        new_pooled = sum(pooled[total_images - new_images :])
        inputs["pixel_values"] = inputs["pixel_values"][-new_crops:]
        inputs["image_token_pooling"] = inputs["image_token_pooling"][-new_pooled:]
        inputs["image_grids"] = grids[total_images - new_images :]
        inputs["image_num_crops"] = num_crops[total_images - new_images :]
        return inputs
