# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from .qwen2 import Qwen2VLInputsPreprocessor


class Lfm2VLInputsPreprocessor(Qwen2VLInputsPreprocessor):
    """LFM2-VL / LFM2.5-VL use the same image/video conversation interface as Qwen2-VL.

    Its chat template renders ``{"type": "image"}`` content items as ``<image>``
    placeholders, and ``Lfm2VlProcessor.__call__`` accepts ``images``/``text``
    (tolerating ``videos=None``), producing ``pixel_values`` and
    ``pixel_attention_mask``. The only model-specific value is the image token id.
    """

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode, model=model)
        self.def_image_token_id = getattr(model.config, "image_token_id", 124907) if model is not None else 124907
