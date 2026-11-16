# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from .qwen2 import Qwen2VLInputsPreprocessor


class Lfm3VLInputsPreprocessor(Qwen2VLInputsPreprocessor):
    """LFM3-VL uses the same image/video conversation interface as Qwen2-VL."""

    def __init__(self, chat_mode: bool = False, model: Optional[Any] = None):
        super().__init__(chat_mode, model=model)
        self.def_image_token_id = getattr(model.config, "image_token_id", 125030) if model is not None else 125030
