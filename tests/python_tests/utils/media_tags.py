# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Universal media tag helpers shared by the VLM and Omni pipeline tests.

`<ov_genai_image_N>` / `<ov_genai_video_N>` / `<ov_genai_audio_N>` are the backend-independent
tags GenAI resolves to each model's native placeholders, so the tests that exercise them span
more than one pipeline test module.
"""

from __future__ import annotations

from enum import Enum

import openvino


class ModalityType(Enum):
    # Values stay UPPERCASE: `.value` is interpolated into parametrize ids, so lowercasing would
    # rewrite every existing test id and break CI filters and xfail selectors.
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"


def get_media_inputs_kwargs(media: list[openvino.Tensor], modality_type: ModalityType) -> dict:
    if modality_type == ModalityType.IMAGE:
        return {"images": media}
    elif modality_type == ModalityType.VIDEO:
        return {"videos": media}
    else:
        return {"audios": media}


def get_universal_tag(modality_type: ModalityType, index: int) -> str:
    if modality_type == ModalityType.IMAGE:
        return f"<ov_genai_image_{index}>"
    elif modality_type == ModalityType.VIDEO:
        return f"<ov_genai_video_{index}>"
    else:
        return f"<ov_genai_audio_{index}>"
