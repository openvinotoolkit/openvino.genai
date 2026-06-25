# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

SPEECHT5_SPEAKER_EMB_SHAPE = (1, 512)
KOKORO_SPEAKER_EMB_SHAPE = (510, 1, 256)
DEFAULT_KOKORO_VOICE = "af_heart"


def is_kokoro_model_id(model_id_or_path):
    model_str = str(model_id_or_path)
    # Fast path: model ID or directory name contains "kokoro".
    if "kokoro" in model_str.lower():
        return True
    # Fallback for local exports whose directory name doesn't include "kokoro":
    # require a voices/ subdirectory containing at least one canonical voice file.
    model_path = Path(model_str)
    voices_dir = model_path / "voices"
    if model_path.is_dir() and voices_dir.is_dir():
        if (voices_dir / "af_heart.bin").exists() or (voices_dir / "af_heart.pt").exists():
            return True
    return False


def normalize_kokoro_lang_code(language):
    if not isinstance(language, str) or language.strip() == "":
        return "a"

    normalized = language.strip().lower()
    lang_map = {
        "en-us": "a",
        "en-gb": "b",
        "es": "e",
        "fr-fr": "f",
        "hi": "h",
        "it": "i",
        "pt-br": "p",
        "ja": "j",
        "zh": "z",
    }
    if normalized in lang_map:
        return lang_map[normalized]
    if len(normalized) == 1:
        return normalized

    raise ValueError(
        f"Unsupported Kokoro language '{language}'. Use one of: en-us, en-gb, es, fr-fr, hi, it, pt-br, ja, zh."
    )
