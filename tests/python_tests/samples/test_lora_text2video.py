# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample


class TestLoraText2Video:
    PROMPT = "A woman with long brown hair smiles at another woman with long blonde hair"

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-ltx-video", PROMPT),
        ],
        indirect=["convert_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["ltx_tiny_dummy_lora.safetensors"], indirect=True)
    @pytest.mark.parametrize("executable", [
        [SAMPLES_CPP_DIR / "lora_text2video"],
        [sys.executable, SAMPLES_PY_DIR / "video_generation/lora_text2video.py"],
    ])
    def test_sample_lora_text2video(self, convert_model, sample_args, download_test_content, executable):
        run_sample(executable + [convert_model, sample_args, download_test_content, "0.7"])
