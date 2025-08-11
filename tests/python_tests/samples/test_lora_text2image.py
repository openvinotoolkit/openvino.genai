# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestLoraText2Image:
    @pytest.mark.samples
    @pytest.mark.dreamlike_anime_1_0
    @pytest.mark.parametrize(
        "convert_model, prompt, sample_args",
        [
            pytest.param("dreamlike-anime-1.0", "curly-haired unicorn in the forest, anime, line", "0.7"),
        ],
        indirect=["convert_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["soulcard.safetensors"], indirect=True)
    @pytest.mark.parametrize("executable", [
        [SAMPLES_CPP_DIR / 'lora_text2image'],
        [sys.executable, SAMPLES_PY_DIR / "image_generation/lora_text2image.py"],
    ])
    def test_sample_lora_text2image(self, convert_model, prompt, download_test_content, sample_args, executable):
        run_sample(executable + [convert_model, prompt, download_test_content, sample_args])
