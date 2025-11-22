# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, download_test_content
from test_utils import run_sample

download_mask_image = download_test_content

class TestInpainting:
    PROMPT = "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"
    IMAGE_PATH = "images/image.png"
    MASK_PATH = "mask_image.png"
    
    @pytest.mark.samples
    @pytest.mark.LCM_Dreamshaper_v7_int8_ov
    @pytest.mark.parametrize(
        "download_model, prompt",
        [
            pytest.param("LCM_Dreamshaper_v7-int8-ov", PROMPT),
        ],
        indirect=["download_model"],
    )
    @pytest.mark.parametrize(
        "download_test_content, download_mask_image",
        [
            pytest.param(IMAGE_PATH, MASK_PATH),
        ],
        indirect=["download_test_content", "download_mask_image"],
    )
    def test_sample_inpainting(self, download_model, prompt, download_test_content, download_mask_image):
        # Run Python sample
        py_script = SAMPLES_PY_DIR / "image_generation/inpainting.py"
        py_command = [sys.executable, py_script, download_model, "'" + prompt + "'", download_test_content, download_mask_image]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / 'inpainting'
        cpp_command = [cpp_sample, download_model, "'" + prompt + "'", download_test_content, download_mask_image]
        run_sample(cpp_command)