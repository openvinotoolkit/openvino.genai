# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestImage2Image:
    @pytest.mark.samples
    @pytest.mark.LCM_Dreamshaper_v7_int8_ov
    @pytest.mark.parametrize("executable", [
        [SAMPLES_CPP_DIR / "image2image"],
        [sys.executable, SAMPLES_PY_DIR / "image_generation/image2image.py"],
    ])
    @pytest.mark.parametrize(
        "download_model, prompt",
        [
            pytest.param("LCM_Dreamshaper_v7-int8-ov", "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"),
        ],
        indirect=["download_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["images/image.png"], indirect=True)
    def test_sample_image2image(self, executable, download_model, prompt, download_test_content):
        run_sample(executable + [download_model, prompt, download_test_content])


    @pytest.mark.samples
    @pytest.mark.LCM_Dreamshaper_v7_int8_ov
    @pytest.mark.parametrize(
        "download_model, prompts",
        [
            pytest.param("LCM_Dreamshaper_v7-int8-ov",
                ("cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting",
                 "village landscape with mountains and a river at sunrise",
                 "midcentury modern house with a garden and a pond at sunset")),
        ],
        indirect=["download_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["images/image.png"], indirect=True)
    def test_sample_image2image_concurrency(self, download_model, prompts, download_test_content):
        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / "image2image_concurrency"
        cpp_command = [cpp_sample, download_model, *prompts, download_test_content]
        run_sample(cpp_command)

