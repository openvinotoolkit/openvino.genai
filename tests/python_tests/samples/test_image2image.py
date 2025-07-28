# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys
import subprocess

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestImage2Image:
    @pytest.mark.samples
    @pytest.mark.image_generation
    @pytest.mark.parametrize(
        "download_model, prompt",
        [
            pytest.param("LCM_Dreamshaper_v7-int8-ov", "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"),
        ],
        indirect=["download_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["images/image.png"], indirect=True)
    @pytest.mark.xfail(reason="Failed to open model_index.json. Ticket 171245", raises=subprocess.CalledProcessError)
    def test_sample_image2image(self, download_model, prompt, download_test_content):
        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "image_generation/image2image.py")
        py_command = [sys.executable, py_script, download_model, prompt, download_test_content]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'image2image')
        cpp_command = [cpp_sample, download_model, prompt, download_test_content]
        run_sample(cpp_command)


    @pytest.mark.samples
    @pytest.mark.image_generation
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
    @pytest.mark.xfail(reason="Failed to open model_index.json. Ticket 171245", raises=subprocess.CalledProcessError)
    def test_sample_image2image_concurrency(self, download_model, prompts, download_test_content):
        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'image2image_concurrency')
        cpp_command = [cpp_sample, download_model, *prompts, download_test_content]
        run_sample(cpp_command)

