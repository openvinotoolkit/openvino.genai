# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

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
    @pytest.mark.parametrize("download_test_content", ["image.png"], indirect=True)
    def test_sample_image2image(self, download_model, prompt, download_test_content):
        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "image_generation/image2image.py")
        py_command = [sys.executable, py_script, download_model, prompt, download_test_content]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'image2image')
        cpp_command = [cpp_sample, download_model, prompt, download_test_content]
        run_sample(cpp_command)
        
        # Run C++ benchmark sample
        benchmark_sample = os.path.join(SAMPLES_CPP_DIR, 'benchmark_image_gen')
        benchmark_cpp_command = [benchmark_sample, "-t"] + cpp_command
        run_sample(benchmark_cpp_command)
        
        # Run Python benchmark sample
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'image_generation/benchmark_image_gen')
        benchmark_py_command = [sys.executable, benchmark_script, "-t"] + cpp_command
        run_sample(benchmark_py_command)