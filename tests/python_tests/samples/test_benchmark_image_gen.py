# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys
import subprocess # nosec B404

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, download_test_content
from test_utils import run_sample

download_mask_image = download_test_content

class TestBenchmarkImageGen:
    @pytest.mark.samples
    @pytest.mark.image_generation
    @pytest.mark.parametrize(
        "download_model, prompt",
        [
            pytest.param("LCM_Dreamshaper_v7-int8-ov", "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"),
        ], indirect=["download_model"],
    )
    @pytest.mark.parametrize("pipeline_type", ["text2image", "image2image", "inpainting"])
    @pytest.mark.parametrize(
        "download_test_content, download_mask_image",
        [
            pytest.param("images/image.png", "mask_image.png"),
        ], indirect=["download_test_content", "download_mask_image"],
    )
    def test_sample_benchmark_image_gen(self, download_model, pipeline_type, prompt, download_test_content, download_mask_image):
        inference_steps = "3"
        # Run C++ benchmark sample
        benchmark_sample = os.path.join(SAMPLES_CPP_DIR, 'benchmark_image_gen')
        benchmark_cpp_command = [benchmark_sample, "-t", pipeline_type, "-m" , download_model, "-p", "'" + prompt + "'", "-i", download_test_content, "--mi", download_mask_image, "--is", inference_steps]
        run_sample(benchmark_cpp_command)
        
        # Run Python benchmark sample
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'image_generation/benchmark_image_gen.py')
        benchmark_py_command = [sys.executable, benchmark_script, "-t", pipeline_type, "-m" , download_model, "-p", "'" + prompt + "'", "-i", download_test_content, "-mi", download_mask_image, "-is", inference_steps]
        run_sample(benchmark_py_command)