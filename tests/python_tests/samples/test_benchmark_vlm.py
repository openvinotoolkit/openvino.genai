# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestBenchmarkVLM:
    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content",
        [
            pytest.param("tiny-random-minicpmv-2_6", "images/image.png"),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_benchmark_vlm(self, convert_model, download_test_content):
        num_iter = "3"
        # Run C++ benchmark sample
        benchmark_sample = os.path.join(SAMPLES_CPP_DIR, 'benchmark_vlm')
        benchmark_cpp_command = [benchmark_sample, "-m" , convert_model, "-i", download_test_content, "-n", num_iter]
        run_sample(benchmark_cpp_command)
        
        # Run Python benchmark sample
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'visual_language_chat/benchmark_vlm.py')
        benchmark_py_command = [sys.executable, benchmark_script, "-m" , convert_model, "-i", download_test_content, "-n", num_iter]
        run_sample(benchmark_py_command)