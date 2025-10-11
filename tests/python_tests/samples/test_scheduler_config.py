# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestSchedulerConfig:
    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content",
        [
            pytest.param("tiny-random-minicpmv-2_6", "images/image.png"),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_scheduler_config(self, convert_model, download_test_content):
        env = os.environ.copy()
        env["OPENVINO_LOG_LEVEL"] = "5"

        num_iter = "1"
        # Run C++ benchmark sample
        benchmark_sample = os.path.join(SAMPLES_CPP_DIR, 'benchmark_vlm')
        benchmark_cpp_command = [benchmark_sample, "-m" , convert_model, "-i", download_test_content, "-n", num_iter]
        cpp_result = run_sample(benchmark_cpp_command, env=env)

        assert cpp_result.stdout.find("SchedulerConfig {") != -1, f"Should print SchedulerConfig info in CPP"

        # Run Python benchmark sample
        benchmark_script = os.path.join(SAMPLES_PY_DIR, 'visual_language_chat/benchmark_vlm.py')
        benchmark_py_command = [sys.executable, benchmark_script, "-m" , convert_model, "-i", download_test_content, "-n", num_iter]
        py_result = run_sample(benchmark_py_command, env=env)

        assert py_result.stdout.find("SchedulerConfig {") != -1, f"Should print SchedulerConfig info in PYTHON"