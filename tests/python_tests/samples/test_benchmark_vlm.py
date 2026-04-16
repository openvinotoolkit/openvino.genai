# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import subprocess  # nosec B404
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
        benchmark_sample = SAMPLES_CPP_DIR / 'benchmark_vlm'
        benchmark_cpp_command = [benchmark_sample, "-m" , convert_model, "-i", download_test_content, "-n", num_iter]
        run_sample(benchmark_cpp_command)
        
        # Run Python benchmark sample
        benchmark_script = SAMPLES_PY_DIR / 'visual_language_chat/benchmark_vlm.py'
        benchmark_py_command = [sys.executable, benchmark_script, "-m" , convert_model, "-i", download_test_content, "-n", num_iter]
        run_sample(benchmark_py_command)

    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content",
        [
            pytest.param("tiny-random-minicpmv-2_6", "images/image.png"),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_benchmark_vlm_with_resize(self, convert_model, download_test_content):
        num_iter = "1"
        image_height = "224"
        image_width = "224"

        # Run C++ benchmark sample with image resizing
        benchmark_sample = SAMPLES_CPP_DIR / "benchmark_vlm"
        benchmark_cpp_command = [
            benchmark_sample,
            "-m",
            convert_model,
            "-i",
            download_test_content,
            "-n",
            num_iter,
            "-H",
            image_height,
            "-W",
            image_width,
        ]
        run_sample(benchmark_cpp_command)

        # Run Python benchmark sample with image resizing
        benchmark_script = SAMPLES_PY_DIR / "visual_language_chat/benchmark_vlm.py"
        benchmark_py_command = [
            sys.executable,
            benchmark_script,
            "-m",
            convert_model,
            "-i",
            download_test_content,
            "-n",
            num_iter,
            "--image_height",
            image_height,
            "--image_width",
            image_width,
        ]
        run_sample(benchmark_py_command)

    @pytest.mark.vlm
    @pytest.mark.samples
    def test_sample_benchmark_vlm_negative_dimensions(self):
        # Validation of negative dimensions in the Python script raises RuntimeError
        # before any image or model is loaded, so no fixtures are required.
        benchmark_script = SAMPLES_PY_DIR / "visual_language_chat/benchmark_vlm.py"

        with pytest.raises(subprocess.CalledProcessError):
            run_sample(
                [
                    sys.executable,
                    benchmark_script,
                    "-m",
                    "fake_model",
                    "-i",
                    "fake_image.jpg",
                    "--image_height",
                    "-1",
                    "--image_width",
                    "224",
                ]
            )

        with pytest.raises(subprocess.CalledProcessError):
            run_sample(
                [
                    sys.executable,
                    benchmark_script,
                    "-m",
                    "fake_model",
                    "-i",
                    "fake_image.jpg",
                    "--image_height",
                    "224",
                    "--image_width",
                    "-1",
                ]
            )
