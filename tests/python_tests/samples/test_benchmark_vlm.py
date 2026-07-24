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
            pytest.param("tiny-random-phi3-vision", "images/image.png"),
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
    @pytest.mark.parametrize(
        "download_test_content",
        [pytest.param("images/image.png")],
        indirect=["download_test_content"],
    )
    def test_sample_benchmark_vlm_invalid_resize(self, download_test_content):
        # Validation of invalid resize dimensions causes the C++ and Python sample subprocess to fail
        # before any model is loaded, so this test observes CalledProcessError.
        benchmark_sample = SAMPLES_CPP_DIR / "benchmark_vlm"
        benchmark_script = SAMPLES_PY_DIR / "visual_language_chat/benchmark_vlm.py"

        invalid_sizes = [(-1, 224), (224, -1), (0, 224), (224, 0), (0, 0), (None, 224), (224, None)]

        for height, width in invalid_sizes:
            cpp_command = [benchmark_sample, "-m", "fake_model", "-i", download_test_content]
            py_command = [
                sys.executable,
                benchmark_script,
                "-m",
                "fake_model",
                "-i",
                "fake_image.jpg",
            ]

            if height is not None:
                cpp_command.extend(["-H", str(height)])
                py_command.extend(["--image_height", str(height)])

            if width is not None:
                cpp_command.extend(["-W", str(width)])
                py_command.extend(["--image_width", str(width)])

            for command in [cpp_command, py_command]:
                with pytest.raises(subprocess.CalledProcessError):
                    run_sample(command)

    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.eagle3_decoding
    @pytest.mark.parametrize(
        "convert_model, convert_draft_model, download_test_content",
        [
            pytest.param("tiny-random-qwen3-vl-layer10", "tiny-random-qwen3-vl-eagle3", "images/image.png"),
        ],
        indirect=["convert_model", "convert_draft_model", "download_test_content"],
    )
    def test_benchmark_eagle3_vlm(self, convert_model, convert_draft_model, download_test_content):
        num_iter = "3"
        num_assistant_tokens = "5"

        benchmark_sample = SAMPLES_CPP_DIR / "benchmark_vlm"
        benchmark_cpp_command = [
            benchmark_sample,
            "-m",
            convert_model,
            "-D",
            convert_draft_model,
            "-A",
            num_assistant_tokens,
            "-i",
            download_test_content,
            "-n",
            num_iter,
        ]
        run_sample(benchmark_cpp_command)

        benchmark_script = SAMPLES_PY_DIR / "visual_language_chat/benchmark_vlm.py"
        benchmark_py_command = [
            sys.executable,
            benchmark_script,
            "-m",
            convert_model,
            "-D",
            convert_draft_model,
            "-A",
            num_assistant_tokens,
            "-i",
            download_test_content,
            "-n",
            num_iter,
        ]
        run_sample(benchmark_py_command)
