# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample, compare_videos


class TestDenoisingProcess:
    PROMPT = "cyberpunk cityscape with neon lights"

    @pytest.mark.samples
    @pytest.mark.tiny_random_latent_consistency
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-latent-consistency", PROMPT),
        ],
        indirect=["convert_model"],
    )
    def test_sample_denoising_process(self, convert_model, sample_args, tmp_path):
        py_script = SAMPLES_PY_DIR / "image_generation" / "denoising_process.py"
        cpp_sample = SAMPLES_CPP_DIR / "denoising_process"

        py_cwd = tmp_path / "py"
        cpp_cwd = tmp_path / "cpp"
        py_cwd.mkdir()
        cpp_cwd.mkdir()

        run_sample([sys.executable, str(py_script), convert_model, sample_args], cwd=str(py_cwd))
        run_sample([str(cpp_sample), convert_model, sample_args], cwd=str(cpp_cwd))

        py_video = py_cwd / "denoising_process.avi"
        cpp_video = cpp_cwd / "denoising_process.avi"

        assert py_video.exists(), f"Python sample did not produce {py_video}"
        assert cpp_video.exists(), f"C++ sample did not produce {cpp_video}"

        # Both samples write MJPG AVIs via cv::CAP_OPENCV_MJPEG, so the OpenCV
        # built-in encoder produces reproducible frames independent of any
        # system FFmpeg/GStreamer backend.
        assert compare_videos(py_video, cpp_video), (
            "Decoded denoising-process video frames produced by the Python and C++ samples are not exactly equal"
        )
