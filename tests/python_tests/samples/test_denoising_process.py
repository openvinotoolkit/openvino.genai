# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

# Must match the constants hardcoded in the denoising_process samples.
NUM_INFERENCE_STEPS = 20
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
CHANNELS = 3
EXPECTED_VIDEO_SHAPE = (NUM_INFERENCE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)


def _read_video_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened(), f"OpenCV failed to open {path}"
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    assert frames, f"Could not read any frames from {path}"
    return np.stack(frames)


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

        # Both samples write MJPG AVIs via cv::CAP_OPENCV_MJPEG, so the OpenCV-built-in
        # encoder produces reproducible frames independent of any system FFmpeg/GStreamer.
        py_frames = _read_video_frames(py_video)
        cpp_frames = _read_video_frames(cpp_video)
        assert py_frames.shape == EXPECTED_VIDEO_SHAPE, (
            f"Python AVI shape {py_frames.shape}, expected {EXPECTED_VIDEO_SHAPE}"
        )
        assert cpp_frames.shape == EXPECTED_VIDEO_SHAPE, (
            f"C++ AVI shape {cpp_frames.shape}, expected {EXPECTED_VIDEO_SHAPE}"
        )
        assert py_frames.dtype == np.uint8
        assert cpp_frames.dtype == np.uint8
        assert np.array_equal(py_frames, cpp_frames), (
            "Decoded denoising-process video frames produced by the Python and C++ samples are not exactly equal"
        )
