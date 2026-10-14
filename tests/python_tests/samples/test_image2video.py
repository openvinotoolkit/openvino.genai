# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

import numpy as np

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample, compare_videos


class TestImage2Video:
    PROMPT = "A golden retriever in a sunlit bedroom slowly stands up and turns its head to look at the camera"

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize(
        "convert_model, sample_args, has_audio",
        [
            pytest.param("tiny-random-ltx-video", PROMPT, False),
            pytest.param("tiny-random-ltx2", PROMPT, True),
        ],
        indirect=["convert_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["overture-creations.png"], indirect=True)
    def test_sample_image2video(self, convert_model, sample_args, has_audio, download_test_content, tmp_path):
        py_dir = tmp_path / "python_output"
        cpp_dir = tmp_path / "cpp_output"
        py_dir.mkdir()
        cpp_dir.mkdir()

        py_script = SAMPLES_PY_DIR / "video_generation/image2video.py"
        py_command = [sys.executable, py_script, convert_model, download_test_content, sample_args, "5"]
        run_sample(py_command, cwd=str(py_dir))

        cpp_sample = SAMPLES_CPP_DIR / "image2video"
        cpp_command = [cpp_sample, convert_model, download_test_content, sample_args, "5"]
        run_sample(cpp_command, cwd=str(cpp_dir))

        py_video = py_dir / "genai_video.avi"
        cpp_video = cpp_dir / "genai_video.avi"

        assert py_video.exists(), f"Python video not found: {py_video}"
        assert cpp_video.exists(), f"C++ video not found: {cpp_video}"
        assert compare_videos(py_video, cpp_video), "Videos from Python and C++ samples are not identical"

        py_audio = py_dir / "genai_audio.wav"
        cpp_audio = cpp_dir / "genai_audio.wav"
        assert py_audio.exists() == has_audio, f"Unexpected audio output presence: {py_audio}"
        assert cpp_audio.exists() == has_audio, f"Unexpected audio output presence: {cpp_audio}"
        if has_audio:
            import soundfile as sf

            py_data, py_rate = sf.read(py_audio, dtype="float32")
            cpp_data, cpp_rate = sf.read(cpp_audio, dtype="float32")
            assert py_rate == cpp_rate, "Audio sample rates from Python and C++ samples differ"
            assert np.array_equal(py_data, cpp_data), "Audio from Python and C++ samples is not identical"
