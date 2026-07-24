# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample, compare_videos


class TestText2Video:
    PROMPT = "A woman with long brown hair smiles at another woman with long blonde hair"

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-ltx-video", PROMPT),
        ],
        indirect=["convert_model"],
    )
    def test_sample_text2video(self, convert_model, sample_args, tmp_path):
        py_dir = tmp_path / "python_output"
        cpp_dir = tmp_path / "cpp_output"
        py_dir.mkdir()
        cpp_dir.mkdir()

        py_script = SAMPLES_PY_DIR / "video_generation/text2video.py"
        py_command = [sys.executable, py_script, convert_model, sample_args, "--num-frames", "5"]
        run_sample(py_command, cwd=str(py_dir))

        cpp_sample = SAMPLES_CPP_DIR / "text2video"
        cpp_command = [cpp_sample, convert_model, sample_args, "--num-frames", "5"]
        run_sample(cpp_command, cwd=str(cpp_dir))

        py_video = py_dir / "genai_video.avi"
        cpp_video = cpp_dir / "genai_video.avi"

        assert py_video.exists(), f"Python video not found: {py_video}"
        assert cpp_video.exists(), f"C++ video not found: {cpp_video}"
        assert compare_videos(py_video, cpp_video), "Videos from Python and C++ samples are not identical"
