# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

import numpy as np

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample, compare_videos


class TestText2Video:
    PROMPT = "A woman with long brown hair smiles at another woman with long blonde hair"

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize("convert_model", ["tiny-random-ltx2.3"], indirect=True)
    @pytest.mark.parametrize("static_shapes", [False, True])
    def test_sample_heterogeneous_ltx2_3(self, convert_model, static_shapes, tmp_path):
        import cv2
        import soundfile as sf

        command = [
            sys.executable,
            SAMPLES_PY_DIR / "video_generation/heterogeneous_ltx2_3_text2video.py",
            convert_model,
            self.PROMPT,
            "CPU",
            "CPU",
            "CPU",
            "--inference-precision",
            "f32",
            "--width",
            "32",
            "--height",
            "32",
            "--num-frames",
            "9",
            "--num-inference-steps",
            "2",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
        if static_shapes:
            command.append("--reshape")
        run_sample(command, cwd=str(tmp_path))

        capture = cv2.VideoCapture(str(tmp_path / "genai_video.avi"))
        try:
            assert capture.isOpened()
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            assert len(frames) == 9
            assert all(frame.shape == (32, 32, 3) for frame in frames)
        finally:
            capture.release()
        audio, sample_rate = sf.read(tmp_path / "genai_audio.wav", dtype="float32", always_2d=True)
        assert sample_rate > 0
        assert audio.shape[0] > 0 and audio.shape[1] == 2
        assert np.isfinite(audio).all()

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize(
        "convert_model, sample_args, has_audio",
        [
            pytest.param("tiny-random-ltx-video", PROMPT, False),
            pytest.param("tiny-random-ltx2", PROMPT, True),
            pytest.param("tiny-random-ltx2.3", PROMPT, True),
        ],
        indirect=["convert_model"],
    )
    def test_sample_text2video(self, convert_model, sample_args, has_audio, tmp_path):
        py_dir = tmp_path / "python_output"
        cpp_dir = tmp_path / "cpp_output"
        py_dir.mkdir()
        cpp_dir.mkdir()

        py_script = SAMPLES_PY_DIR / "video_generation/text2video.py"
        py_command = [sys.executable, py_script, convert_model, sample_args, "5"]
        run_sample(py_command, cwd=str(py_dir))

        cpp_sample = SAMPLES_CPP_DIR / "text2video"
        cpp_command = [cpp_sample, convert_model, sample_args, "5"]
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
