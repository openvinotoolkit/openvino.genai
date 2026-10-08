# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf

from conftest import SAMPLES_CPP_DIR, SAMPLES_PY_DIR
from test_utils import run_sample

# Qwen3-Omni speech output is 24kHz mono.
SPEECH_SAMPLE_RATE = 24000
# The second turn passes no media, so it checks that speech output still works on a follow-up turn.
QUESTIONS = "Describe <ov_genai_audio_0>.\nWhat happens in the video?"
TURNS = 2
VIDEO_FRAMES = 8
VIDEO_SIZE = 128


@pytest.fixture(scope="module")
def mjpeg_video(tmp_path_factory: pytest.TempPathFactory) -> str:
    """An MJPEG AVI clip, the only video format OpenCV decodes without FFmpeg.

    The OpenCV that the C++ samples build has no FFmpeg in CI, so omni_chat can't open an .mp4 there.
    """
    path = tmp_path_factory.mktemp("omni_video") / "video.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (VIDEO_SIZE, VIDEO_SIZE)
    )
    assert writer.isOpened(), f"Could not create {path}"
    for frame_idx in range(VIDEO_FRAMES):
        writer.write(np.full((VIDEO_SIZE, VIDEO_SIZE, 3), frame_idx * 30, dtype=np.uint8))
    writer.release()
    return str(path)


def omni_chat_command(sample: str, model_dir: str, image: str, audio: str, video: str) -> list[str]:
    if sample == "cpp":
        return [str(SAMPLES_CPP_DIR / "omni_chat"), model_dir, image, audio, video]
    return [sys.executable, str(SAMPLES_PY_DIR / "omni/qwen3_omni_chat.py"), model_dir, image, video, "--audio", audio]


class TestOmniChat:
    @pytest.mark.qwen3_omni
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["tiny-random-qwen3-omni"], indirect=True)
    @pytest.mark.parametrize("download_test_content", [["monalisa.jpg", "how_are_you_doing_today.wav"]], indirect=True)
    @pytest.mark.parametrize("sample", ["cpp", "py"])
    def test_sample_omni_chat(
        self, convert_model: str, download_test_content: list[str], mjpeg_video: str, sample: str, tmp_path: Path
    ):
        image, audio = download_test_content
        # The sample writes output_audio_<turn>.wav to its working directory.
        run_sample(omni_chat_command(sample, convert_model, image, audio, mjpeg_video), QUESTIONS, cwd=str(tmp_path))

        for turn in range(TURNS):
            waveform, sample_rate = sf.read(tmp_path / f"output_audio_{turn}.wav")
            assert sample_rate == SPEECH_SAMPLE_RATE
            assert waveform.ndim == 1, "Speech output must be mono"
            assert waveform.size > 0, f"Turn {turn} produced an empty waveform"
