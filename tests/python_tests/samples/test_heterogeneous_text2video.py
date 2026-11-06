# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import cv2
import pytest

from conftest import SAMPLES_CPP_DIR, SAMPLES_PY_DIR
from test_utils import compare_videos, run_sample


@pytest.mark.samples
@pytest.mark.video_generation
@pytest.mark.parametrize("convert_model", ["tiny-random-ltx-video"], indirect=True)
def test_sample_heterogeneous_text2video(convert_model, tmp_path):
    prompt = "A cat walks through grass in the morning."
    num_frames = 9
    videos = []
    commands = [
        [sys.executable, SAMPLES_PY_DIR / "video_generation/heterogeneous_text2video.py"],
        [SAMPLES_CPP_DIR / "heterogeneous_text2video"],
    ]
    for name, command in zip(("python", "cpp"), commands):
        output_dir = tmp_path / name
        output_dir.mkdir()
        run_sample(command + [convert_model, prompt, "CPU", "CPU", "CPU", str(num_frames)], cwd=str(output_dir))
        video = output_dir / "video_0.avi"
        assert video.exists(), f"Video not found: {video}"
        capture = cv2.VideoCapture(str(video))
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == num_frames
            assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 704
            assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 480
            assert capture.get(cv2.CAP_PROP_FPS) == 25
        finally:
            capture.release()
        videos.append(video)

    assert compare_videos(*videos), "Videos from Python and C++ heterogeneous samples differ"
