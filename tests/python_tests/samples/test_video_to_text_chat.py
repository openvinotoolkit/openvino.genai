# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import pytest
import subprocess  # nosec B404
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR
from test_utils import run_sample


class TestVisualLanguageChat:
    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.transformers_lower_v5(
        reason="llava-next-video hasn't supported by optimum-intel 423b423 with transformers>=5.0 yet"
    )
    @pytest.mark.parametrize(
        "convert_model, download_test_content, questions",
        [pytest.param("tiny-random-llava-next-video", "video0.mp4", "What is unusual on this video?\nGo on.")],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_visual_language_chat(self, convert_model, download_test_content, questions):
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, "video_to_text_chat")
        cpp_command = [cpp_sample, convert_model, download_test_content]
        cpp_result = run_sample(cpp_command, questions)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/video_to_text_chat.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command, questions)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"

    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, questions",
        [pytest.param("tiny-videochat-flash-qwen", "video0.mp4", "Describe this video.\nGo on.")],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_video_to_text_chat_videochat_flash(self, convert_model, download_test_content, questions):
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, "video_to_text_chat")
        cpp_command = [cpp_sample, convert_model, download_test_content]
        cpp_result = run_sample(cpp_command, questions)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/video_to_text_chat.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command, questions)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"

    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, questions",
        [pytest.param("tiny-videochat-flash-qwen", "video0.mp4", "Describe these videos.\nGo on.")],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_video_to_text_chat_videochat_flash_multiple_videos(
        self, convert_model, download_test_content, questions, tmp_path
    ):
        # Build a directory containing two copies of the video so the sample loads multiple videos.
        multi_video_dir = tmp_path / "multi_videos"
        multi_video_dir.mkdir()
        shutil.copy(download_test_content, multi_video_dir / "video0.mp4")
        shutil.copy(download_test_content, multi_video_dir / "video1.mp4")

        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, "video_to_text_chat")
        cpp_command = [cpp_sample, convert_model, str(multi_video_dir)]
        cpp_result = run_sample(cpp_command, questions)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/video_to_text_chat.py")
        py_command = [sys.executable, py_script, convert_model, str(multi_video_dir)]
        py_result = run_sample(py_command, questions)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
