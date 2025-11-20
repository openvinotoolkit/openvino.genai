# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import subprocess # nosec B404
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR
from test_utils import run_sample

class TestVisualLanguageChat:
    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, questions",
        [
            pytest.param("tiny-random-llava-next-video", "videos/sample_video.mp4", 'What is unusual on this video?\nGo on.')
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_visual_language_chat(self, convert_model, download_test_content, questions):
        # Test CPP sample
        # TODO

        # Test C sample
        # TODO

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/video_to_text_chat.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command, questions)

        # Compare results
        # assert py_result.stdout == cpp_result.stdout, f"Results should match"
        # assert cpp_result.stdout == c_result.stdout, f"Results should match"
