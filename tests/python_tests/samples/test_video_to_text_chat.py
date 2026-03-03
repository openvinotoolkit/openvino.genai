# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import subprocess  # nosec B404
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR, SAMPLES_JS_DIR
from test_utils import run_sample, run_js_chat


class TestVisualLanguageChat:
    @pytest.mark.vlm
    @pytest.mark.samples
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

        # Test JavaScript sample
        js_script = os.path.join(SAMPLES_JS_DIR, "visual_language_chat/video_to_text_chat.js")
        js_command = ["node", js_script, convert_model, download_test_content]
        js_stdout = run_js_chat(js_command, questions)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
        assert py_result.stdout == js_stdout, f"JS results should match with Python results"
