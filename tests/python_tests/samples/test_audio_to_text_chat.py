# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from conftest import SAMPLES_CPP_DIR, SAMPLES_PY_DIR
from test_utils import run_sample


class TestAudioToTextChat:
    @pytest.mark.vlm
    @pytest.mark.vlm_audio
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, questions",
        [
            pytest.param(
                "tiny-random-gemma4",
                "how_are_you_doing_today.wav",
                "What is said in this recording?\nWhat was my question?",
            ),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_audio_to_text_chat(self, convert_model, download_test_content, questions):
        cpp_command = [SAMPLES_CPP_DIR / "audio_to_text_chat", convert_model, download_test_content]
        cpp_result = run_sample(cpp_command, questions)

        py_script = SAMPLES_PY_DIR / "visual_language_chat/audio_to_text_chat.py"
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command, questions)

        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
