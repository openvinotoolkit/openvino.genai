# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import re

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR
from test_utils import run_sample


def filter_word_level_timestamps(text: str) -> str:
    """
    example:
     How are you doing today?
    timestamps: [0.00, 2.00] text:  How are you doing today?
    [0.00, 0.58]:  How
    [0.58, 0.70]:  are
    [0.70, 0.80]:  you
    [0.80, 1.06]:  doing
    [1.06, 1.40]:  today?
    """
    pattern = r"\[\d+\.\d{2}, \d+\.\d{2}\]:\s+\S+"
    filtered_text = re.sub(pattern, "", text)
    return filtered_text


class TestWhisperSpeechRecognition:
    @pytest.mark.whisper
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["how_are_you_doing_today.wav"], indirect=True)
    def test_sample_whisper_speech_recognition(self, convert_model, download_test_content):
        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / "whisper_speech_recognition"
        cpp_command = [cpp_sample, convert_model, download_test_content]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = SAMPLES_PY_DIR / "whisper_speech_recognition/whisper_speech_recognition.py"
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command)

        # Run C sample
        c_sample = SAMPLES_C_DIR / "whisper_speech_recognition_c"
        c_command = [c_sample, convert_model, download_test_content]
        c_result = run_sample(c_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        # C API has no word-level timestamps support which enabled in Python and CPP samples
        # ticket to enable C API: 180115
        assert filter_word_level_timestamps(py_result.stdout) == c_result.stdout, (
            "Python and C results should match without word-level timestamps"
        )
