# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR
from test_utils import run_sample

class TestWhisperSpeechRecognition:
    @pytest.mark.whisper
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["how_are_you_doing_today.wav"], indirect=True)
    def test_sample_whisper_speech_recognition(self, convert_model, download_test_content):
        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'whisper_speech_recognition')
        cpp_command = [cpp_sample, convert_model, download_test_content]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "whisper_speech_recognition/whisper_speech_recognition.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command)

        # Run C sample
        c_sample = os.path.join(SAMPLES_C_DIR, 'whisper_speech_recognition_c')
        c_command = [c_sample, convert_model, download_test_content]
        c_result = run_sample(c_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        assert py_result.stdout == c_result.stdout, "Python and C results should match"
