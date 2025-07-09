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
    def test_py_sample_whisper_speech_recognition(self, convert_model, download_test_content):           
        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "whisper_speech_recognition/whisper_speech_recognition.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        run_sample(py_command)

    @pytest.mark.whisper
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["how_are_you_doing_today.wav"], indirect=True)
    def test_cpp_sample_whisper_speech_recognition(self, convert_model, download_test_content):
        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'whisper_speech_recognition')
        cpp_command = [cpp_sample, convert_model, download_test_content]
        run_sample(cpp_command)

    @pytest.mark.whisper
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", ["how_are_you_doing_today.wav"], indirect=True)
    def test_c_sample_whisper_speech_recognition(self, convert_model, download_test_content):
        # Run C sample
        c_sample = os.path.join(SAMPLES_C_DIR, 'whisper_speech_recognition')
        c_command = [c_sample, '-m', convert_model, '-i', download_test_content]
        run_sample(c_command)
