# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import TEST_FILES, SAMPLES_PY_DIR, SAMPLES_CPP_DIR

class TestWhisperSpeechRecognition:
    @pytest.mark.whisper
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", [TEST_FILES["how_are_you_doing_today.wav"]], indirect=True)
    def test_python_sample_whisper_speech_recognition(self, convert_model, download_test_content):
        script = os.path.join(SAMPLES_PY_DIR, "whisper_speech_recognition/whisper_speech_recognition.py")
        subprocess.run(["python", script, convert_model, download_test_content], check=True)

    @pytest.mark.whisper
    @pytest.mark.parametrize("convert_model", ["WhisperTiny"], indirect=True)
    @pytest.mark.parametrize("download_test_content", [TEST_FILES["how_are_you_doing_today.wav"]], indirect=True)
    def test_cpp_sample_whisper_speech_recognition(self, convert_model, download_test_content):
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'whisper_speech_recognition')
        subprocess.run([cpp_sample, convert_model, download_test_content], check=True)