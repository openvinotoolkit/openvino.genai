# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR, SAMPLES_JS_DIR
from test_utils import run_sample

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

        # Run JS sample
        js_sample = SAMPLES_JS_DIR / "whisper_speech_recognition/whisper_speech_recognition.js"
        js_command = ["node", js_sample, convert_model, download_test_content]
        js_result = run_sample(js_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        assert py_result.stdout == js_result.stdout, "Python and JS results should match"

        # C sample currently cannot enable word_timestamps, so compare only main transcription line
        py_main_line = next((line.strip() for line in py_result.stdout.splitlines() if line.strip()), "")
        c_main_line = next((line.strip() for line in c_result.stdout.splitlines() if line.strip()), "")
        assert py_main_line == c_main_line, "Python and C main transcription should match"
