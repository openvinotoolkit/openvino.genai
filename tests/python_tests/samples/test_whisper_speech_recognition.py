# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import TEST_FILES, SAMPLES_PY_DIR, SAMPLES_CPP_DIR

# whisper_speech_recognition sample
@pytest.mark.whisper
@pytest.mark.py
@pytest.mark.parametrize("convert_model", [{"model_id": "WhisperTiny", "extra_args": ["--trust-remote-code"]}], 
                         indirect=True, ids=lambda p: f"model={p['model_id']}")
@pytest.mark.parametrize("download_test_content", [TEST_FILES["how_are_you_doing_today.wav"]], indirect=True)
def test_python_sample_whisper_speech_recognition(convert_model, download_test_content):
    script = os.path.join(SAMPLES_PY_DIR, "whisper_speech_recognition/whisper_speech_recognition.py")
    result = subprocess.run(["python", script, convert_model, download_test_content], check=True)
    assert result.returncode == 0, f"Script execution failed for model {convert_model}"

@pytest.mark.whisper
@pytest.mark.cpp
@pytest.mark.parametrize("convert_model", [{"model_id": "WhisperTiny"}], 
                         indirect=True, ids=lambda p: f"model={p['model_id']}")
@pytest.mark.parametrize("download_test_content", [TEST_FILES["how_are_you_doing_today.wav"]], indirect=True)
def test_cpp_sample_whisper_speech_recognition(convert_model, download_test_content):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'whisper_speech_recognition')
    exit_code = subprocess.run([cpp_sample, convert_model, download_test_content], check=True).returncode
    assert exit_code == 0, "C++ sample execution failed"