# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import tempfile
import pytest
import shutil

# Define model names and directories
MODELS = {
    "TinyLlama-1.1B-Chat-v1.1": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3TB-Chat-v1.0",
    "WhisperTiny": "openai/whisper-tiny",
    "open_llama_3b_v2": "openlm-research/open_llama_3b_v2"
}

TEMP_DIR = os.environ.get("TEMP_DIR", tempfile.mkdtemp())
MODELS_DIR = os.path.join(TEMP_DIR, "test_models")
TEST_DATA = os.path.join(TEMP_DIR, "test_data")
TEST_FILE_URL = "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav"

SAMPLES_PY_DIR = os.environ.get("SAMPLES_PY_DIR", os.getcwd())
SAMPLES_CPP_DIR = os.environ.get("SAMPLES_CPP_DIR", os.getcwd())

@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown(request):
    """Fixture to set up and tear down the temporary directories."""
    print(f"Creating directories: {MODELS_DIR} and {TEST_DATA}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TEST_DATA, exist_ok=True)
    yield
    if not os.environ.get("TEMP_DIR"):
        print(f"Removing temporary directory: {TEMP_DIR}")
        shutil.rmtree(TEMP_DIR)
    else:
        print(f"Skipping cleanup of temporary directory: {TEMP_DIR}")

@pytest.fixture(scope="session")
def convert_model(request):
    """Fixture to convert the model once for the session."""
    model_id, *extra_args = request.param
    model_name = MODELS[model_id]
    model_path = os.path.join(MODELS_DIR, model_name)
    # Convert the model if not already converted
    if not os.path.exists(model_path):
        command = [
            "optimum-cli", "export", "openvino",
            "--model", model_name, model_path
        ]
        if extra_args:
            command.extend(extra_args)
        result = subprocess.run(command, check=True)
        assert result.returncode == 0, f"Model {model_name} conversion failed"
    return model_path

@pytest.fixture(scope="session")
def download_test_content(request):
    """Download the test content from the given URL and return the file path."""
    file_url = request.param
    file_name = os.path.basename(file_url)
    file_path = os.path.join(TEST_DATA, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading test content from {file_url}...")
        result = subprocess.run(
            ["wget", file_url, "-O", file_path],
            check=True
        )
        assert result.returncode == 0, "Failed to download test content"
        print(f"Downloaded test content to {file_path}")
    else:
        print(f"Test content already exists at {file_path}")
    return file_path

@pytest.mark.llm
@pytest.mark.py
@pytest.mark.parametrize("convert_model", [("TinyLlama-1.1B-Chat-v1.1", "--trust-remote-code")], indirect=True)
def test_python_sample_multinomial_causal_lm(convert_model):
    script = os.path.join(SAMPLES_PY_DIR, "multinomial_causal_lm/multinomial_causal_lm.py")
    exit_code = subprocess.run(["python", script, convert_model, "0"]).returncode
    assert exit_code == 0, f"Script execution failed for model {convert_model}"

@pytest.mark.py
@pytest.mark.whisper
@pytest.mark.parametrize("convert_model", [("WhisperTiny", "--trust-remote-code")], indirect=True)
@pytest.mark.parametrize("download_test_content", [TEST_FILE_URL], indirect=True)
def test_python_sample_whisper_speech_recognition(convert_model, download_test_content):
    script = os.path.join(SAMPLES_PY_DIR, "whisper_speech_recognition/whisper_speech_recognition.py")
    exit_code = subprocess.run(["python", script, convert_model, download_test_content]).returncode
    assert exit_code == 0, f"Script execution failed for model {convert_model}"

@pytest.mark.cpp
@pytest.mark.llm
@pytest.mark.parametrize("convert_model", [("TinyLlama-1.1B-Chat-v1.1",)], indirect=True)
def test_cpp_sample_greedy_causal_lm(convert_model):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
    exit_code = subprocess.run([cpp_sample, convert_model, ""]).returncode
    assert exit_code == 0, "C++ sample execution failed"

@pytest.mark.cpp
@pytest.mark.whisper
@pytest.mark.parametrize("convert_model", [("WhisperTiny",)], indirect=True)
@pytest.mark.parametrize("download_test_content", [TEST_FILE_URL], indirect=True)
def test_cpp_sample_whisper_speech_recognition(convert_model, download_test_content):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'whisper_speech_recognition')
    exit_code = subprocess.run([cpp_sample, convert_model, download_test_content]).returncode
    assert exit_code == 0, "C++ sample execution failed"
    
@pytest.mark.py
@pytest.mark.llm
@pytest.mark.parametrize("convert_model", [("open_llama_3b_v2", "--trust-remote-code', '--weight-format fp16")], indirect=True)
def test_python_sample_multinomial_causal_lm(convert_model):
    script = os.path.join(SAMPLES_PY_DIR, "multinomial_causal_lm/multinomial_causal_lm.py")
    exit_code = subprocess.run(["python", script, convert_model, 'b']).returncode
    assert exit_code == 0, f"Script execution failed for model {convert_model}"
    
@pytest.mark.cpp
@pytest.mark.llm
@pytest.mark.parametrize("convert_model", [("open_llama_3b_v2","--trust-remote-code", "--weight-format fp16")], indirect=True)
def test_cpp_sample_multinomial_causal_lm(convert_model):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'multinomial_causal_lm')
    exit_code = subprocess.run([cpp_sample, convert_model, "a"]).returncode
    assert exit_code == 0, "C++ sample execution failed"