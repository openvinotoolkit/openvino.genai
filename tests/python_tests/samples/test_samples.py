# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import tempfile
import pytest
import shutil

# Define model names and directories
MODELS = {
    "TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama-1.1B-intermediate-step-1431k-3T": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "WhisperTiny": "openai/whisper-tiny",
    "open_llama_3b_v2": "openlm-research/open_llama_3b_v2"
}

TEST_FILES = {
    "how_are_you_doing_today.wav": "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav",
    "adapter_model.safetensors": "https://huggingface.co/smangrul/tinyllama_lora_sql/resolve/main/adapter_model.safetensors"
}

TEMP_DIR = os.environ.get("TEMP_DIR", tempfile.mkdtemp())
MODELS_DIR = os.path.join(TEMP_DIR, "test_models")
TEST_DATA = os.path.join(TEMP_DIR, "test_data")

SAMPLES_PY_DIR = os.environ.get("SAMPLES_PY_DIR", os.getcwd())
SAMPLES_CPP_DIR = os.environ.get("SAMPLES_CPP_DIR", os.getcwd())

# A shared fixture to hold data
@pytest.fixture(scope="session")
def shared_data():
    return {}

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
    params = request.param
    model_id = params.get("model_id")
    extra_args = params.get("extra_args", [])
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
    yield model_path
    # Cleanup the model after tests
    if os.path.exists(model_path):
        print(f"Removing converted model: {model_path}")
        shutil.rmtree(model_path)

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
    yield file_path
    # Cleanup the test content after tests
    if os.path.exists(file_path):
        print(f"Removing test content: {file_path}")
        os.remove(file_path)

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

# multinomial_causal_lm sample
@pytest.mark.llm
@pytest.mark.py
@pytest.mark.parametrize("convert_model, sample_args", [
    ({"model_id": "TinyLlama-1.1B-Chat-v1.0", "extra_args": ["--trust-remote-code"]}, "0"),
    ({"model_id": "open_llama_3b_v2", "extra_args": ["--trust-remote-code", "--weight-format", "fp16"]}, "a"),
    ({"model_id": "open_llama_3b_v2", "extra_args": ["--trust-remote-code", "--weight-format", "fp16"]}, "return 0"),
], indirect=["convert_model"])
def test_python_sample_multinomial_causal_lm(convert_model, sample_args, shared_data):
    script = os.path.join(SAMPLES_PY_DIR, "multinomial_causal_lm/multinomial_causal_lm.py")
    result = subprocess.run(["python", script, convert_model, sample_args], check=True)
    assert result.returncode == 0, f"Script execution failed for model {convert_model} with argument {sample_args}"
    shared_data.setdefault("multinomial_causal_lm", {}).setdefault("py", {}).setdefault(convert_model, {})[sample_args] = result.stdout

@pytest.mark.llm    
@pytest.mark.cpp
@pytest.mark.parametrize("convert_model, sample_args", [
    ({"model_id": "open_llama_3b_v2", "extra_args": ["--trust-remote-code", "--weight-format", "fp16"]}, "b"),
    ({"model_id": "open_llama_3b_v2", "extra_args": ["--trust-remote-code", "--weight-format", "fp16"]}, "return 0"),
], indirect=["convert_model"])
def test_cpp_sample_multinomial_causal_lm(convert_model, sample_args, shared_data):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'multinomial_causal_lm')
    result = subprocess.run([cpp_sample, convert_model, sample_args], check=True)
    assert result.returncode == 0, "C++ sample execution failed"
    shared_data.setdefault("multinomial_causal_lm", {}).setdefault("cpp", {}).setdefault(convert_model, {})[sample_args] = result.stdout

@pytest.mark.llm    
@pytest.mark.cpp
@pytest.mark.py
def test_sample_multinomial_causal_lm_diff(shared_data):
    py_result = shared_data.get("multinomial_causal_lm", {}).get("py", {}).get("open_llama_3b_v2", {}).get("return 0")
    cpp_result = shared_data.get("multinomial_causal_lm", {}).get("cpp", {}).get("open_llama_3b_v2", {}).get("return 0")
    if not py_result or not cpp_result:
        pytest.skip("Skipping because one of the prior tests was skipped or failed.")
    assert py_result == cpp_result, "Results should match"

# Greedy causal LM samples
@pytest.mark.llm
@pytest.mark.cpp
@pytest.mark.parametrize("convert_model, sample_args", [
    ({"model_id": "TinyLlama-1.1B-Chat-v1.0"}, ""),
    ({"model_id": "open_llama_3b_v2", "extra_args": ["--trust-remote-code", "--weight-format", "fp16"]}, "return 0")
], indirect=["convert_model"])
def test_cpp_sample_greedy_causal_lm(convert_model, sample_args):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
    exit_code = subprocess.run([cpp_sample, convert_model, sample_args], check=True).returncode
    assert exit_code == 0, "C++ sample execution failed"
    
# text_generation sample
@pytest.mark.llm
@pytest.mark.py
@pytest.mark.parametrize("convert_model, sample_args", [
    (
        {
            "model_id": "TinyLlama-1.1B-intermediate-step-1431k-3T", 
            "extra_args": ["--trust-remote-code"]
        }, 
        "How to create a table with two columns, one of them has type float, another one has type int?"
    )
], indirect=["convert_model"])
@pytest.mark.parametrize("download_test_content", [TEST_FILES["adapter_model.safetensors"]], indirect=True)
def test_python_sample_text_generation(convert_model, download_test_content, sample_args):
    script = os.path.join(SAMPLES_PY_DIR, "text_generation/lora.py")
    result = subprocess.run(["python", script, convert_model, download_test_content, sample_args], check=True)
    assert result.returncode == 0, f"Script execution failed for model {convert_model}"
