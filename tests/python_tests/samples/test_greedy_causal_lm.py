# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import TEST_FILES, SAMPLES_PY_DIR, SAMPLES_CPP_DIR

# Greedy causal LM samples

@pytest.mark.llm
@pytest.mark.cpp
@pytest.mark.parametrize("convert_model", [
    {"model_id": "TinyLlama-1.1B-Chat-v1.0"}
], indirect=["convert_model"])
@pytest.mark.parametrize("sample_args", [""])
def test_cpp_sample_greedy_causal_lm_tiny_llama(convert_model, sample_args):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
    exit_code = subprocess.run([cpp_sample, convert_model, sample_args], check=True).returncode
    assert exit_code == 0, "C++ sample execution failed"

@pytest.mark.llm
@pytest.mark.cpp
@pytest.mark.parametrize("convert_model", [
    {"model_id": "open_llama_3b_v2"}
], indirect=["convert_model"])
@pytest.mark.parametrize("sample_args", ["return 0"])
def test_cpp_sample_greedy_causal_lm_open_llama(convert_model, sample_args):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
    exit_code = subprocess.run([cpp_sample, convert_model, sample_args], check=True).returncode
    assert exit_code == 0, "C++ sample execution failed"
    
# text_generation sample
@pytest.mark.llm
@pytest.mark.py
@pytest.mark.parametrize("convert_model", [
    {"model_id": "TinyLlama-1.1B-intermediate-step-1431k-3T"}
], indirect=["convert_model"])
@pytest.mark.parametrize("sample_args", ["How to create a table with two columns, one of them has type float, another one has type int?"])
@pytest.mark.parametrize("download_test_content", [TEST_FILES["adapter_model.safetensors"]], indirect=True)
def test_python_sample_text_generation(convert_model, download_test_content, sample_args):
    script = os.path.join(SAMPLES_PY_DIR, "text_generation/lora.py")
    result = subprocess.run(["python", script, convert_model, download_test_content, sample_args], check=True)
    assert result.returncode == 0, f"Script execution failed for model {convert_model}"