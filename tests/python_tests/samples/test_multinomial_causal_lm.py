# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR

# multinomial_causal_lm sample

@pytest.mark.llm
@pytest.mark.py
@pytest.mark.parametrize("convert_model", [
    {"model_id": "TinyLlama-1.1B-Chat-v1.0"}
], indirect=["convert_model"])
@pytest.mark.parametrize("sample_args", ["0"])
def test_python_sample_multinomial_causal_lm_tiny_llama(convert_model, sample_args):
    script = os.path.join(SAMPLES_PY_DIR, "text_generation/multinomial_causal_lm.py")
    result = subprocess.run(["python", script, convert_model, sample_args], check=True)
    assert result.returncode == 0, f"Script execution failed for model {convert_model} with argument {sample_args}"
    
@pytest.mark.llm
@pytest.mark.py
@pytest.mark.parametrize("convert_model", [
    {"model_id": "open_llama_3b_v2"}
], indirect=["convert_model"])
@pytest.mark.parametrize("sample_args", ["a", "return 0"])
def test_python_sample_multinomial_causal_lm_open_llama(convert_model, sample_args, shared_data):
    script = os.path.join(SAMPLES_PY_DIR, "text_generation/multinomial_causal_lm.py")
    result = subprocess.run(["python", script, convert_model, sample_args], check=True)
    assert result.returncode == 0, f"Script execution failed for model {convert_model} with argument {sample_args}"
    shared_data.setdefault("multinomial_causal_lm", {}).setdefault("py", {}).setdefault("open_llama_3b_v2", {})[sample_args] = result.stdout

@pytest.mark.llm
@pytest.mark.cpp
@pytest.mark.parametrize("convert_model", [
    {"model_id": "open_llama_3b_v2"}
], indirect=["convert_model"])
@pytest.mark.parametrize("sample_args", ["return 0"])
def test_cpp_sample_multinomial_causal_lm_open_llama(convert_model, sample_args,  shared_data):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
    result = subprocess.run([cpp_sample, convert_model, sample_args], check=True)
    assert result.returncode == 0, "C++ sample execution failed"
    shared_data.setdefault("multinomial_causal_lm", {}).setdefault("cpp", {}).setdefault("open_llama_3b_v2", {})[sample_args] = result.stdout


@pytest.mark.llm    
@pytest.mark.cpp
@pytest.mark.py
def test_sample_multinomial_causal_lm_diff(shared_data):
    py_result = shared_data.get("multinomial_causal_lm", {}).get("py", {}).get("open_llama_3b_v2", {}).get("return 0")
    cpp_result = shared_data.get("multinomial_causal_lm", {}).get("cpp", {}).get("open_llama_3b_v2", {}).get("return 0")
    # if not py_result or not cpp_result:
    #     pytest.skip("Skipping because one of the prior tests was skipped or failed.")
    assert py_result == cpp_result, "Results should match"
