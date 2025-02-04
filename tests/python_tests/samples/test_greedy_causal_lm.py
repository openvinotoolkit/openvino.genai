# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR

# Greedy causal LM samples

@pytest.mark.llm
@pytest.mark.cpp
@pytest.mark.parametrize(
    "convert_model, sample_args",
    [
        pytest.param("TinyLlama-1.1B-Chat-v1.0", ""),
        pytest.param("open_llama_3b_v2", "return 0"),
        pytest.param("Qwen-7B-Chat", "69"),
    ],
    indirect=["convert_model"],
)
def test_cpp_sample_greedy_causal_lm(convert_model, sample_args, shared_data, request):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
    result = subprocess.run([cpp_sample, convert_model, sample_args], check=True)
    assert result.returncode == 0, "C++ sample execution failed"
    model_id = request.node.callspec.params["convert_model"]
    shared_data.setdefault("greedy_causal_lm", {}).setdefault("cpp", {}).setdefault(model_id, {})[sample_args] = result.stdout
    
@pytest.mark.llm
@pytest.mark.py
@pytest.mark.parametrize(
    "convert_model, sample_args",
    [
        pytest.param("Qwen-7B-Chat", "69"),
    ],
    indirect=["convert_model"],
)
def test_python_sample_greedy_causal_lm(convert_model, sample_args, shared_data, request):
    script = os.path.join(SAMPLES_PY_DIR, "text_generation/greedy_causal_lm.py")
    result = subprocess.run(["python", script, convert_model, sample_args], capture_output=True, text=True, check=True)
    assert result.returncode == 0, f"Script execution failed for model {convert_model} with argument {sample_args}"
    model_id = request.node.callspec.params["convert_model"]
    shared_data.setdefault("greedy_causal_lm", {}).setdefault("py", {}).setdefault(model_id, {})[sample_args] = result.stdout

@pytest.mark.llm    
@pytest.mark.cpp
@pytest.mark.py
@pytest.mark.parametrize(
    "model_id, sample_args",
    [
        pytest.param("Qwen-7B-Chat", "69"),
    ]
)
def test_sample_greedy_causal_lm_diff(shared_data, model_id, sample_args):
    py_result = shared_data.get("greedy_causal_lm", {}).get("py", {}).get(model_id, {}).get(sample_args)
    cpp_result = shared_data.get("greedy_causal_lm", {}).get("cpp", {}).get(model_id, {}).get(sample_args)
    if not py_result or not cpp_result:
        pytest.skip("Skipping because one of the prior tests was skipped or failed.")
    assert py_result == cpp_result, "Results should match"