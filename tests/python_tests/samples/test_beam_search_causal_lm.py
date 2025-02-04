# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, logger
    
@pytest.mark.llm
@pytest.mark.py
@pytest.mark.parametrize(
    "convert_model, sample_args",
    [
        pytest.param("Qwen1.5-7B-Chat", "你好！", id="Qwen1.5-7B-Chat_Chinese"),
        pytest.param("phi-2", "69", id="phi-2_Numeric"),
    ],
    indirect=["convert_model"],
)
def test_python_sample_beam_search_causal_lm(convert_model, sample_args, shared_data, request):
    script = os.path.join(SAMPLES_PY_DIR, "text_generation/beam_search_causal_lm.py")
    result = subprocess.run(["python", script, convert_model, sample_args], capture_output=True, text=True, check=True)
    assert result.returncode == 0, f"Script execution failed for model {convert_model} with argument {sample_args}"
    model_id = request.node.callspec.params["convert_model"]
    shared_data.setdefault("beam_search_causal_lm", {}).setdefault("py", {}).setdefault(model_id, {})[sample_args] = result.stdout
    logger.info(f"Testing model: {model_id} with input: {sample_args} and output: {result.stdout}")

@pytest.mark.llm
@pytest.mark.cpp
@pytest.mark.parametrize(
    "convert_model, sample_args",
    [
        pytest.param("Qwen1.5-7B-Chat", "你好！", id="Qwen1.5-7B-Chat_Chinese"),
        pytest.param("phi-2", "69", id="phi-2_Numeric"),
    ],
    indirect=["convert_model"],
)
def test_cpp_sample_beam_search_causal_lm(convert_model, sample_args, shared_data, request):
    cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'beam_search_causal_lm')
    result = subprocess.run([cpp_sample, convert_model, sample_args], capture_output=True, text=True, check=True)
    assert result.returncode == 0, "C++ sample execution failed"
    # save output for the next test
    model_id = request.node.callspec.params["convert_model"]
    shared_data.setdefault("beam_search_causal_lm", {}).setdefault("cpp", {}).setdefault(model_id, {})[sample_args] = result.stdout
    logger.info(f"Testing model: {model_id} with input: {sample_args} and output: {result.stdout}")


@pytest.mark.llm    
@pytest.mark.cpp
@pytest.mark.py
@pytest.mark.parametrize(
    "model, sample_args",
    [
        pytest.param("Qwen1.5-7B-Chat", "你好！", id="Qwen1.5-7B-Chat_Chinese"),
        pytest.param("phi-2", "69", id="phi-2_Numeric"),
    ]
)
def test_sample_beam_search_causal_lm_diff(model, sample_args, shared_data):
    py_result = shared_data.get("beam_search_causal_lm", {}).get("py", {}).get(model, {}).get(sample_args)
    cpp_result = shared_data.get("beam_search_causal_lm", {}).get("cpp", {}).get(model, {}).get(sample_args)
    if not py_result or not cpp_result:
        pytest.skip("Skipping because one of the prior tests was skipped or failed.")
    assert py_result == cpp_result, "Results should match"
