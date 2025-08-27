# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

test_prompt = """Code:
def add(a, b):
    return a + b

Question: Can you please add 2 and 3
A:"""

class TestPromptLookupDecodingLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("Qwen2.5-0.5B-Instruct", test_prompt),
        ],
        indirect=["convert_model"],
    )
    def test_prompt_lookup_decoding_lm(self, convert_model, sample_args):
        env = os.environ.copy()
        env["OPENVINO_LOG_LEVEL"] = "0"
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'prompt_lookup_decoding_lm')
        cpp_command =[cpp_sample, convert_model, sample_args]
        cpp_result = run_sample(cpp_command, env=env)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/prompt_lookup_decoding_lm.py")
        py_command = [sys.executable, py_script, convert_model, sample_args]
        py_result = run_sample(py_command, env=env)

        
        # Greedy decoding
        cpp_sample_ref = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
        cpp_command_ref = [cpp_sample_ref, convert_model, sample_args]
        cpp_result_ref = run_sample(cpp_command_ref, env=env)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and CPP results should match"
        assert cpp_result_ref.stdout == cpp_result.stdout, "Greedy and speculative decoding results should match"

