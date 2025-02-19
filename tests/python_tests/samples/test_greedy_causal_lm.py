# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR

class TestGreedyCausalLM:
    @pytest.mark.llm
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("LaMini-GPT-124M", "test"),
            pytest.param("SmolLM-135M-Instruct", "return 0"),
            pytest.param("Qwen2.5-0.5B-Instruct", "69"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_greedy_causal_lm(self, convert_model, sample_args):
        # Test Python sample
        script = os.path.join(SAMPLES_PY_DIR, "text_generation/greedy_causal_lm.py")
        py_result = subprocess.run(["python", script, convert_model, sample_args], capture_output=True, text=True, check=True)

        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
        cpp_result = subprocess.run([cpp_sample, convert_model, sample_args], capture_output=True, text=True, check=True)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
