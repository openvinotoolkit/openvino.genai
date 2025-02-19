# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample
    
class TestBeamSearchCausalLM:
    @pytest.mark.llm
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("Qwen2-0.5B-Instruct", "你好！"),
            pytest.param("phi-1_5", "69"),
            pytest.param("SmolLM2-135M", "69"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_beam_search_causal_lm(self, convert_model, sample_args):
        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/beam_search_causal_lm.py")
        py_command = ["python", py_script, convert_model, sample_args]
        py_result = run_sample(py_command)

        # C++ test
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'beam_search_causal_lm')
        cpp_command = [cpp_sample, convert_model, sample_args]
        cpp_result = run_sample(cpp_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
