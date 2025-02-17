# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
    
class TestBeamSearchCausalLM:
    @pytest.mark.llm
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("Qwen1.5-7B-Chat", "你好！", id="Qwen1.5-7B-Chat_Chinese"),
            pytest.param("phi-2", "69", id="phi-2_Numeric"),
            pytest.param("notus-7b-v1", "69"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_beam_search_causal_lm(self, convert_model, sample_args):
        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/beam_search_causal_lm.py")
        py_result = subprocess.run(["python", py_script, convert_model, sample_args], capture_output=True, text=True, check=True)

        # C++ test
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'beam_search_causal_lm')
        cpp_result = subprocess.run([cpp_sample, convert_model, sample_args], capture_output=True, text=True, check=True)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
