# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, convert_model
from test_utils import run_sample

convert_model_1 = convert_model
convert_model_2 = convert_model

class TestSpeculativeDecodingLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model_1, convert_model_2, sample_args",
        [
            pytest.param("dolly-v2-7b", "dolly-v2-3b", "Alan Turing was a"),
        ],
        indirect=["convert_model_1", "convert_model_2"],
    )
    def test_sample_speculative_decoding_lm(self, convert_model_1, convert_model_2, sample_args):      
        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/speculative_decoding_lm.py")
        py_command = [sys.executable, py_script, convert_model_1, convert_model_2, sample_args]
        py_result = run_sample(py_command)

        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'speculative_decoding_lm')
        cpp_command =[cpp_sample, convert_model_1, sample_args]
        cpp_result = run_sample(cpp_command)
        
        # Greedy decoding
        cpp_sample_ref = os.path.join(SAMPLES_CPP_DIR, 'speculative_decoding_lm')
        cpp_command_ref = [cpp_sample_ref, convert_model_1, convert_model_2, sample_args]
        cpp_result_ref = run_sample(cpp_command_ref)
        
        # Greedy decoding
        cpp_command_ref = [cpp_sample, convert_model_1, convert_model_2, sample_args]
        cpp_result_ref = run_sample(cpp_command_ref)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and CPP results should match"
        assert cpp_result_ref.stdout == cpp_result.stdout, "Greedy and speculative decoding results should match"

