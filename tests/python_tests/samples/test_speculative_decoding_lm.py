# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, convert_model
from test_utils import run_sample

convert_draft_model = convert_model

class TestSpeculativeDecodingLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, convert_draft_model, sample_args",
        [
            pytest.param("SmolLM2-360M", "SmolLM2-135M", "Alan Turing was a"),
        ],
        indirect=["convert_model", "convert_draft_model"],
    )
    def test_sample_speculative_decoding_lm(self, convert_model, convert_draft_model, sample_args):
        env = os.environ.copy()
        env["OPENVINO_LOG_LEVEL"] = "0"
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'speculative_decoding_lm')
        cpp_command =[cpp_sample, convert_model, convert_draft_model, sample_args]
        cpp_result = run_sample(cpp_command, env=env)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/speculative_decoding_lm.py")
        py_command = [sys.executable, py_script, convert_model, convert_draft_model, sample_args]
        py_result = run_sample(py_command, env=env)
        
        # Greedy decoding
        cpp_sample_ref = os.path.join(SAMPLES_CPP_DIR, 'greedy_causal_lm')
        cpp_command_ref = [cpp_sample_ref, convert_model, sample_args]
        cpp_result_ref = run_sample(cpp_command_ref, env=env)

        # Compare results
        assert cpp_result_ref.stdout.strip() in py_result.stdout.strip(), "Python and CPP results should match"
        assert cpp_result_ref.stdout.strip() in cpp_result.stdout.strip(), "Greedy and speculative decoding results should match"

