# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestMultinomialCausalLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("SmolLM-135M", "return 0"),
            pytest.param("LaMini-GPT-124M", "0"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_multinomial_causal_lm(self, convert_model, sample_args):
        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/multinomial_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model, sample_args]
        py_result = run_sample(py_command)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'multinomial_causal_lm')
        cpp_command = [cpp_sample, convert_model, sample_args]
        cpp_result = run_sample(cpp_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
