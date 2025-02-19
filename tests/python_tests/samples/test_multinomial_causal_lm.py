# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR

class TestMultinomialCausalLM:
    @pytest.mark.llm
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
        py_result = subprocess.run(["python", py_script, convert_model, sample_args], capture_output=True, text=True, check=True)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'multinomial_causal_lm')
        cpp_result = subprocess.run([cpp_sample, convert_model, sample_args], capture_output=True, text=True, check=True)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
