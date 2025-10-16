# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_JS_DIR
from test_utils import run_sample

class TestMultinomialCausalLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("SmolLM-135M", '"return 0"'),
            pytest.param(
                "TinyLlama-1.1B-Chat-v1.0", 
                "0", 
                marks=pytest.mark.skipif(sys.platform == "darwin", reason="CVS-163463")
            ),
        ],
        indirect=["convert_model"],
    )
    def test_sample_multinomial_causal_lm(self, convert_model, sample_args):
        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'multinomial_causal_lm')
        cpp_command = [cpp_sample, convert_model, sample_args]
        cpp_result = run_sample(cpp_command)

        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/multinomial_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model, sample_args]
        py_result = run_sample(py_command)


        # Test JS sample
        js_sample = os.path.join(SAMPLES_JS_DIR, "text_generation/multinomial_causal_lm.js")
        js_command =['node', js_sample, convert_model, sample_args]
        js_result = run_sample(js_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
        assert py_result.stdout == js_result.stdout, "Python and JS results should match"
