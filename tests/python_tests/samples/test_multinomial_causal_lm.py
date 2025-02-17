# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR

class TestMultinomialCausalLM:
    @pytest.mark.llm
    @pytest.mark.parametrize(
        "convert_model, sample_args, sample_type",
        [
            pytest.param("open_llama_3b_v2", "return 0", "py and cpp"),
            pytest.param("TinyLlama-1.1B-Chat-v1.0", "0", "py"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_multinomial_causal_lm(self, convert_model, sample_args, sample_type):
        py_result = cpp_result = None

        if "py" in sample_type:
            # Run Python sample
            py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/multinomial_causal_lm.py")
            py_result = subprocess.run(["python", py_script, convert_model, sample_args], capture_output=True, text=True, check=True)

        if "cpp" in sample_type:
            # Run C++ sample
            cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'multinomial_causal_lm')
            cpp_result = subprocess.run([cpp_sample, convert_model, sample_args], capture_output=True, text=True, check=True)

        if "py" in sample_type and "cpp" in sample_type:
            # Compare results
            assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
