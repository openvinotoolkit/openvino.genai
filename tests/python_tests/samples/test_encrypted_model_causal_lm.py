# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestEncryptedLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["Qwen2.5-0.5B-Instruct"], indirect=True)
    @pytest.mark.parametrize("prompt", ["Why is the sun yellow?"])

    def test_sample_encrypted_lm(self, convert_model, prompt):
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'encrypted_model_causal_lm')
        cpp_command =[cpp_sample, convert_model, prompt]
        cpp_result = run_sample(cpp_command)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/encrypted_model_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model, prompt]
        py_result = run_sample(py_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
