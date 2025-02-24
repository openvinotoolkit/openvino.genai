# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from conftest import SAMPLES_CPP_DIR
from test_utils import run_sample

class TestEncryptedModelCausalLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("SmolLM2-135M", "Why is the sun yellow?"),
        ],
        indirect=["convert_model"],
    )
    def test_cpp_sample_encrypted_model_causal_lm(self, convert_model, sample_args):
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'encrypted_model_causal_lm')
        cpp_command =[cpp_sample, convert_model, sample_args]
        run_sample(cpp_command)