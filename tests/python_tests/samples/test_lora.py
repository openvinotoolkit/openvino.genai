# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR
from test_utils import run_sample

class TestLora:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["TinyStories-1M"], indirect=True)
    @pytest.mark.parametrize("sample_args", ["How to create a table with two columns, one of them has type float, another one has type int?"])
    @pytest.mark.parametrize("download_test_content", ["adapter_model.safetensors"], indirect=True)
    def test_python_sample_lora(self, convert_model, download_test_content, sample_args):      
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/lora_greedy_causal_lm.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content, sample_args]
        run_sample(py_command)