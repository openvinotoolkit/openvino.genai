# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess # nosec B404
import pytest
from conftest import TEST_FILES, SAMPLES_PY_DIR

class TestLora:
    @pytest.mark.llm
    @pytest.mark.py
    @pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-intermediate-step-1431k-3T"], indirect=True)
    @pytest.mark.parametrize("sample_args", ["How to create a table with two columns, one of them has type float, another one has type int?"])
    @pytest.mark.parametrize("download_test_content", [TEST_FILES["adapter_model.safetensors"]], indirect=True)
    def test_python_sample_lora(self, convert_model, download_test_content, sample_args):
        script = os.path.join(SAMPLES_PY_DIR, "text_generation/lora.py")
        subprocess.run(["python", script, convert_model, download_test_content, sample_args], check=True)
    