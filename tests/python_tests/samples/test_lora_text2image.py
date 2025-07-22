# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestLoraText2Image:
    @pytest.mark.samples
    @pytest.mark.image_generation
    @pytest.mark.parametrize(
        "convert_model, prompt, sample_args",
        [
            pytest.param("dreamlike-anime-1.0", "curly-haired unicorn in the forest, anime, line", "0.7"),
        ],
        indirect=["convert_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["soulcard.safetensors"], indirect=True)
    def test_sample_lora_text2image(self, convert_model, prompt, download_test_content, sample_args):
        pytest.skip(reason="Ticket 170878")

        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "image_generation/lora_text2image.py")
        py_command = [sys.executable, py_script, convert_model, prompt, download_test_content, sample_args]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'lora_text2image')
        cpp_command = [cpp_sample, convert_model, prompt, download_test_content, sample_args]
        run_sample(cpp_command)