# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestDecodeIntermediateResultConcurrency:
    @pytest.mark.samples
    @pytest.mark.dreamlike_anime_1_0
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("dreamlike-anime-1.0", "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting", "CPU", "NPU", "GPU"),
        ],
        indirect=["convert_model"],
    )
    def test_decode_intermediate_result_concurrency(self, convert_model, sample_args):
        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "image_generation/decode_intermediate_result_concurrency.py")
        py_command = [sys.executable, py_script, convert_model, sample_args]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'decode_intermediate_result_concurrency')
        cpp_command = [cpp_sample, convert_model, sample_args]
        run_sample(cpp_command)
