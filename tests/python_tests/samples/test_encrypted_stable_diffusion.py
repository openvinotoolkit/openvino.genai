# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestEncryptedStableDiffusion:
    PROMPT = "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"

    @pytest.mark.samples
    @pytest.mark.dreamlike_anime_1_0
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("dreamlike-anime-1.0", PROMPT),
        ],
        indirect=["convert_model"],
    )
    def test_sample_encrypted_stable_diffusion(self, convert_model, sample_args):
        # Run Python sample
        py_script = SAMPLES_PY_DIR / "image_generation/encrypted_stable_diffusion.py"
        py_command = [sys.executable, py_script, convert_model, sample_args]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / 'encrypted_stable_diffusion'
        cpp_command = [cpp_sample, convert_model, sample_args]
        run_sample(cpp_command)
