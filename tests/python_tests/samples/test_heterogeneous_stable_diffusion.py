# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestHeterogeneousStableDiffusion:
    @pytest.mark.samples
    @pytest.mark.image_generation
    @pytest.mark.parametrize(
        "download_model, prompt",
        [
            pytest.param("LCM_Dreamshaper_v7-int8-ov", "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting"),
        ],
        indirect=["download_model"],
    )
    def test_sample_heterogeneous_stable_diffusion(self, download_model, prompt):
        # Run Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "image_generation/heterogeneous_stable_diffusion.py")
        py_command = [sys.executable, py_script, download_model, '"' + prompt + '"']
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'heterogeneous_stable_diffusion')
        cpp_command = [cpp_sample, download_model, '"' + prompt + '"']
        run_sample(cpp_command)