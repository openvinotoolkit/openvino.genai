# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestHeterogeneousStableDiffusion:
    @pytest.mark.samples
    @pytest.mark.LCM_Dreamshaper_v7_int8_ov
    @pytest.mark.parametrize("executable", [
        [SAMPLES_CPP_DIR / "heterogeneous_stable_diffusion"],
        [sys.executable, SAMPLES_PY_DIR / "image_generation/heterogeneous_stable_diffusion.py"],
    ])
    @pytest.mark.parametrize(
        "download_model, prompt",
        [
            pytest.param("LCM_Dreamshaper_v7-int8-ov", "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting"),
        ],
        indirect=["download_model"],
    )
    def test_sample_heterogeneous_stable_diffusion(self, executable, download_model, prompt):
        run_sample(executable + [download_model, '"' + prompt + '"'])
