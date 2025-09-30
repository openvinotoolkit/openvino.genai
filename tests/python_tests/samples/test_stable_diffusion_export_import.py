# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestStableDiffusionExportImport:
    @pytest.mark.samples
    @pytest.mark.stabilityai_sdxl_turbo
    @pytest.mark.parametrize("executable", [
        [SAMPLES_CPP_DIR / "stable_diffusion_export_import"],
        [sys.executable, SAMPLES_PY_DIR / "image_generation/stable_diffusion_export_import.py"],
    ])
    @pytest.mark.parametrize(
        "download_model, prompt",
        [
            pytest.param("stabilityai_sdxl_turbo", "cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting"),
        ],
        indirect=["download_model"],
    )
    def test_sample_stable_diffusion_export_import(self, executable, download_model, prompt):
        run_sample(executable + [download_model, '"' + prompt + '"'])
