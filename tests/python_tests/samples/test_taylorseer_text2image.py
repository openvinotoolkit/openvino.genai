# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample


class TestTaylorSeerText2Image:
    PROMPT = "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"

    @pytest.mark.samples
    @pytest.mark.tiny_random_flux
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-flux", PROMPT),
        ],
        indirect=["convert_model"],
    )
    def test_sample_taylorseer_text2image_default(self, convert_model, sample_args):
        """Test TaylorSeer text2image sample with default cache configuration."""
        # Run Python sample
        py_script = SAMPLES_PY_DIR / "image_generation/taylorseer_text2image.py"
        py_command = [sys.executable, py_script, convert_model, sample_args]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / "taylorseer_text2image"
        cpp_command = [cpp_sample, convert_model, sample_args]
        run_sample(cpp_command)

    @pytest.mark.samples
    @pytest.mark.tiny_random_flux
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param(
                "tiny-random-flux",
                (PROMPT, "--cache-interval", "4", "--disable-before", "8", "--disable-after", "-3", "--steps", "20"),
            ),
        ],
        indirect=["convert_model"],
    )
    def test_sample_taylorseer_text2image_custom_config(self, convert_model, sample_args):
        """Test TaylorSeer text2image sample with custom cache configuration."""
        # Run Python sample
        py_script = SAMPLES_PY_DIR / "image_generation/taylorseer_text2image.py"
        py_command = [sys.executable, py_script, convert_model, *sample_args]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = SAMPLES_CPP_DIR / "taylorseer_text2image"
        cpp_command = [cpp_sample, convert_model, *sample_args]
        run_sample(cpp_command)
