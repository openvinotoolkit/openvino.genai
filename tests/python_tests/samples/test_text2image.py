# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestText2Image:
    @pytest.mark.samples
    @pytest.mark.dreamlike_anime_1_0
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("dreamlike-anime-1.0", "cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting"),
        ],
        indirect=["convert_model"],
    )
    def test_sample_text2image(self, convert_model, sample_args):
        # Run Python sample
        py_script = (SAMPLES_PY_DIR / "image_generation/text2image.py").as_posix()
        py_command = [sys.executable, py_script, convert_model, sample_args]
        run_sample(py_command)

        # Run C++ sample
        cpp_sample = (SAMPLES_CPP_DIR / 'text2image').as_posix()
        cpp_command = [cpp_sample, convert_model, sample_args]
        run_sample(cpp_command)


    @pytest.mark.samples
    @pytest.mark.dreamlike_anime_1_0
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("dreamlike-anime-1.0",
                ("cyberpunk cityscape like Tokyo New York with tall buildings at dusk golden hour cinematic lighting",
                 "village landscape with mountains and a river at sunrise",
                 "midcentury modern house with a garden and a pond at sunset")),
        ],
        indirect=["convert_model"],
    )
    def test_sample_text2image_concurrency(self, convert_model, sample_args):
        # Run C++ sample
        cpp_sample = (SAMPLES_CPP_DIR / 'text2image_concurrency').as_posix()
        cpp_command = [cpp_sample, convert_model, *sample_args]
        run_sample(cpp_command)
