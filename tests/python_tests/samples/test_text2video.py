# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample


class TestText2Video:
    PROMPT = "A woman with long brown hair smiles at another woman with long blonde hair"

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-ltx-video", PROMPT),
        ],
        indirect=["convert_model"],
    )
    def test_sample_text2video(self, convert_model, sample_args):
        py_script = SAMPLES_PY_DIR / "video_generation/text2video.py"
        py_command = [sys.executable, py_script, convert_model, sample_args]
        run_sample(py_command)

        cpp_sample = SAMPLES_CPP_DIR / "text2video"
        cpp_command = [cpp_sample, convert_model, sample_args]
        run_sample(cpp_command)
