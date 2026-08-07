# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample


class TestImage2Video:
    PROMPT = "A golden retriever in a sunlit bedroom slowly stands up and turns its head to look at the camera"

    @pytest.mark.samples
    @pytest.mark.video_generation
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("tiny-random-ltx-video", PROMPT),
        ],
        indirect=["convert_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["overture-creations.png"], indirect=True)
    def test_sample_image2video(self, convert_model, sample_args, download_test_content):
        py_script = SAMPLES_PY_DIR / "video_generation/image2video.py"
        py_command = [sys.executable, py_script, convert_model, download_test_content, sample_args]
        run_sample(py_command)

        cpp_sample = SAMPLES_CPP_DIR / "image2video"
        cpp_command = [cpp_sample, convert_model, download_test_content, sample_args]
        run_sample(cpp_command)
