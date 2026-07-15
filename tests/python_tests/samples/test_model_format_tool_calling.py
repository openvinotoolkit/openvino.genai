# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import sys

import pytest

from conftest import SAMPLES_PY_DIR
from test_utils import run_sample


@pytest.mark.vlm
@pytest.mark.samples
@pytest.mark.parametrize("convert_model", ["tiny-random-qwen3.5"], indirect=True)
@pytest.mark.parametrize("download_test_content", ["images/image.png"], indirect=True)
def test_python_model_format_tool_calling_sample(convert_model, download_test_content):
    py_script = SAMPLES_PY_DIR / "visual_language_chat" / "model_format_tool_calling.py"
    py_command = [
        sys.executable,
        py_script,
        convert_model,
        download_test_content,
        "--prompt",
        "Look at the image and call get_weather for tomorrow. Use Dublin if no city is visible.",
    ]

    run_sample(py_command)
