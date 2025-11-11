# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_JS_DIR
from test_utils import run_sample


@pytest.mark.llm
@pytest.mark.samples
@pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-Chat-v1.0"], indirect=True)
def test_structured_output_sample(convert_model):
    # Python test
    py_script = SAMPLES_PY_DIR / "text_generation" / "structural_tags_generation.py"
    py_command = [sys.executable, py_script, convert_model]
    py_result = run_sample(py_command)
    py_predictions = py_result.stdout

    # JS test
    js_script = SAMPLES_JS_DIR / "text_generation" / "structural_tags_generation.js"
    js_command = ["node", js_script, convert_model]
    js_result = run_sample(js_command)
    js_predictions = js_result.stdout

    # Compare results
    assert py_predictions == js_predictions, "Python and JS results should match"
