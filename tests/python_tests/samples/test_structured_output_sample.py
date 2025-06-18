# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys
import json

from conftest import logger, SAMPLES_PY_DIR, MODELS
from test_utils import run_sample

@pytest.mark.llm
@pytest.mark.samples
@pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-Chat-v1.0"], indirect=True)
@pytest.mark.parametrize("prompt,expected_keys", [
    ("2 persons and 1 car.", {"person": 2, "car": 1, "transaction": 0}),
    ("Give me a json for 3 persons and 7 cars.", {"person": 3, "car": 7, "transaction": 0}),
    ("Generate 1 transaction.", {"person": 0, "car": 0, "transaction": 1}),
    ("Generate 1000 horses.", "No items generated. Please try again with a different request."),
])
def test_structured_output_sample(convert_model, prompt, expected_keys):
    py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/structured_output_generation.py")
    py_command = [sys.executable, py_script, convert_model]

    user_input = prompt + "\n"
    result = run_sample(py_command, user_input)
    output = result.stdout
    
    # Find the line with "Generated JSON with item quantities:"
    for line in output.splitlines():
        if line.startswith("Generated JSON with item quantities:"):
            json_str = line.split(":", 1)[1].strip()
            break
        # In case if the prompt is incorrect and no JSON is generated only a message is printed
        elif line.startswith(expected_keys):
            break
    else:
        pytest.fail("No generated JSON found in output.")

    # Parse and check the JSON
    try:
        data = json.loads(json_str)
    except Exception as e:
        pytest.fail(f"Output is not valid JSON: {e}")

    # Check that all expected keys are present and are integers >= 0
    for key in expected_keys:
        assert key in data, f"Missing key '{key}' in output"
        assert isinstance(data[key], int), f"Value for '{key}' is not int"
        assert data[key] >= 0, f"Value for '{key}' is negative"

    logger.info(f"Structured output sample test passed for prompt: {prompt}")
