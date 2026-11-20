# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from conftest import SAMPLES_CPP_DIR, SAMPLES_PY_DIR
from test_utils import run_sample


@pytest.mark.classification
@pytest.mark.samples
@pytest.mark.parametrize("convert_model", ["tiny-random-minicpm-ocr-jev"], indirect=True)
@pytest.mark.parametrize("generate_test_content", ["images/lines.png"], indirect=True)
@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
def test_sample_classification(convert_model, generate_test_content, with_image):
    prompt = "<ov_genai_image_0>Classify the input." if with_image else "Classify the input."
    images = [generate_test_content] if with_image else []

    py_script = SAMPLES_PY_DIR / "classification/classification.py"
    py_result = run_sample([sys.executable, py_script, convert_model, prompt, *images])

    cpp_sample = SAMPLES_CPP_DIR / "classification"
    cpp_result = run_sample([cpp_sample, convert_model, prompt, *images])

    assert py_result.stdout, "The sample must print label scores"
    assert py_result.stdout == cpp_result.stdout, "Python and C++ results should match"
