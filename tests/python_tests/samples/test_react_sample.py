# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
 
import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_JS_DIR
from test_utils import run_sample
    
class TestReactSample:
    @pytest.mark.llm
    @pytest.mark.agent
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["TinyLlama-1.1B-Chat-v1.0"], indirect=True)
    def test_react_sample_refs(self, request, convert_model):
        if sys.platform == 'darwin':
            pytest.xfail("Ticket 173586")
        # Python test
        py_script = os.path.join(SAMPLES_PY_DIR, "text_generation/react_sample.py")
        py_command = [sys.executable, py_script, convert_model]
        py_result = run_sample(py_command)

        # Test JS sample
        js_sample = os.path.join(SAMPLES_JS_DIR, "text_generation/react_sample.js")
        js_command =['node', js_sample, convert_model]
        js_result = run_sample(js_command)

        assert py_result.stdout == js_result.stdout, f"Results should match"

