# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestPromptLookupDecodingVLM:
    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, question",
        [
            pytest.param("Qwen2-VL-2B-Instruct", "monalisa.jpg", "Who drew this painting? Please answer the answer in JSON format: {\"Painter\":\"Van Gogh\"}."),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_prompt_lookup_decoding_vlm(self, convert_model, download_test_content, question):
        env = os.environ.copy()
        env["OPENVINO_LOG_LEVEL"] = "0"
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'prompt_lookup_decoding_vlm')
        cpp_command =[cpp_sample, convert_model, download_test_content, question]
        cpp_result = run_sample(cpp_command, env=env)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/prompt_lookup_decoding_vlm.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content, question]
        py_result = run_sample(py_command, env=env)

        # Test Python sample, disable lookup decoding.
        cpp_sample_ref = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/prompt_lookup_decoding_vlm.py")
        cpp_command_ref = [sys.executable, cpp_sample_ref, convert_model, download_test_content, question, "False"]
        cpp_result_ref = run_sample(cpp_command_ref, env=env)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, "Python and CPP results should match"
        assert cpp_result_ref.stdout == cpp_result.stdout, "Results should be identical when running without lookup and lookup speculative decoding enabled."
