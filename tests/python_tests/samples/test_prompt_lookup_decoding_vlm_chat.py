# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR
from test_utils import run_sample


class TestPromptLookupDecodingVLMChat:
    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content",
        [
            pytest.param(
                "Qwen2-VL-2B-Instruct",
                "monalisa.jpg",
            ),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    @pytest.mark.parametrize(
        "prompts",
        [
            ["Who drew this painting?", "When did the painter live?"],
        ],
    )
    def test_prompt_lookup_decoding_vlm_chat(self, convert_model, download_test_content, prompts):
        env = os.environ.copy()
        env["OPENVINO_LOG_LEVEL"] = "0"

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/prompt_lookup_decoding_vlm_chat.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command, "\n".join(prompts), env=env)

        # Test Python sample, disable lookup decoding.
        py_command_ref = [sys.executable, py_script, convert_model, download_test_content, "--disable_lookup"]
        py_result_ref = run_sample(py_command_ref, "\n".join(prompts), env=env)

        # Compare results
        assert py_result.stdout == py_result_ref.stdout, (
            "Results should be identical when running without lookup and lookup speculative decoding enabled."
        )
