# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample

class TestEncryptedVLM:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["tiny-random-minicpmv-2_6"], indirect=True)
    @pytest.mark.parametrize("sample_args", ["Describe the images."])
    @pytest.mark.parametrize("download_test_content", ["images/image.png"], indirect=True)
    @pytest.mark.parametrize("generate_test_content", ["images/lines.png"], indirect=True)
    def test_sample_encrypted_lm(self, convert_model, download_test_content, generate_test_content, sample_args):
        env = os.environ.copy()
        env["OPENVINO_LOG_LEVEL"] = "0"
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'encrypted_model_vlm')
        cpp_command =[cpp_sample, convert_model, os.path.dirname(generate_test_content), sample_args]
        cpp_result = run_sample(cpp_command, env=env)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/encrypted_model_vlm.py")
        py_command = [sys.executable, py_script, convert_model, os.path.dirname(generate_test_content), sample_args]
        py_result = run_sample(py_command, env=env)

        # Test common sample
        py_common_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/visual_language_chat.py")
        py_common_command = [sys.executable, py_common_script, convert_model, os.path.dirname(generate_test_content)]
        py_common_result = run_sample(py_common_command, sample_args, env)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
        # results from visual_language_chat sample also contain additional outputs like "question:".
        # So just check if results of encrypted_model_vlm sample is a substring of it.
        assert py_result.stdout in py_common_result.stdout, f"Results should match"
