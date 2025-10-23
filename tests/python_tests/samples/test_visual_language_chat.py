# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import subprocess # nosec B404
import sys

from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR, SAMPLES_C_DIR
from test_utils import run_sample

class TestVisualLanguageChat:
    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, questions",
        [
            pytest.param("llava-1.5-7b-hf", "monalisa.jpg", 'Who drew this painting?\nWhen did the painter live?'),
            pytest.param("llava-v1.6-mistral-7b-hf", "monalisa.jpg", 'Who drew this painting?\nWhen did the painter live?'),
            pytest.param("InternVL2-1B", "monalisa.jpg", 'Who drew this painting?\nWhen did the painter live?'),
            pytest.param("Qwen2-VL-2B-Instruct", "monalisa.jpg", 'Who drew this painting?\nWhen did the painter live?'),
            pytest.param("tiny-random-minicpmv-2_6", "images/image.png", 'What is unusual on this image?\nGo on.')
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_visual_language_chat(self, convert_model, download_test_content, questions):
        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'visual_language_chat')
        cpp_command =[cpp_sample, convert_model, download_test_content]
        cpp_result = run_sample(cpp_command, questions)

        # Test C sample
        c_sample = os.path.join(SAMPLES_C_DIR, 'vlm_pipeline_c')
        c_command =[c_sample, convert_model, "CPU", download_test_content]
        c_result = run_sample(c_command, questions)

        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/visual_language_chat.py")
        py_command = [sys.executable, py_script, convert_model, download_test_content]
        py_result = run_sample(py_command, questions)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
        assert cpp_result.stdout == c_result.stdout, f"Results should match"

    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, questions",
        [
            pytest.param("tiny-random-minicpmv-2_6", 'Describe the images?'),
        ],
        indirect=["convert_model"],
    )
    @pytest.mark.parametrize("download_test_content", ["images/image.png"], indirect=True)
    @pytest.mark.parametrize("generate_test_content", ["images/lines.png"], indirect=True)
    def test_sample_visual_language_chat_images(self, convert_model, download_test_content, generate_test_content, questions):
        # Test Python sample
        py_script = os.path.join(SAMPLES_PY_DIR, "visual_language_chat/visual_language_chat.py")
        py_command = [sys.executable, py_script, convert_model, os.path.dirname(generate_test_content)]
        py_result = run_sample(py_command, questions)

        # Test CPP sample
        cpp_sample = os.path.join(SAMPLES_CPP_DIR, 'visual_language_chat')
        cpp_command =[cpp_sample, convert_model, os.path.dirname(generate_test_content)]
        cpp_result = run_sample(cpp_command, questions)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"
