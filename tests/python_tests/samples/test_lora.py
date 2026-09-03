# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import sys

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from conftest import SAMPLES_PY_DIR, SAMPLES_CPP_DIR
from test_utils import run_sample
from PIL import Image

class TestLora:
    @pytest.mark.llm
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["TinyStories-1M"], indirect=True)
    @pytest.mark.parametrize("sample_args", ["How to create a table with two columns, one of them has type float, another one has type int?"])
    @pytest.mark.parametrize("download_test_content", ["adapter_model.safetensors"], indirect=True)
    def test_python_sample_lora(self, convert_model, download_test_content, sample_args):      
        py_script = SAMPLES_PY_DIR / "text_generation/lora_greedy_causal_lm.py"
        py_command = [sys.executable, py_script, convert_model, download_test_content, sample_args]
        run_sample(py_command)

    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, prompt, alpha",
        [
            pytest.param(
                "Qwen2-VL-2B-Instruct",
                ("qwen2b_lora_100_adapter_model.safetensors", "monalisa.jpg"),
                "Who drew this painting?",
                "2.0",
            ),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_visual_language_lora(self, convert_model, download_test_content, prompt, alpha):
        adapter_path, image_path = download_test_content
        assert os.path.exists(image_path), f"Missing test image: {image_path}"

        # Test CPP sample
        cpp_sample = SAMPLES_CPP_DIR / "visual_language_lora"
        cpp_command = [cpp_sample, convert_model, image_path, prompt, adapter_path, alpha]
        cpp_result = run_sample(cpp_command)

        # Test Python sample
        py_script = SAMPLES_PY_DIR / "visual_language_chat/visual_language_lora.py"
        py_command = [sys.executable, py_script, convert_model, image_path, prompt, adapter_path, alpha]
        py_result = run_sample(py_command)

        # Compare results
        assert py_result.stdout == cpp_result.stdout, f"Results should match"

    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, prompt, alpha",
        [
            pytest.param(
                "Qwen2-VL-2B-Instruct",
                ("qwen2b_lora_100_adapter_model.safetensors", "monalisa.jpg"),
                "Who drew this painting?",
                "2.0",
            ),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_visual_language_lora_multi_alpha_zero_matches_single(
        self, convert_model, download_test_content, prompt, alpha
    ):
        adapter_path, image_path = download_test_content
        assert os.path.exists(image_path), f"Missing test image: {image_path}"

        # Baseline: single adapter
        cpp_sample = SAMPLES_CPP_DIR / "visual_language_lora"
        cpp_single_command = [cpp_sample, convert_model, image_path, prompt, adapter_path, alpha]
        cpp_single_result = run_sample(cpp_single_command)

        py_script = SAMPLES_PY_DIR / "visual_language_chat/visual_language_lora.py"
        py_single_command = [sys.executable, py_script, convert_model, image_path, prompt, adapter_path, alpha]
        py_single_result = run_sample(py_single_command)
        assert py_single_result.stdout == cpp_single_result.stdout, "Single-LoRA C++/Python results should match"

        # Multi-LoRA: add the same adapter with alpha=0.0)
        cpp_multi_command = [cpp_sample, convert_model, image_path, prompt, adapter_path, alpha, adapter_path, "0.0"]
        cpp_multi_result = run_sample(cpp_multi_command)

        py_multi_command = [
            sys.executable,
            py_script,
            convert_model,
            image_path,
            prompt,
            adapter_path,
            alpha,
            adapter_path,
            "0.0",
        ]
        py_multi_result = run_sample(py_multi_command)

        assert py_multi_result.stdout == cpp_multi_result.stdout, "Multi-LoRA C++/Python results should match"
        assert cpp_multi_result.stdout == cpp_single_result.stdout, (
            "Multi-LoRA (with alpha=0) should match single-LoRA output"
        )
        assert py_multi_result.stdout == py_single_result.stdout, (
            "Multi-LoRA (with alpha=0) should match single-LoRA output"
        )

    @pytest.mark.vlm
    @pytest.mark.samples
    @pytest.mark.parametrize(
        "convert_model, download_test_content, prompt",
        [
            pytest.param(
                "Qwen2-VL-2B-Instruct",
                ("qwen2b_lora_100_adapter_model.safetensors", "monalisa.jpg"),
                "Who drew this painting?",
            ),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_sample_visual_language_lora_multi_both_nonzero(self, convert_model, download_test_content, prompt):
        adapter_path, image_path = download_test_content
        assert os.path.exists(image_path), f"Missing test image: {image_path}"

        cpp_sample = SAMPLES_CPP_DIR / "visual_language_lora"
        cpp_command = [cpp_sample, convert_model, image_path, prompt, adapter_path, "1.0", adapter_path, "1.0"]
        cpp_result = run_sample(cpp_command)

        py_script = SAMPLES_PY_DIR / "visual_language_chat/visual_language_lora.py"
        py_command = [
            sys.executable,
            py_script,
            convert_model,
            image_path,
            prompt,
            adapter_path,
            "1.0",
            adapter_path,
            "1.0",
        ]
        py_result = run_sample(py_command)

        assert py_result.stdout == cpp_result.stdout, "Multi-LoRA C++/Python results should match"

    @pytest.mark.vlm
    @pytest.mark.parametrize(
        "convert_model, download_test_content, prompt, alpha",
        [
            pytest.param(
                "Qwen2-VL-2B-Instruct",
                ("qwen2b_lora_100_adapter_model.safetensors", "monalisa.jpg"),
                "Who drew this painting?",
                2.0,
            ),
        ],
        indirect=["convert_model", "download_test_content"],
    )
    def test_visual_language_lora_pa_backend(self, convert_model, download_test_content, prompt, alpha):
        adapter_path, image_path = download_test_content
        assert os.path.exists(image_path), f"Missing test image: {image_path}"

        image = Image.open(image_path).convert("RGB")
        image_tensor = ov.Tensor(np.array(image))

        adapter = ov_genai.Adapter(adapter_path)
        adapter_config = ov_genai.AdapterConfig()
        adapter_config.add(adapter, alpha)

        pipe = ov_genai.VLMPipeline(convert_model, "CPU", ATTENTION_BACKEND="PA", adapters=adapter_config)

        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = 100

        result_with_lora = pipe.generate(
            prompt,
            images=[image_tensor],
            generation_config=generation_config,
        )
        assert len(result_with_lora.texts[0]) > 0, "Generation with LoRA should produce output"

        result_without_lora = pipe.generate(
            prompt,
            images=[image_tensor],
            generation_config=generation_config,
            adapters=ov_genai.AdapterConfig(),
        )
        assert len(result_without_lora.texts[0]) > 0, "Generation without LoRA should produce output"
