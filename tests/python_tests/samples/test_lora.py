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

    @pytest.mark.vlm
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
    def test_visual_language_lora_alpha_switch(self, convert_model, download_test_content, prompt):
        # Check alpha-zero equivalence and output restoration after switching.
        adapter_path, image_path = download_test_content
        assert os.path.exists(image_path), f"Missing test image: {image_path}"

        image = Image.open(image_path).convert("RGB")
        image_tensor = ov.Tensor(np.array(image))

        adapter = ov_genai.Adapter(adapter_path)
        config_a = ov_genai.AdapterConfig()
        config_a.add(adapter, 2.0)
        # A and B use the same adapter and differ only by alpha.
        config_b = ov_genai.AdapterConfig()
        config_b.add(adapter, 0.0)

        pipe = ov_genai.VLMPipeline(convert_model, "CPU", ATTENTION_BACKEND="PA", adapters=config_a)

        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = 100

        result_a_first = pipe.generate(prompt, images=[image_tensor], generation_config=generation_config)
        assert len(result_a_first.texts[0]) > 0, "Generation with config A should produce output"

        result_b = pipe.generate(prompt, images=[image_tensor], generation_config=generation_config, adapters=config_b)
        assert len(result_b.texts[0]) > 0, "Generation with config B should produce output"

        # Alpha 0.0 turns the adapter off, so B must match a run with no adapter at all.
        result_no_adapter = pipe.generate(
            prompt, images=[image_tensor], generation_config=generation_config, adapters=ov_genai.AdapterConfig()
        )
        assert result_b.texts[0] == result_no_adapter.texts[0], (
            "Config B (alpha=0.0) should exactly match a no-adapter baseline"
        )

        result_a_second = pipe.generate(
            prompt, images=[image_tensor], generation_config=generation_config, adapters=config_a
        )
        assert len(result_a_second.texts[0]) > 0, "Generation after switching back to config A should produce output"

        assert result_a_second.texts[0] == result_a_first.texts[0], (
            "Switching A -> B -> A should reproduce the original config A output"
        )

    @pytest.mark.vlm
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
    def test_visual_language_lora_concat_to_single_switch(self, convert_model, download_test_content, prompt, tmp_path):
        from safetensors.torch import load_file, save_file

        adapter_path, image_path = download_test_content
        # Different ranks expose incorrect concat offsets and B row strides during subset selection.
        # Write the modified weights only to tmp_path so the downloaded adapter stays unchanged.
        weights = load_file(adapter_path)
        changed_a = changed_b = 0
        for name, tensor in weights.items():
            if ".lora_A." in name:
                assert tensor.shape[0] > 1, f"Expected rank greater than one: {name}"
                weights[name] = tensor[: tensor.shape[0] // 2].contiguous()
                changed_a += 1
            elif ".lora_B." in name:
                assert tensor.shape[1] > 1, f"Expected rank greater than one: {name}"
                weights[name] = (-tensor[:, : tensor.shape[1] // 2]).contiguous()
                changed_b += 1
        assert changed_a > 0 and changed_a == changed_b, "Expected paired LoRA A/B weights"
        second_adapter_path = tmp_path / "second_adapter.safetensors"
        save_file(weights, str(second_adapter_path))
        del weights

        adapter_a = ov_genai.Adapter(adapter_path)
        adapter_b = ov_genai.Adapter(second_adapter_path)
        config_a = ov_genai.AdapterConfig()
        config_a.add(adapter_a, 2.0)
        config_b = ov_genai.AdapterConfig()
        config_b.add(adapter_b, 1.0)
        config_ab = ov_genai.AdapterConfig()
        config_ab.add(adapter_a, 2.0)
        config_ab.add(adapter_b, 1.0)
        config_b_zero = ov_genai.AdapterConfig()
        config_b_zero.add(adapter_b, 0.0)

        with Image.open(image_path) as image:
            image_tensor = ov.Tensor(np.array(image.convert("RGB")))
        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = 12
        generation_config.do_sample = False

        def generate(pipe, config):
            result = pipe.generate(prompt, images=[image_tensor], generation_config=generation_config, adapters=config)
            assert result.texts and result.texts[0], "Generation should produce output"
            return result.texts[0]

        # Single-adapter pipelines provide baselines without using the concat-to-single view path.
        # Release each baseline before creating the next pipeline to limit peak model memory.
        expected = []
        for config in (config_a, config_b):
            baseline = ov_genai.VLMPipeline(convert_model, "CPU", ATTENTION_BACKEND="PA", adapters=config)
            expected.append(generate(baseline, config))
            del baseline

        pipe = ov_genai.VLMPipeline(convert_model, "CPU", ATTENTION_BACKEND="PA", adapters=config_ab)
        first_ab = generate(pipe, config_ab)
        assert generate(pipe, config_a) == expected[0], "First concat interval should match adapter A alone"
        assert generate(pipe, config_b) == expected[1], "Second concat interval should match adapter B alone"
        assert generate(pipe, config_ab) == first_ab, "Switching back should restore the combined output"
        assert generate(pipe, config_b_zero) == generate(pipe, ov_genai.AdapterConfig()), (
            "Adapter B with alpha zero should match the no-adapter baseline"
        )
        assert generate(pipe, config_b) == expected[1], "Adapter B should be restored after alpha-zero switching"
        assert generate(pipe, config_a) == expected[0], "Repeated selection should restore adapter A"
