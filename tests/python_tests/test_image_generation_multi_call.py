# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import openvino as ov
import openvino_genai as ov_genai


def get_random_image(height: int = 64, width: int = 64) -> ov.Tensor:
    image_data = np.random.randint(0, 255, (1, height, width, 3), dtype=np.uint8)
    return ov.Tensor(image_data)


def get_mask_image(height: int = 64, width: int = 64) -> ov.Tensor:
    mask_data = np.zeros((1, height, width, 3), dtype=np.uint8)
    mask_data[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4, :] = 255
    return ov.Tensor(mask_data)


def create_generator(generator_type):
    if generator_type == "cpp_std":
        return ov_genai.CppStdGenerator(42)
    if generator_type == "torch":
        pytest.importorskip("torch")
        return ov_genai.TorchGenerator(42)
    return None


GENERATE_KWARGS = dict(width=64, height=64, num_inference_steps=2)
NUM_CALLS = 3


class TestText2ImageMultipleGenerations:
    @pytest.mark.parametrize("generator_type", [None, "cpp_std", "torch"])
    def test_multiple_generate(self, image_generation_model, generator_type):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = create_generator(generator_type)
            kwargs = {**GENERATE_KWARGS, **({"generator": gen} if gen else {})}
            image = pipe.generate(f"prompt {i}", **kwargs)
            assert image is not None

    @pytest.mark.parametrize("generator_type", [None, "torch"])
    def test_multiple_generate_with_callback(self, image_generation_model, generator_type):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = create_generator(generator_type)
            steps = []

            def callback(step, num_steps, latent):
                steps.append(step)
                return False

            kwargs = {**GENERATE_KWARGS, "callback": callback, **({"generator": gen} if gen else {})}
            image = pipe.generate(f"prompt {i}", **kwargs)
            assert image is not None
            assert len(steps) > 0


class TestImage2ImageMultipleGenerations:
    @pytest.mark.parametrize("generator_type", [None, "cpp_std", "torch"])
    def test_multiple_generate(self, image_generation_model, generator_type):
        pipe = ov_genai.Image2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = create_generator(generator_type)
            kwargs = {**GENERATE_KWARGS, "strength": 0.8, **({"generator": gen} if gen else {})}
            image = pipe.generate(f"prompt {i}", get_random_image(), **kwargs)
            assert image is not None


class TestInpaintingMultipleGenerations:
    @pytest.mark.parametrize("generator_type", [None, "cpp_std", "torch"])
    def test_multiple_generate(self, image_generation_model, generator_type):
        pipe = ov_genai.InpaintingPipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = create_generator(generator_type)
            kwargs = {**GENERATE_KWARGS, "strength": 0.8, **({"generator": gen} if gen else {})}
            image = pipe.generate(f"prompt {i}", get_random_image(), get_mask_image(), **kwargs)
            assert image is not None
