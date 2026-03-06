# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import subprocess  # nosec B404
import logging
from pathlib import Path
import numpy as np
import openvino as ov
import openvino_genai as ov_genai

from utils.constants import get_ov_cache_converted_models_dir
from utils.atomic_download import AtomicDownloadManager
from utils.network import retry_request

logger = logging.getLogger(__name__)

MODEL_ID = "tiny-random-latent-consistency"
MODEL_NAME = "echarlaix/tiny-random-latent-consistency"


@pytest.fixture(scope="module")
def image_generation_model():
    models_dir = get_ov_cache_converted_models_dir()
    model_path = Path(models_dir) / MODEL_ID / MODEL_NAME

    manager = AtomicDownloadManager(model_path)

    def convert_model(temp_path: Path) -> None:
        command = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            MODEL_NAME,
            "--trust-remote-code",
            "--weight-format",
            "fp16",
            str(temp_path),
        ]
        logger.info(f"Conversion command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True, text=True, capture_output=True))

    try:
        manager.execute(convert_model)
    except subprocess.CalledProcessError as error:
        logger.exception(f"optimum-cli returned {error.returncode}. Output:\n{error.output}")
        raise

    return str(model_path)


def get_random_image(height: int = 64, width: int = 64) -> ov.Tensor:
    image_data = np.random.randint(0, 255, (1, height, width, 3), dtype=np.uint8)
    return ov.Tensor(image_data)


def get_mask_image(height: int = 64, width: int = 64) -> ov.Tensor:
    mask_data = np.zeros((1, height, width, 3), dtype=np.uint8)
    mask_data[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4, :] = 255
    return ov.Tensor(mask_data)


GENERATE_KWARGS = dict(width=64, height=64, num_inference_steps=2)
NUM_CALLS = 3


class TestText2ImageMultipleGenerations:
    def test_multiple_generate_no_kwargs(self, image_generation_model):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            image = pipe.generate(f"prompt {i}", **GENERATE_KWARGS)
            assert image is not None

    def test_multiple_generate_with_cpp_generator(self, image_generation_model):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = ov_genai.CppStdGenerator(42)
            image = pipe.generate(f"prompt {i}", generator=gen, **GENERATE_KWARGS)
            assert image is not None

    def test_multiple_generate_with_torch_generator(self, image_generation_model):
        torch = pytest.importorskip("torch")
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = ov_genai.TorchGenerator(42)
            image = pipe.generate(f"prompt {i}", generator=gen, **GENERATE_KWARGS)
            assert image is not None

    def test_multiple_generate_with_callback(self, image_generation_model):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            steps = []

            def callback(step, num_steps, latent):
                steps.append(step)
                return False

            image = pipe.generate(f"prompt {i}", callback=callback, **GENERATE_KWARGS)
            assert image is not None
            assert len(steps) > 0

    def test_multiple_generate_with_torch_generator_and_callback(self, image_generation_model):
        torch = pytest.importorskip("torch")
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = ov_genai.TorchGenerator(42)
            steps = []

            def callback(step, num_steps, latent):
                steps.append(step)
                return False

            image = pipe.generate(f"prompt {i}", generator=gen, callback=callback, **GENERATE_KWARGS)
            assert image is not None
            assert len(steps) > 0


class TestImage2ImageMultipleGenerations:
    def test_multiple_generate_no_kwargs(self, image_generation_model):
        pipe = ov_genai.Image2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            image = pipe.generate(f"prompt {i}", get_random_image(), strength=0.8, **GENERATE_KWARGS)
            assert image is not None

    def test_multiple_generate_with_torch_generator(self, image_generation_model):
        torch = pytest.importorskip("torch")
        pipe = ov_genai.Image2ImagePipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = ov_genai.TorchGenerator(42)
            image = pipe.generate(f"prompt {i}", get_random_image(), strength=0.8, generator=gen, **GENERATE_KWARGS)
            assert image is not None


class TestInpaintingMultipleGenerations:
    def test_multiple_generate_no_kwargs(self, image_generation_model):
        pipe = ov_genai.InpaintingPipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            image = pipe.generate(f"prompt {i}", get_random_image(), get_mask_image(), strength=0.8, **GENERATE_KWARGS)
            assert image is not None

    def test_multiple_generate_with_torch_generator(self, image_generation_model):
        torch = pytest.importorskip("torch")
        pipe = ov_genai.InpaintingPipeline(image_generation_model, "CPU")
        for i in range(NUM_CALLS):
            gen = ov_genai.TorchGenerator(42)
            image = pipe.generate(
                f"prompt {i}", get_random_image(), get_mask_image(), strength=0.8, generator=gen, **GENERATE_KWARGS
            )
            assert image is not None
