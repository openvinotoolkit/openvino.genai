# Copyright (C) 2025 Intel Corporation
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
            "optimum-cli", "export", "openvino",
            "--model", MODEL_NAME,
            "--trust-remote-code",
            "--weight-format", "fp16",
            str(temp_path)
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
    mask_data[:, height//4:3*height//4, width//4:3*width//4, :] = 255
    return ov.Tensor(mask_data)


class TestImageGenerationCallback:
    
    def test_text2image_with_simple_callback(self, image_generation_model):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        
        callback_calls = []
        
        def callback(step, num_steps, latent):
            callback_calls.append((step, num_steps))
            return False
        
        image = pipe.generate(
            "test prompt",
            width=64,
            height=64,
            num_inference_steps=2,
            callback=callback
        )
        
        assert len(callback_calls) > 0, "Callback should be called at least once"
        assert image is not None
    
    def test_text2image_with_stateful_callback(self, image_generation_model):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        
        class ProgressTracker:
            def __init__(self):
                self.steps = []
                self.total = 0
            
            def reset(self, total):
                self.total = total
                self.steps = []
            
            def update(self, step):
                self.steps.append(step)
        
        tracker = ProgressTracker()
        
        def callback(step, num_steps, latent):
            if tracker.total != num_steps:
                tracker.reset(num_steps)
            tracker.update(step)
            return False
        
        image = pipe.generate(
            "test prompt",
            width=64,
            height=64,
            num_inference_steps=2,
            callback=callback
        )
        
        assert len(tracker.steps) > 0, "Callback should track steps"
        assert image is not None
    
    def test_text2image_callback_early_stop(self, image_generation_model):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        
        callback_calls = []
        
        def callback(step, num_steps, latent):
            callback_calls.append(step)
            return step >= 1
        
        image = pipe.generate(
            "test prompt",
            width=64,
            height=64,
            num_inference_steps=5,
            callback=callback
        )
        
        assert len(callback_calls) <= 3, "Callback should stop early"
        assert image is not None
    
    def test_text2image_multiple_generates_with_callback(self, image_generation_model):
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")
        
        for i in range(3):
            callback_calls = []
            
            def callback(step, num_steps, latent):
                callback_calls.append(step)
                return False
            
            image = pipe.generate(
                f"test prompt {i}",
                width=64,
                height=64,
                num_inference_steps=2,
                callback=callback
            )
            
            assert len(callback_calls) > 0
            assert image is not None
    
    def test_image2image_with_callback(self, image_generation_model):
        pipe = ov_genai.Image2ImagePipeline(image_generation_model, "CPU")
        
        callback_calls = []
        
        def callback(step, num_steps, latent):
            callback_calls.append((step, num_steps))
            return False
        
        input_image = get_random_image()
        
        image = pipe.generate(
            "test prompt",
            input_image,
            strength=0.8,
            callback=callback
        )
        
        assert len(callback_calls) > 0
        assert image is not None
    
    def test_inpainting_with_callback(self, image_generation_model):
        pipe = ov_genai.InpaintingPipeline(image_generation_model, "CPU")
        
        callback_calls = []
        
        def callback(step, num_steps, latent):
            callback_calls.append((step, num_steps))
            return False
        
        input_image = get_random_image()
        mask_image = get_mask_image()
        
        image = pipe.generate(
            "test prompt",
            input_image,
            mask_image,
            callback=callback
        )
        
        assert len(callback_calls) > 0
        assert image is not None
