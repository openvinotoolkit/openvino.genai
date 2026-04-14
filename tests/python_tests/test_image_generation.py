# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pathlib import Path
import openvino as ov
import openvino_genai as ov_genai

from utils.constants import NPUW_CPU_PROPERTIES
from utils.ov_genai_pipelines import should_skip_npuw_tests

FLUX_MODEL_ID = "tiny-random-flux"
SDXL_MODEL_ID = "tiny-random-sdxl"
SD3_MODEL_ID = "tiny-random-stable-diffusion-3"


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


class TestTaylorSeerImageGeneration:
    @pytest.mark.parametrize("image_generation_model", [FLUX_MODEL_ID], indirect=True)
    def test_flux_text2image_taylorseer_with_callback(self, image_generation_model):
        """Test Flux text2image with TaylorSeer and callback."""
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU")

        # Configure TaylorSeer
        taylorseer_config = ov_genai.TaylorSeerCacheConfig()
        taylorseer_config.cache_interval = 5
        taylorseer_config.disable_cache_before_step = 2
        taylorseer_config.disable_cache_after_step = -1

        generation_config = pipe.get_generation_config()
        generation_config.taylorseer_config = taylorseer_config
        pipe.set_generation_config(generation_config)

        callback_calls = []

        def callback(step, num_steps, latent):
            callback_calls.append((step, num_steps))
            return False

        image = pipe.generate("test prompt", width=64, height=64, num_inference_steps=5, callback=callback)

        assert image is not None
        assert len(callback_calls) > 0


class TestLoRAFluxImageGeneration:
    @pytest.mark.parametrize("image_generation_model", [FLUX_MODEL_ID], indirect=True)
    def test_lora_adapters_constructor(self, image_generation_model):
        """Test that LoRA adapters can be passed to the Flux pipeline constructor without error"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU", adapters=adapter_config)
        assert pipe is not None

    @pytest.mark.parametrize("image_generation_model", [FLUX_MODEL_ID], indirect=True)
    def test_lora_adapters_generate(self, image_generation_model):
        """Test that LoRA adapters can be passed to Flux generate() without error"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU", adapters=adapter_config)
        image = pipe.generate(
            "test prompt", width=64, height=64, num_inference_steps=1, adapters=adapter_config
        )
        assert image is not None

    @pytest.mark.parametrize("image_generation_model", [FLUX_MODEL_ID], indirect=True)
    def test_lora_adapters_default_from_constructor(self, image_generation_model):
        """Test that LoRA adapters passed to the Flux constructor are used by default in generate()"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU", adapters=adapter_config)
        image = pipe.generate("test prompt", width=64, height=64, num_inference_steps=1)
        assert image is not None

    @pytest.mark.parametrize("image_generation_model", [FLUX_MODEL_ID], indirect=True)
    def test_t5_encoder_has_set_adapters_method(self, image_generation_model):
        """Test that T5EncoderModel has set_adapters (used by Flux for text encoder LoRA)"""
        model_path = Path(image_generation_model) / "text_encoder_2"
        if not model_path.exists():
            pytest.skip("T5 text encoder (text_encoder_2) not present in this model")
        model = ov_genai.T5EncoderModel(str(model_path))
        model.compile("CPU")
        assert hasattr(model, "set_adapters")
        model.set_adapters(None)


class TestLoRASD3ImageGeneration:
    @pytest.mark.parametrize("image_generation_model", [SD3_MODEL_ID], indirect=True)
    def test_lora_adapters_constructor(self, image_generation_model):
        """Test that LoRA adapters can be passed to the SD3 pipeline constructor without error"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU", adapters=adapter_config)
        assert pipe is not None

    @pytest.mark.parametrize("image_generation_model", [SD3_MODEL_ID], indirect=True)
    def test_lora_adapters_generate(self, image_generation_model):
        """Test that LoRA adapters can be passed to SD3 generate() without error"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU", adapters=adapter_config)
        image = pipe.generate(
            "test prompt", width=64, height=64, num_inference_steps=1, adapters=adapter_config
        )
        assert image is not None

    @pytest.mark.parametrize("image_generation_model", [SD3_MODEL_ID], indirect=True)
    def test_lora_adapters_default_from_constructor(self, image_generation_model):
        """Test that LoRA adapters passed to the SD3 constructor are used by default in generate()"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2ImagePipeline(image_generation_model, "CPU", adapters=adapter_config)
        image = pipe.generate("test prompt", width=64, height=64, num_inference_steps=1)
        assert image is not None

    @pytest.mark.parametrize("image_generation_model", [SD3_MODEL_ID], indirect=True)
    def test_t5_encoder_has_set_adapters_method(self, image_generation_model):
        """Test that T5EncoderModel has the set_adapters method added for SD3 LoRA support"""
        model_path = Path(image_generation_model) / "text_encoder_3"
        if not model_path.exists():
            pytest.skip("T5 text encoder (text_encoder_3) not present in this model")
        model = ov_genai.T5EncoderModel(str(model_path))
        model.compile("CPU")
        assert hasattr(model, "set_adapters")
        model.set_adapters(None)


class TestImageGenerationOnNpuByNpuwCpu:
    def _construct_reshaped(self, model_dir):
        pipe = ov_genai.Text2ImagePipeline(model_dir)
        pipe.reshape(
            num_images_per_prompt=1, height=64, width=64, guidance_scale=pipe.get_generation_config().guidance_scale
        )
        return pipe

    def _get_generation_args(self):
        return {"prompt": "Will Smith eating spaghetti", "num_inference_steps": 5, "rng_seed": 69}

    @pytest.mark.skipif(**should_skip_npuw_tests())
    def test_image_generation_cpu_vs_npuw_cpu(self, image_generation_model):
        generation_args = self._get_generation_args()

        cpu_pipe = self._construct_reshaped(image_generation_model)
        cpu_pipe.compile("CPU")
        cpu_image = cpu_pipe.generate(**generation_args)

        npuw_pipe = self._construct_reshaped(image_generation_model)
        npuw_pipe.compile("NPU", **NPUW_CPU_PROPERTIES)
        npuw_image = npuw_pipe.generate(**generation_args)

        assert cpu_image.data.shape == npuw_image.data.shape
        assert (cpu_image.data == npuw_image.data).all()

    @pytest.mark.parametrize("image_generation_model", [SDXL_MODEL_ID], indirect=True)
    @pytest.mark.skipif(**should_skip_npuw_tests())
    def test_image_generation_cpu_vs_npuw_cpu_with_blob_model(self, image_generation_model):
        generation_args = self._get_generation_args()

        cpu_pipe = self._construct_reshaped(image_generation_model)
        cpu_pipe.compile("CPU")
        cpu_image = cpu_pipe.generate(**generation_args)

        npuw_pipe = self._construct_reshaped(image_generation_model)
        npuw_pipe.compile("NPU", **NPUW_CPU_PROPERTIES)
        npuw_pipe.export_model("tmp_blob_model")
        imported_npuw_pipe = ov_genai.Text2ImagePipeline(
            image_generation_model, "NPU", blob_path="tmp_blob_model", **NPUW_CPU_PROPERTIES
        )
        imported_npuw_image = imported_npuw_pipe.generate(**generation_args)

        assert cpu_image.data.shape == imported_npuw_image.data.shape
        assert (cpu_image.data == imported_npuw_image.data).all()
