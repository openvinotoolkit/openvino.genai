# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import subprocess  # nosec B404
import logging
from pathlib import Path

import openvino_genai as ov_genai

from utils.constants import get_ov_cache_converted_models_dir
from utils.atomic_download import AtomicDownloadManager
from utils.network import retry_request

logger = logging.getLogger(__name__)

MODEL_ID = "tiny-random-ltx-video"
MODEL_NAME = "optimum-intel-internal-testing/tiny-random-ltx-video"


@pytest.fixture(scope="module")
def video_generation_model() -> str:
    models_dir = get_ov_cache_converted_models_dir()
    model_path = Path(models_dir) / MODEL_ID / MODEL_NAME

    manager = AtomicDownloadManager(model_path)

    def convert_model(temp_path: Path) -> None:
        command = ["optimum-cli", "export", "openvino", "--model", MODEL_NAME, "--trust-remote-code", str(temp_path)]
        logger.info(f"Conversion command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True, text=True, capture_output=True))

    try:
        manager.execute(convert_model)
    except subprocess.CalledProcessError as error:
        logger.exception(f"optimum-cli returned {error.returncode}. Output:\n{error.output}")
        raise

    return str(model_path)


class TestVideoGenerationConfig:
    def test_config_default_values(self):
        config = ov_genai.VideoGenerationConfig()
        assert config.num_inference_steps == -1  # sentinel value, replaced by pipeline
        assert config.guidance_scale >= 1.0

    def test_config_video_specific_fields(self):
        config = ov_genai.VideoGenerationConfig()
        assert hasattr(config, "num_frames")
        assert hasattr(config, "frame_rate")
        assert hasattr(config, "num_videos_per_prompt")
        assert hasattr(config, "guidance_rescale")

    def test_config_inherited_fields(self):
        config = ov_genai.VideoGenerationConfig()
        assert hasattr(config, "height")
        assert hasattr(config, "width")
        assert hasattr(config, "guidance_scale")
        assert hasattr(config, "num_inference_steps")
        assert hasattr(config, "max_sequence_length")

    def test_config_update(self):
        config = ov_genai.VideoGenerationConfig()
        config.num_frames = 17
        config.height = 32
        config.width = 64
        assert config.num_frames == 17
        assert config.height == 32
        assert config.width == 64

    def test_config_validate_invalid(self):
        pipe_path = get_ov_cache_converted_models_dir() / MODEL_ID / MODEL_NAME
        if not pipe_path.exists():
            pytest.skip("Model not available for validation test")

        pipe = ov_genai.Text2VideoPipeline(str(pipe_path), "CPU")
        config = ov_genai.VideoGenerationConfig()
        config.guidance_scale = 0.5
        config.negative_prompt = "bad quality"

        with pytest.raises(Exception):
            pipe.set_generation_config(config)


class TestText2VideoPipelineConstructor:
    def test_constructor_path_only(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        assert pipe is not None

    def test_constructor_with_device(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        assert pipe is not None


class TestText2VideoPipelineGenerate:
    def test_generate_basic(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", height=32, width=32, num_frames=9, num_inference_steps=2)
        assert result is not None
        assert result.video is not None

    def test_generate_with_negative_prompt(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate(
            "test prompt",
            negative_prompt="bad quality",
            height=32,
            width=32,
            num_frames=9,
            num_inference_steps=2,
            guidance_scale=3.0,
        )
        assert result is not None
        assert result.video is not None

    def test_generate_with_callback(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")

        callback_calls = []

        def callback(step, num_steps, latent):
            callback_calls.append((step, num_steps))
            return False

        result = pipe.generate(
            "test prompt", height=32, width=32, num_frames=9, num_inference_steps=2, callback=callback
        )

        assert result.video is not None
        assert len(callback_calls) == 2
        assert callback_calls[0] == (0, 2)
        assert callback_calls[1] == (1, 2)

    def test_generate_callback_early_stop(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")

        def callback(step, num_steps, latent):
            return step >= 1

        result = pipe.generate(
            "test prompt", height=32, width=32, num_frames=9, num_inference_steps=5, callback=callback
        )

        assert result.video is not None


class TestText2VideoPipelineConfig:
    def test_get_generation_config(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        config = pipe.get_generation_config()
        assert config is not None
        assert isinstance(config, ov_genai.VideoGenerationConfig)

    def test_set_generation_config(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        config = ov_genai.VideoGenerationConfig()
        config.num_frames = 17
        config.height = 64
        config.width = 64
        config.num_inference_steps = 3
        pipe.set_generation_config(config)

        retrieved_config = pipe.get_generation_config()
        assert retrieved_config.num_frames == 17


class TestVideoGenerationResult:
    def test_result_has_video(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", height=32, width=32, num_frames=9, num_inference_steps=2)
        assert hasattr(result, "video")
        assert result.video is not None

    def test_result_has_perf_metrics(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", height=32, width=32, num_frames=9, num_inference_steps=2)
        assert hasattr(result, "perf_metrics")


class TestGenerators:
    def test_cpp_std_generator(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        generator = ov_genai.CppStdGenerator(42)

        result = pipe.generate(
            "test prompt", height=32, width=32, num_frames=9, num_inference_steps=2, generator=generator
        )
        assert result.video is not None


class TestLTXVideoTransformer3DModel:
    def test_constructor(self, video_generation_model):
        model_path = Path(video_generation_model) / "transformer"
        if model_path.exists():
            model = ov_genai.LTXVideoTransformer3DModel(str(model_path))
            assert model is not None

    def test_get_config(self, video_generation_model):
        model_path = Path(video_generation_model) / "transformer"
        if model_path.exists():
            model = ov_genai.LTXVideoTransformer3DModel(str(model_path))
            config = model.get_config()
            assert config is not None
            assert hasattr(config, "in_channels")
            assert hasattr(config, "patch_size")


class TestAutoEncoderKLLTXVideo:
    def test_constructor(self, video_generation_model):
        model_path = Path(video_generation_model) / "vae_decoder"
        if model_path.exists():
            vae = ov_genai.AutoencoderKLLTXVideo(str(model_path))
            assert vae is not None

    def test_get_config(self, video_generation_model):
        model_path = Path(video_generation_model) / "vae_decoder"
        if model_path.exists():
            vae = ov_genai.AutoencoderKLLTXVideo(str(model_path))
            config = vae.get_config()
            assert config is not None
            assert hasattr(config, "latent_channels")
            assert hasattr(config, "scaling_factor")


class TestText2VideoPipelineAdvanced:
    def test_reshape(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        pipe.reshape(1, 9, 32, 32, 3.0)
        pipe.compile("CPU")

        result = pipe.generate("test prompt", num_inference_steps=2)
        assert result.video is not None

    def test_generate_without_cfg_default_compile(self, video_generation_model):
        """Regression test: direct-compile constructor should work with guidance_scale <= 1."""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate(
            "test prompt",
            guidance_scale=1.0,
            height=32,
            width=32,
            num_frames=9,
            num_inference_steps=2,
        )
        assert result.video is not None

    def test_generate_without_cfg_after_reshape_with_cfg(self, video_generation_model):
        """Test: reshape with CFG then generate without CFG should raise error."""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        pipe.reshape(1, 9, 32, 32, 3.0)
        pipe.compile("CPU")

        with pytest.raises(RuntimeError, match="guidance_scale <= 1 requested, but the compiled model expects CFG"):
            pipe.generate(
                "test prompt",
                guidance_scale=1.0,
                height=32,
                width=32,
                num_frames=9,
                num_inference_steps=2,
            )

    def test_generate_with_cfg_after_reshape_without_cfg(self, video_generation_model):
        """Regression test: reshape without CFG then generate with CFG triggers rebuild."""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        pipe.reshape(1, 9, 32, 32, 1.0)
        pipe.compile("CPU")

        result = pipe.generate(
            "test prompt",
            guidance_scale=3.0,
            height=32,
            width=32,
            num_frames=9,
            num_inference_steps=2,
        )
        assert result.video is not None
