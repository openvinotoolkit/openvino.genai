# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import subprocess  # nosec B404
import logging
from pathlib import Path

import numpy as np
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


class TestAutoEncoderKLLTXVideoEncoder:
    @pytest.fixture
    def require_encoder(self, video_generation_model):
        encoder_path = Path(video_generation_model) / "vae_encoder"
        if not encoder_path.exists():
            pytest.skip("vae_encoder not available in test model")

    def test_constructor_with_encoder(self, video_generation_model, require_encoder):
        encoder_path = Path(video_generation_model) / "vae_encoder"
        decoder_path = Path(video_generation_model) / "vae_decoder"
        vae = ov_genai.AutoencoderKLLTXVideo(str(encoder_path), str(decoder_path))
        assert vae is not None

    def test_encode_without_compile_raises(self, video_generation_model, require_encoder):
        import openvino as ov
        import numpy as np

        encoder_path = Path(video_generation_model) / "vae_encoder"
        decoder_path = Path(video_generation_model) / "vae_decoder"
        vae = ov_genai.AutoencoderKLLTXVideo(str(encoder_path), str(decoder_path))  # no compile
        generator = ov_genai.CppStdGenerator(42)
        dummy = ov.Tensor(np.zeros([1, 3, 9, 32, 32], dtype=np.float32))

        with pytest.raises(Exception, match="must be compiled first"):
            vae.encode(dummy, generator)

    def test_encode_without_encoder_raises(self, video_generation_model):
        import openvino as ov
        import numpy as np

        decoder_path = Path(video_generation_model) / "vae_decoder"
        # Use 1-arg constructor + explicit compile to avoid overload ambiguity with 2-path constructor
        vae = ov_genai.AutoencoderKLLTXVideo(str(decoder_path))
        vae.compile("CPU")
        generator = ov_genai.CppStdGenerator(42)
        dummy = ov.Tensor(np.zeros([1, 3, 9, 32, 32], dtype=np.float32))

        with pytest.raises(Exception, match="without 'VAE encoder' capability"):
            vae.encode(dummy, generator)

    def test_encode_output_shape(self, video_generation_model, require_encoder):
        import openvino as ov
        import numpy as np

        encoder_path = Path(video_generation_model) / "vae_encoder"
        decoder_path = Path(video_generation_model) / "vae_decoder"
        vae = ov_genai.AutoencoderKLLTXVideo(str(encoder_path), str(decoder_path), "CPU")
        config = vae.get_config()
        generator = ov_genai.CppStdGenerator(42)

        video = ov.Tensor(np.zeros([1, 3, 9, 32, 32], dtype=np.float32))
        latent = vae.encode(video, generator)

        assert latent is not None
        shape = latent.shape
        assert len(shape) == 5, f"Expected 5D latent [B, C, F, H, W], got shape {shape}"
        assert shape[0] == 1
        assert shape[1] == config.latent_channels

    def test_encode_is_deterministic(self, video_generation_model, require_encoder):
        import openvino as ov
        import numpy as np

        encoder_path = Path(video_generation_model) / "vae_encoder"
        decoder_path = Path(video_generation_model) / "vae_decoder"
        vae = ov_genai.AutoencoderKLLTXVideo(str(encoder_path), str(decoder_path), "CPU")

        video = ov.Tensor(np.ones([1, 3, 9, 32, 32], dtype=np.float32) * 0.5)
        latent1 = vae.encode(video, ov_genai.CppStdGenerator(42))
        latent2 = vae.encode(video, ov_genai.CppStdGenerator(42))

        np.testing.assert_array_equal(latent1.data, latent2.data)

    def test_encode_varies_with_seed(self, video_generation_model, require_encoder):
        import openvino as ov
        import numpy as np

        encoder_path = Path(video_generation_model) / "vae_encoder"
        decoder_path = Path(video_generation_model) / "vae_decoder"
        vae = ov_genai.AutoencoderKLLTXVideo(str(encoder_path), str(decoder_path), "CPU")

        video = ov.Tensor(np.ones([1, 3, 9, 32, 32], dtype=np.float32) * 0.5)
        latent1 = vae.encode(video, ov_genai.CppStdGenerator(42))
        latent2 = vae.encode(video, ov_genai.CppStdGenerator(99))

        assert not np.array_equal(latent1.data, latent2.data), \
            "Different generator seeds should produce different latents"


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


class TestTaylorSeer:
    def test_taylorseer_custom_config(self, video_generation_model):
        """Test TaylorSeer with custom cache configuration.

        Uses disable_cache_before_step = N-1 and disable_cache_after_step = N so that
        TaylorSeer is only active on the last inference step.

        Verifies via callback that latents at steps 0..N-2 are identical between baseline
        and TaylorSeer runs, while the last step differs due to Taylor prediction.
        """
        num_inference_steps = 4
        generate_kwargs = dict(height=32, width=32, num_inference_steps=num_inference_steps)

        taylorseer_config = ov_genai.TaylorSeerCacheConfig()
        taylorseer_config.cache_interval = 2
        taylorseer_config.disable_cache_before_step = num_inference_steps - 1
        taylorseer_config.disable_cache_after_step = num_inference_steps  # never disable

        baseline_latents = []
        taylorseer_latents = []

        def make_callback(latents_list):
            def callback(step, num_steps, latent):
                latents_list.append(latent.data[:].copy())
                return False

            return callback

        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")

        pipe.generate("test prompt", callback=make_callback(baseline_latents), **generate_kwargs)
        ts_result = pipe.generate(
            "test prompt",
            taylorseer_config=taylorseer_config,
            callback=make_callback(taylorseer_latents),
            **generate_kwargs,
        )

        assert ts_result.video is not None
        assert len(baseline_latents) == num_inference_steps
        assert len(taylorseer_latents) == num_inference_steps

        # Steps 0..N-2: TaylorSeer inactive, latents must be identical to baseline
        for step in range(num_inference_steps - 1):
            assert np.array_equal(baseline_latents[step], taylorseer_latents[step]), (
                f"Step {step} latents differ unexpectedly — TaylorSeer should not be active yet"
            )

        # Last step: TaylorSeer prediction was used, result must differ from baseline
        assert not np.array_equal(baseline_latents[-1], taylorseer_latents[-1]), (
            "Last step latents are identical — TaylorSeer prediction should have changed the result"
        )
