# Copyright (C) 2025-2026 Intel Corporation
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

LTX_VIDEO_MODEL_ID = "tiny-random-ltx-video"
LTX2_MODEL_ID = "tiny-random-ltx2"

VIDEO_GEN_MODELS = {
    LTX_VIDEO_MODEL_ID: "optimum-intel-internal-testing/tiny-random-ltx-video",
    LTX2_MODEL_ID: "optimum-intel-internal-testing/tiny-random-ltx2",
}

DEFAULT_VIDEO_GEN_MODEL_ID = LTX_VIDEO_MODEL_ID

GEN_KWARGS = dict(height=32, width=32, num_frames=9, num_inference_steps=2)


@pytest.fixture(scope="module")
def video_generation_model(request) -> str:
    model_id = getattr(request, "param", DEFAULT_VIDEO_GEN_MODEL_ID)
    model_name = VIDEO_GEN_MODELS[model_id]
    models_dir = get_ov_cache_converted_models_dir()
    model_path = Path(models_dir) / model_id / model_name

    manager = AtomicDownloadManager(model_path)

    def convert_model(temp_path: Path) -> None:
        command = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            model_name,
            "--trust-remote-code",
            str(temp_path),
        ]
        logger.info(f"Conversion command: {' '.join(command)}")
        retry_request(lambda: subprocess.run(command, check=True, text=True, encoding="utf-8", capture_output=True))

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

    def test_config_validate_guidance_scale_with_negative_prompt(self, video_generation_model):
        """guidance_scale <= 1 with negative_prompt is accepted (warning only)."""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        config = ov_genai.VideoGenerationConfig()
        config.guidance_scale = 0.5
        config.negative_prompt = "bad quality"

        pipe.set_generation_config(config)
        retrieved = pipe.get_generation_config()
        assert retrieved.guidance_scale == pytest.approx(0.5)
        assert retrieved.negative_prompt == "bad quality"


class TestVideoGenerationPipelines:
    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_constructor_path_only(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        assert pipe is not None

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_constructor_with_device(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        assert pipe is not None

    @pytest.mark.parametrize(
        ("video_generation_model", "audio_sample_rate"),
        [(LTX_VIDEO_MODEL_ID, 0), (LTX2_MODEL_ID, 24000)],
        indirect=["video_generation_model"],
    )
    def test_generate_basic(self, video_generation_model, audio_sample_rate):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]
        assert result.video.element_type.to_dtype() == np.uint8
        assert hasattr(result, "perf_metrics")
        assert result.audio_sample_rate == audio_sample_rate
        if audio_sample_rate:
            audio_shape = list(result.audio.shape)
            assert len(audio_shape) == 3
            assert audio_shape[0] == 1
            assert audio_shape[1] == 2
            assert audio_shape[2] > 0
        else:
            assert result.audio.size == 0

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID], indirect=True)
    def test_audio_guidance_scale_rejected(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        with pytest.raises(RuntimeError, match="audio_guidance_scale"):
            pipe.generate("test prompt", audio_guidance_scale=5.0, **GEN_KWARGS)

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_generate_with_negative_prompt(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", negative_prompt="bad quality", guidance_scale=3.0, **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_generate_with_guidance_rescale(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate(
            "test prompt",
            negative_prompt="bad quality",
            guidance_scale=3.0,
            guidance_rescale=0.7,
            **GEN_KWARGS,
        )
        assert result.video.shape == [1, 9, 32, 32, 3]

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_guidance_rescale_differs_from_no_rescale(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")

        def run(guidance_rescale):
            result = pipe.generate(
                "test prompt",
                negative_prompt="blurry, low quality, distorted",
                guidance_scale=3.0,
                guidance_rescale=guidance_rescale,
                generator=ov_genai.CppStdGenerator(42),
                **GEN_KWARGS,
            )
            return np.array(result.video.data)

        assert not np.array_equal(run(0.0), run(0.7))

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_generate_with_callback(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        callback_calls = []

        def callback(step, num_steps, latent):
            callback_calls.append((step, num_steps))
            return False

        result = pipe.generate("test prompt", callback=callback, **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]
        assert callback_calls == [(0, 2), (1, 2)]

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_generate_callback_early_stop(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")

        def callback(step, num_steps, latent):
            return True

        result = pipe.generate("test prompt", callback=callback, **GEN_KWARGS)
        assert len(list(result.video.shape)) == 0

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_generate_deterministic_with_seed(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")

        def run():
            result = pipe.generate("test prompt", generator=ov_genai.CppStdGenerator(42), **GEN_KWARGS)
            return np.array(result.video.data), np.array(result.audio.data)

        first_video, first_audio = run()
        second_video, second_audio = run()
        assert np.array_equal(first_video, second_video)
        assert np.array_equal(first_audio, second_audio)

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_num_videos_per_prompt(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", num_videos_per_prompt=2, **GEN_KWARGS)
        assert result.video.shape == [2, 9, 32, 32, 3]
        if result.audio_sample_rate:
            assert list(result.audio.shape)[0] == 2

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_num_frames_floored_with_matching_audio(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", **dict(GEN_KWARGS, num_frames=10))
        reference = pipe.generate("test prompt", **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]
        assert result.audio.shape == reference.audio.shape

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_get_generation_config(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        config = pipe.get_generation_config()
        assert isinstance(config, ov_genai.VideoGenerationConfig)

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
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

    @pytest.mark.parametrize("video_generation_model", [LTX2_MODEL_ID], indirect=True)
    def test_ltx2_default_config(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        config = pipe.get_generation_config()
        assert config.guidance_scale == pytest.approx(4.0)
        assert config.height == 512
        assert config.width == 768
        assert config.num_frames == 121
        assert config.num_inference_steps == 40
        assert config.max_sequence_length == 1024
        assert config.audio_guidance_scale is None

    @pytest.mark.parametrize("video_generation_model", [LTX2_MODEL_ID], indirect=True)
    def test_audio_guidance_scale_roundtrip(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        config = pipe.get_generation_config()
        config.audio_guidance_scale = 7.0
        pipe.set_generation_config(config)
        assert pipe.get_generation_config().audio_guidance_scale == pytest.approx(7.0)

    @pytest.mark.parametrize("video_generation_model", [LTX2_MODEL_ID], indirect=True)
    def test_audio_guidance_scale_affects_audio(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")

        # negative prompt must differ from the prompt in token length: the tiny-random tokenizer
        # maps every input to the same token id, so equal lengths make CFG a no-op
        def run(audio_guidance_scale):
            return pipe.generate(
                "test prompt",
                negative_prompt="blurry, low quality, distorted",
                guidance_scale=3.0,
                audio_guidance_scale=audio_guidance_scale,
                generator=ov_genai.CppStdGenerator(42),
                **GEN_KWARGS,
            )

        low = run(1.5)
        high = run(7.0)
        assert not np.array_equal(np.array(low.audio.data), np.array(high.audio.data))

    @pytest.mark.parametrize("video_generation_model", [LTX2_MODEL_ID], indirect=True)
    def test_taylorseer_rejected(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        with pytest.raises(RuntimeError, match="TaylorSeer"):
            pipe.generate("test prompt", taylorseer_config=ov_genai.TaylorSeerCacheConfig(), **GEN_KWARGS)

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_cpp_std_generator(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", generator=ov_genai.CppStdGenerator(42), **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_reshape(self, video_generation_model):
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        pipe.reshape(1, 9, 32, 32, 3.0)
        pipe.compile("CPU")

        result = pipe.generate("test prompt", negative_prompt="bad quality", num_inference_steps=2)
        assert result.video.shape == [1, 9, 32, 32, 3]

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_reshape_multiple_videos_with_cfg(self, video_generation_model):
        """Widest timestep shape: batch is num_videos_per_prompt * 2 when CFG is enabled."""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        pipe.reshape(2, 9, 32, 32, 3.0)
        pipe.compile("CPU")

        result = pipe.generate("test prompt", num_videos_per_prompt=2, **GEN_KWARGS)
        assert result.video.shape == [2, 9, 32, 32, 3]

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_generate_without_cfg_default_compile(self, video_generation_model):
        """Regression test: direct-compile constructor should work with guidance_scale <= 1."""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate("test prompt", guidance_scale=1.0, **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_generate_without_cfg_after_reshape_with_cfg(self, video_generation_model):
        """Test: reshape with CFG then generate without CFG should raise error."""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        pipe.reshape(1, 9, 32, 32, 3.0)
        pipe.compile("CPU")

        with pytest.raises(RuntimeError, match="guidance_scale <= 1 requested, but the compiled model expects CFG"):
            pipe.generate("test prompt", guidance_scale=1.0, **GEN_KWARGS)

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_generate_with_cfg_after_reshape_without_cfg(self, video_generation_model):
        """Regression test: reshape without CFG then generate with CFG triggers rebuild."""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model)
        pipe.reshape(1, 9, 32, 32, 1.0)
        pipe.compile("CPU")

        result = pipe.generate("test prompt", guidance_scale=3.0, **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]


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
    @pytest.fixture(autouse=True)
    def require_encoder(self, video_generation_model):
        if not Path(video_generation_model, "vae_encoder").exists():
            pytest.skip("vae_encoder not available in test model")
        if not Path(video_generation_model, "vae_decoder").exists():
            pytest.skip("vae_decoder not available in test model")

    def _make_vae(self, video_generation_model, compiled=True):
        enc = str(Path(video_generation_model) / "vae_encoder")
        dec = str(Path(video_generation_model) / "vae_decoder")
        return ov_genai.AutoencoderKLLTXVideo(enc, dec, "CPU") if compiled else ov_genai.AutoencoderKLLTXVideo(enc, dec)

    def _encoder_output_name(self, video_generation_model):
        enc_xml = str(Path(video_generation_model) / "vae_encoder" / "openvino_model.xml")
        return ov.Core().read_model(enc_xml).outputs[0].get_any_name()

    def test_constructor_with_encoder(self, video_generation_model):
        assert self._make_vae(video_generation_model, compiled=False) is not None

    def test_encode_without_compile_raises(self, video_generation_model):
        vae = self._make_vae(video_generation_model, compiled=False)
        dummy = ov.Tensor(np.zeros([1, 3, 9, 32, 32], dtype=np.float32))
        with pytest.raises(RuntimeError, match="must be compiled first"):
            vae.encode(dummy, ov_genai.CppStdGenerator(42))

    def test_encode_without_encoder_raises(self, video_generation_model):
        decoder_path = Path(video_generation_model) / "vae_decoder"
        if not decoder_path.exists():
            pytest.skip("vae_decoder not available in test model")
        vae = ov_genai.AutoencoderKLLTXVideo(str(decoder_path))
        vae.compile("CPU")
        dummy = ov.Tensor(np.zeros([1, 3, 9, 32, 32], dtype=np.float32))
        with pytest.raises(RuntimeError, match="without 'VAE encoder' capability"):
            vae.encode(dummy, ov_genai.CppStdGenerator(42))

    def test_encode_output_shape(self, video_generation_model):
        vae = self._make_vae(video_generation_model)
        config = vae.get_config()
        dummy = ov.Tensor(np.zeros([1, 3, 9, 32, 32], dtype=np.float32))
        latent = vae.encode(dummy, ov_genai.CppStdGenerator(42))
        assert latent is not None
        shape = latent.shape
        assert len(shape) == 5, f"Expected 5D latent [B, C, F, H, W], got shape {shape}"
        assert shape[0] == 1
        assert shape[1] == config.latent_channels

    def test_encode_is_deterministic(self, video_generation_model):
        vae = self._make_vae(video_generation_model)
        video = ov.Tensor(np.ones([1, 3, 9, 32, 32], dtype=np.float32) * 0.5)
        latent1 = vae.encode(video, ov_genai.CppStdGenerator(42))
        latent2 = vae.encode(video, ov_genai.CppStdGenerator(42))
        np.testing.assert_array_equal(latent1.data, latent2.data)

    def test_encode_varies_with_seed(self, video_generation_model):
        vae = self._make_vae(video_generation_model)
        output_name = self._encoder_output_name(video_generation_model)
        video = ov.Tensor(np.ones([1, 3, 9, 32, 32], dtype=np.float32) * 0.5)
        latent1 = vae.encode(video, ov_genai.CppStdGenerator(42))
        latent2 = vae.encode(video, ov_genai.CppStdGenerator(99))
        if output_name == "latent_parameters":
            assert not np.array_equal(latent1.data, latent2.data), (
                "Different generator seeds should produce different latents"
            )
        elif output_name == "latent_sample":
            np.testing.assert_array_equal(latent1.data, latent2.data)
        else:
            pytest.skip(f"Unexpected encoder output name '{output_name}'")

    def test_encode_none_generator_raises_for_stochastic_encoder(self, video_generation_model):
        output_name = self._encoder_output_name(video_generation_model)
        if output_name != "latent_parameters":
            pytest.skip("Encoder is deterministic (latent_sample) — generator=None is valid")
        vae = self._make_vae(video_generation_model)
        video = ov.Tensor(np.ones([1, 3, 9, 32, 32], dtype=np.float32) * 0.5)
        with pytest.raises(RuntimeError, match="requires a non-null generator"):
            vae.encode(video, None)


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

        pipe.generate(
            "test prompt", callback=make_callback(baseline_latents), taylorseer_config=None, **generate_kwargs
        )
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

    def test_taylorseer_default_disabled(self, video_generation_model):
        """Test that TaylorSeer is disabled by default"""
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU")
        assert pipe.get_generation_config().taylorseer_config is None

class TestLoRAVideoGeneration:
    def test_lora_adapters_constructor(self, video_generation_model):
        """Test that LoRA adapters can be passed to the constructor without error"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU", adapters=adapter_config)
        assert pipe is not None

    def test_lora_adapters_generate(self, video_generation_model):
        """Test that LoRA adapters can be passed to generate() without error"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU", adapters=adapter_config)

        result = pipe.generate(
            "test prompt", height=32, width=32, num_frames=9, num_inference_steps=2, adapters=adapter_config
        )
        assert result is not None
        assert result.video is not None

    def test_lora_adapters_default_from_constructor(self, video_generation_model):
        """Test that LoRA adapters passed to the constructor are used by default in generate()"""
        adapter_config = ov_genai.AdapterConfig()
        pipe = ov_genai.Text2VideoPipeline(video_generation_model, "CPU", adapters=adapter_config)
        result = pipe.generate(
            "test prompt",
            height=32,
            width=32,
            num_frames=9,
            num_inference_steps=2,
        )
        assert result is not None
        assert result.video is not None

    def test_transformer_has_set_adapters_method(self, video_generation_model):
        """Test that the LTXVideoTransformer3DModel has the set_adapters method"""
        model_path = Path(video_generation_model) / "transformer"
        assert model_path.exists(), f"Transformer subdirectory not found at: {model_path}"
        model = ov_genai.LTXVideoTransformer3DModel(str(model_path))
        model.compile("CPU")

        assert hasattr(model, "set_adapters")

        model.set_adapters(None)


class TestImage2VideoPipeline:
    def _make_image(self, height=32, width=32):
        # Structured content: a uniform image makes conditioning-related assertions vacuous.
        image_data = np.random.randint(0, 255, (1, height, width, 3), dtype=np.uint8)
        return ov.Tensor(image_data)

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_constructor_without_encoder_raises(self, video_generation_model, tmp_path):
        import shutil

        no_encoder_dir = tmp_path / "no_encoder_model"
        shutil.copytree(video_generation_model, no_encoder_dir, ignore=shutil.ignore_patterns("vae_encoder"))
        with pytest.raises(RuntimeError, match="vae_encoder"):
            ov_genai.Image2VideoPipeline(str(no_encoder_dir))

    @pytest.mark.parametrize(
        ("video_generation_model", "audio_sample_rate"),
        [(LTX_VIDEO_MODEL_ID, 0), (LTX2_MODEL_ID, 24000)],
        indirect=["video_generation_model"],
    )
    def test_generate_runs(self, video_generation_model, audio_sample_rate):
        pipe = ov_genai.Image2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate(self._make_image(), "test prompt", **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]
        assert result.audio_sample_rate == audio_sample_rate
        if audio_sample_rate:
            assert list(result.audio.shape)[0] == 1
        else:
            assert result.audio.size == 0

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_reshape(self, video_generation_model):
        pipe = ov_genai.Image2VideoPipeline(video_generation_model)
        pipe.reshape(1, 9, 32, 32, 3.0)
        pipe.compile("CPU")

        result = pipe.generate(self._make_image(), "test prompt", **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_reshape_updates_generation_config(self, video_generation_model):
        pipe = ov_genai.Image2VideoPipeline(video_generation_model)
        pipe.reshape(1, 9, 32, 32, 3.0)

        config = pipe.get_generation_config()
        assert config.num_videos_per_prompt == 1
        assert config.num_frames == 9
        assert config.height == 32
        assert config.width == 32
        assert config.guidance_scale == 3.0

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_reshape_multiple_videos_per_prompt(self, video_generation_model):
        pipe = ov_genai.Image2VideoPipeline(video_generation_model)
        pipe.reshape(2, 9, 32, 32, 3.0)
        pipe.compile("CPU")

        result = pipe.generate(self._make_image(), "test prompt", num_videos_per_prompt=2, **GEN_KWARGS)
        assert result.video.shape == [2, 9, 32, 32, 3]
        if result.audio_sample_rate:
            assert list(result.audio.shape)[0] == 2

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_determinism(self, video_generation_model):
        pipe = ov_genai.Image2VideoPipeline(video_generation_model, "CPU")
        image = self._make_image()

        def run():
            result = pipe.generate(image, "test prompt", generator=ov_genai.CppStdGenerator(42), **GEN_KWARGS)
            return np.array(result.video.data), np.array(result.audio.data)

        first_video, first_audio = run()
        second_video, second_audio = run()
        assert np.array_equal(first_video, second_video)
        assert np.array_equal(first_audio, second_audio)

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID, LTX2_MODEL_ID], indirect=True)
    def test_image_conditions_output(self, video_generation_model):
        pipe = ov_genai.Image2VideoPipeline(video_generation_model, "CPU")

        def run(image):
            result = pipe.generate(image, "test prompt", generator=ov_genai.CppStdGenerator(42), **GEN_KWARGS)
            return np.array(result.video.data)

        assert not np.array_equal(run(self._make_image()), run(self._make_image()))

    @pytest.mark.parametrize("video_generation_model", [LTX_VIDEO_MODEL_ID], indirect=True)
    def test_lora_passthrough(self, video_generation_model):
        pipe = ov_genai.Image2VideoPipeline(video_generation_model, "CPU")
        result = pipe.generate(self._make_image(), "test prompt", adapters=ov_genai.AdapterConfig(), **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]

    @pytest.mark.parametrize("video_generation_model", [LTX2_MODEL_ID], indirect=True)
    def test_lora_rejected(self, video_generation_model):
        pipe = ov_genai.Image2VideoPipeline(video_generation_model, "CPU")
        with pytest.raises(RuntimeError, match="LoRA"):
            pipe.generate(self._make_image(), "test prompt", adapters=ov_genai.AdapterConfig(), **GEN_KWARGS)
