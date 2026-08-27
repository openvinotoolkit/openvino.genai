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

MODEL_ID = "tiny-random-ltx2"
MODEL_NAME = "optimum-intel-internal-testing/tiny-random-ltx2"

GEN_KWARGS = dict(height=32, width=32, num_frames=9, num_inference_steps=2)


@pytest.fixture(scope="module")
def ltx2_model() -> str:
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


class TestLTX2PipelineConstructor:
    def test_constructor_path_only(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model)
        assert pipe is not None

    def test_constructor_with_device(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")
        assert pipe is not None


class TestLTX2PipelineConfig:
    def test_default_config(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model)
        config = pipe.get_generation_config()
        assert config.guidance_scale == pytest.approx(4.0)
        assert config.height == 512
        assert config.width == 768
        assert config.num_frames == 121
        assert config.num_inference_steps == 40
        assert config.max_sequence_length == 1024
        assert config.audio_guidance_scale is None

    def test_audio_guidance_scale_roundtrip(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model)
        config = pipe.get_generation_config()
        config.audio_guidance_scale = 7.0
        pipe.set_generation_config(config)
        assert pipe.get_generation_config().audio_guidance_scale == pytest.approx(7.0)

    def test_audio_sample_rate(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model)
        assert pipe.get_audio_sample_rate() == 24000


class TestLTX2PipelineGenerate:
    def test_generate_basic(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")
        result = pipe.generate("test prompt", guidance_scale=1.0, **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]
        assert result.video.element_type.to_dtype() == np.uint8
        audio_shape = list(result.audio.shape)
        assert len(audio_shape) == 3
        assert audio_shape[0] == 1
        assert audio_shape[1] == 2
        assert audio_shape[2] > 0

    def test_generate_with_negative_prompt(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")
        result = pipe.generate(
            "test prompt",
            negative_prompt="bad quality",
            guidance_scale=3.0,
            **GEN_KWARGS,
        )
        assert result.video.shape == [1, 9, 32, 32, 3]
        assert list(result.audio.shape)[1] == 2

    def test_audio_guidance_scale_affects_audio(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")

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
        assert np.array_equal(np.array(low.video.data), np.array(high.video.data))
        assert not np.array_equal(np.array(low.audio.data), np.array(high.audio.data))

    def test_generate_deterministic_with_seed(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")

        def run():
            return pipe.generate(
                "test prompt",
                generator=ov_genai.CppStdGenerator(42),
                guidance_scale=1.0,
                **GEN_KWARGS,
            )

        first = run()
        second = run()
        assert np.array_equal(np.array(first.video.data), np.array(second.video.data))
        assert np.array_equal(np.array(first.audio.data), np.array(second.audio.data))

    def test_generate_with_callback(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")
        steps_seen = []

        def callback(step, num_steps, latent):
            steps_seen.append((step, num_steps))
            return False

        result = pipe.generate("test prompt", guidance_scale=1.0, callback=callback, **GEN_KWARGS)
        assert result.video.shape == [1, 9, 32, 32, 3]
        assert len(steps_seen) == 2

    def test_callback_early_stop(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")

        def callback(step, num_steps, latent):
            return True

        result = pipe.generate("test prompt", guidance_scale=1.0, callback=callback, **GEN_KWARGS)
        assert len(list(result.video.shape)) == 0

    def test_num_videos_per_prompt(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")
        result = pipe.generate("test prompt", guidance_scale=1.0, num_videos_per_prompt=2, **GEN_KWARGS)
        assert result.video.shape == [2, 9, 32, 32, 3]
        assert list(result.audio.shape)[0] == 2

    def test_taylorseer_rejected(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model, "CPU")
        with pytest.raises(RuntimeError, match="TaylorSeer"):
            pipe.generate(
                "test prompt",
                guidance_scale=1.0,
                taylorseer_config=ov_genai.TaylorSeerCacheConfig(),
                **GEN_KWARGS,
            )


class TestLTX2PipelineReshape:
    def test_reshape_compile_generate(self, ltx2_model):
        pipe = ov_genai.Text2VideoPipeline(ltx2_model)
        pipe.reshape(1, 9, 32, 32, 3.0)
        pipe.compile("CPU")
        result = pipe.generate("test prompt", negative_prompt="bad quality", guidance_scale=3.0, num_inference_steps=2)
        assert result.video.shape == [1, 9, 32, 32, 3]
        assert list(result.audio.shape)[1] == 2


class TestLTX2Image2VideoRejected:
    def test_image2video_rejects_ltx2(self, ltx2_model):
        with pytest.raises(RuntimeError, match="LTX2Pipeline"):
            ov_genai.Image2VideoPipeline(ltx2_model)
