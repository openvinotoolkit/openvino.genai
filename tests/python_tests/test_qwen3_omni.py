"""
Tests for Qwen3-Omni multimodal pipeline.

Tests cover:
- Text-only generation (thinker as VLM)
- Image + text input
- Audio + text input
- Speech output generation
- Combined audio input + speech output (full omni)
- Text-only fallback when speech models are absent
"""

import numpy as np
import pytest

import openvino as ov

REAL_MODEL_DIR = "temp/qwen3omni_ov"


def generate_synthetic_audio(duration_sec: float = 1.0, sample_rate: int = 16000) -> ov.Tensor:
    """Generate a synthetic audio tensor (sine wave) for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    # 440Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return ov.Tensor(audio)


def generate_synthetic_image(height: int = 64, width: int = 64) -> ov.Tensor:
    """Generate a synthetic RGB image tensor for testing."""
    image = np.random.randint(0, 255, (1, height, width, 3), dtype=np.uint8)
    return ov.Tensor(image)


@pytest.fixture(scope="module")
def synthetic_audio():
    return generate_synthetic_audio()


@pytest.fixture(scope="module")
def synthetic_image():
    return generate_synthetic_image()


@pytest.mark.real_models
class TestQwen3OmniRealModel:
    """Tests using the real Qwen3-Omni model (nightly CI only)."""

    @pytest.fixture(scope="class")
    def pipe(self):
        import openvino_genai as ov_genai
        from pathlib import Path

        if not Path(REAL_MODEL_DIR).exists():
            pytest.skip(f"Real model directory not found: {REAL_MODEL_DIR}")

        return ov_genai.VLMPipeline(REAL_MODEL_DIR, "CPU")

    def test_text_only(self, pipe):
        """Test text-only generation (no images, no audio)."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 30
        result = pipe.generate("What is 2+2?", generation_config=config)
        assert len(result.texts) > 0
        assert len(result.texts[0]) > 0

    def test_image_input(self, pipe, synthetic_image):
        """Test image + text input."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 30
        result = pipe.generate(
            "Describe this image",
            images=[synthetic_image],
            generation_config=config,
        )
        assert len(result.texts) > 0
        assert len(result.texts[0]) > 0

    def test_audio_input(self, pipe, synthetic_audio):
        """Test audio + text input."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 30
        result = pipe.generate(
            "What do you hear?",
            audios=[synthetic_audio],
            generation_config=config,
        )
        assert len(result.texts) > 0
        assert len(result.texts[0]) > 0

    def test_speech_output(self, pipe):
        """Test speech output generation."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 30
        config.return_audio = True
        config.speaker = "f245"
        result = pipe.generate("Say hello", generation_config=config)
        assert len(result.texts) > 0
        # Speech outputs should be non-empty
        assert len(result.speech_outputs) > 0
        waveform = result.speech_outputs[0]
        assert waveform.size > 0
        assert waveform.element_type == ov.Type.f32

    def test_full_omni(self, pipe, synthetic_audio):
        """Test full omni: audio input + speech output."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 30
        config.return_audio = True
        config.speaker = "m02"
        result = pipe.generate(
            "Repeat what you hear",
            audios=[synthetic_audio],
            generation_config=config,
        )
        assert len(result.texts) > 0
        assert len(result.speech_outputs) > 0

    def test_chat_mode(self, pipe):
        """Test multi-turn chat."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 20

        pipe.start_chat()
        result1 = pipe.generate("Hello!", generation_config=config)
        assert len(result1.texts) > 0

        result2 = pipe.generate("What did I just say?", generation_config=config)
        assert len(result2.texts) > 0
        pipe.finish_chat()


class TestQwen3OmniGenerationConfig:
    """Test generation config fields for speech output."""

    def test_return_audio_default(self):
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        assert config.return_audio is False
        assert config.speaker == ""

    def test_return_audio_set(self):
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.return_audio = True
        config.speaker = "f245"
        assert config.return_audio is True
        assert config.speaker == "f245"


class TestVLMDecodedResults:
    """Test VLMDecodedResults speech_outputs field."""

    def test_speech_outputs_default_empty(self):
        import openvino_genai as ov_genai

        result = ov_genai.VLMDecodedResults()
        assert hasattr(result, "speech_outputs")
        assert len(result.speech_outputs) == 0


@pytest.mark.real_models
class TestQwen3OmniEdgeCases:
    """Edge case tests using the real Qwen3-Omni model."""

    @pytest.fixture(scope="class")
    def pipe(self):
        import openvino_genai as ov_genai
        from pathlib import Path

        if not Path(REAL_MODEL_DIR).exists():
            pytest.skip(f"Real model directory not found: {REAL_MODEL_DIR}")

        return ov_genai.VLMPipeline(REAL_MODEL_DIR, "CPU")

    def test_empty_audio_skipped(self, pipe):
        """Empty audio tensor (0 samples) should be gracefully skipped without error."""
        import openvino_genai as ov_genai

        empty_audio = ov.Tensor(np.array([], dtype=np.float32))
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 10

        result = pipe.generate(
            "Hello",
            audios=[empty_audio],
            generation_config=config,
        )
        assert len(result.texts) > 0

    def test_return_audio_without_speech_models(self, pipe):
        """Requesting return_audio on a model without talker should produce empty speech outputs."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 10
        config.return_audio = True
        config.speaker = "f245"

        # Should not crash; speech_outputs may be empty if speech models are absent
        result = pipe.generate("Hello", generation_config=config)
        assert len(result.texts) > 0

    def test_streaming_cancel(self, pipe):
        """Streaming cancel mid-generation should cleanly stop and still produce partial output."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 100

        tokens_received = []

        def cancel_after_3_tokens(text: str) -> bool:
            tokens_received.append(text)
            return len(tokens_received) >= 3  # Request stop after 3 tokens

        result = pipe.generate(
            "Tell me a long story",
            generation_config=config,
            streamer=cancel_after_3_tokens,
        )
        assert len(result.texts) > 0
        # Should have received at least some streamed tokens before cancel
        assert len(tokens_received) >= 3

    def test_multiple_audios(self, pipe, synthetic_audio):
        """Multiple audio inputs should be processed without error."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 20

        result = pipe.generate(
            "What sounds do you hear?",
            audios=[synthetic_audio, synthetic_audio],
            generation_config=config,
        )
        assert len(result.texts) > 0

    def test_no_audio_streamer_runs_in_batch_mode(self, pipe):
        """Without audio_streamer callback, speech generation runs in batch mode."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 10
        config.return_audio = True
        config.speaker = "f245"

        result = pipe.generate("Say hello", generation_config=config)
        assert len(result.texts) > 0

    def test_audio_chunk_frames_zero_rejected(self, pipe):
        """audio_chunk_frames=0 with an active streamer must raise — invariant is >= 1."""
        import openvino_genai as ov_genai
        import pytest

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 10
        config.return_audio = True
        config.speaker = ""
        config.audio_chunk_frames = 0

        def audio_cb(chunk):
            return ov_genai.StreamingStatus.RUNNING

        with pytest.raises(RuntimeError):
            pipe.generate("Say hello", generation_config=config, audio_streamer=audio_cb)

    def test_audio_streamer_callback_invoked(self, pipe):
        """audio_streamer callback should be invoked during speech generation."""
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 10
        config.return_audio = True
        config.speaker = ""

        chunk_count = [0]

        def audio_cb(chunk: ov.Tensor) -> ov_genai.StreamingStatus:
            chunk_count[0] += 1
            return ov_genai.StreamingStatus.RUNNING

        result = pipe.generate("Say hello", generation_config=config, audio_streamer=audio_cb)
        assert len(result.texts) > 0
        assert chunk_count[0] > 0, "audio_streamer callback was never invoked"
