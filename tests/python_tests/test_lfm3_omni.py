# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for LFM3-Omni in VLMPipeline and OmniPipeline.

LFM3-Omni has no separate talker: its language model emits text tokens and speech frames in one
stream, and OmniPipeline decodes the speech inside the VLM (SDPA backend). The checks run on
`optimum-intel-internal-testing/tiny-random-lfm3-omni` and compare with optimum-intel on the
same export. The tiny checkpoint sets no audio sampling, so each code is the argmax, but the
MaskGIT codebook order is drawn from a random generator that differs between torch and GenAI.
Speech is therefore checked for structure (frame counts, the text slots around it, streaming,
seeds) rather than for equal codes.

Exporting the checkpoint needs transformers >= 5.20.2 and an optimum-intel with the LFM3-Omni
export (CVS-197646). `lfm3_omni_model_path` skips with that reason when the installed versions
cannot export it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_tokenizers
import pytest
import transformers
from huggingface_hub import snapshot_download
from optimum.utils.import_utils import is_transformers_version
from PIL import Image
from transformers.video_utils import VideoMetadata as HFVideoMetadata

import openvino_genai as ov_genai
from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir
from utils.network import retry_request

pytestmark = pytest.mark.lfm3_omni

MODEL_ID = "optimum-intel-internal-testing/tiny-random-lfm3-omni"
EXPORT_SKIP_REASON = (
    "CVS-197646: exporting LFM3-Omni needs transformers >= 5.20.2 and an optimum-intel with the lfm3_omni export"
)
PROMPT = "Describe what you get."
TEXT_TOKENS = 8
INTERLEAVED_TEXT_SLOTS = 6
INTERLEAVED_AUDIO_FRAMES = 12
VIDEO_FPS = 2.0


def _lfm3_omni_export_supported() -> bool:
    if not is_transformers_version(">=", "5.20.2"):
        return False
    import optimum.exporters.openvino  # noqa: F401  registers the OpenVINO export configs
    from optimum.exporters.tasks import TasksManager

    return "lfm3_omni" in TasksManager._SUPPORTED_MODEL_TYPE


def _export(target_dir: Path) -> None:
    from optimum.intel import OVModelForMultimodalLM

    model_cached = retry_request(lambda: snapshot_download(MODEL_ID))
    processor = transformers.AutoProcessor.from_pretrained(model_cached, trust_remote_code=True)
    model = OVModelForMultimodalLM.from_pretrained(
        model_cached, export=True, trust_remote_code=True, compile=False, load_in_8bit=False
    )
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(processor.tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, target_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, target_dir / "openvino_detokenizer.xml")
    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


@pytest.fixture(scope="module")
def lfm3_omni_model_path() -> Path:
    if not _lfm3_omni_export_supported():
        pytest.skip(EXPORT_SKIP_REASON)
    model_dir = get_ov_cache_converted_models_dir() / MODEL_ID.replace("/", "_")
    manager = AtomicDownloadManager(model_dir)
    if not manager.is_complete() and not (model_dir / "openvino_language_model.xml").exists():
        manager.execute(_export)
    return model_dir


@pytest.fixture(scope="module")
def frame_samples(lfm3_omni_model_path: Path) -> int:
    codec = json.loads((lfm3_omni_model_path / "audio_tokenizer" / "config.json").read_text())
    return codec["stft_frame_stack"] * codec["stft_hop_length"]


@pytest.fixture(scope="module")
def omni_pipe(lfm3_omni_model_path: Path) -> ov_genai.OmniPipeline:
    return ov_genai.OmniPipeline(lfm3_omni_model_path, "CPU")


@pytest.fixture(scope="module")
def omni_chat_pipe(lfm3_omni_model_path: Path) -> ov_genai.OmniPipeline:
    # A ChatHistory call keeps the pipeline in chat mode, so prompt calls on it would continue that chat.
    return ov_genai.OmniPipeline(lfm3_omni_model_path, "CPU")


@pytest.fixture(scope="module")
def omni_pa_pipe(lfm3_omni_model_path: Path) -> ov_genai.OmniPipeline:
    return ov_genai.OmniPipeline(lfm3_omni_model_path, "CPU", ATTENTION_BACKEND="PA")


@pytest.fixture(scope="module", params=["SDPA", "PA"])
def vlm_pipe(request: pytest.FixtureRequest, lfm3_omni_model_path: Path) -> ov_genai.VLMPipeline:
    return ov_genai.VLMPipeline(lfm3_omni_model_path, "CPU", ATTENTION_BACKEND=request.param)


@dataclass(frozen=True)
class OptimumReference:
    model: object  # OVModelForMultimodalLM; imported lazily since the pinned optimum-intel may lack it
    processor: transformers.ProcessorMixin


@pytest.fixture(scope="module")
def optimum_reference(lfm3_omni_model_path: Path) -> OptimumReference:
    from optimum.intel import OVModelForMultimodalLM

    return OptimumReference(
        model=OVModelForMultimodalLM.from_pretrained(lfm3_omni_model_path, device="CPU", trust_remote_code=True),
        processor=transformers.AutoProcessor.from_pretrained(lfm3_omni_model_path, trust_remote_code=True),
    )


@dataclass(frozen=True)
class Media:
    image: np.ndarray  # [H, W, 3] uint8
    audio: np.ndarray  # [samples] f32, 16 kHz
    video: np.ndarray  # [frames, H, W, 3] uint8


@pytest.fixture(scope="module")
def media() -> Media:
    rng = np.random.default_rng(0)
    seconds = np.arange(16000) / 16000
    return Media(
        image=rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
        audio=(0.5 * np.sin(2 * np.pi * 440 * seconds)).astype(np.float32),
        video=rng.integers(0, 256, (8, 64, 96, 3), dtype=np.uint8),
    )


def _genai_inputs(case: str, media: Media) -> dict[str, list]:
    inputs: dict[str, list] = {}
    if "image" in case:
        inputs["images"] = [ov.Tensor(media.image[None])]
    if "audio" in case:
        inputs["audios"] = [ov.Tensor(media.audio)]
    if "video" in case:
        metadata = ov_genai.VideoMetadata()
        metadata.fps = VIDEO_FPS
        inputs["videos"] = [ov.Tensor(media.video)]
        inputs["videos_metadata"] = [metadata]
    return inputs


def _optimum_inputs(reference: OptimumReference, case: str, media: Media) -> dict:
    kwargs: dict = {}
    if "image" in case:
        kwargs["image"] = Image.fromarray(media.image)
    if "audio" in case:
        kwargs["audio"] = media.audio
    if "video" in case:
        frames = media.video.shape[0]
        metadata = HFVideoMetadata(total_num_frames=frames, fps=VIDEO_FPS, frames_indices=list(range(frames)))
        kwargs["video"] = (media.video, metadata)
    return reference.model.preprocess_inputs(text=PROMPT, processor=reference.processor, **kwargs)


class TokenCollector(ov_genai.StreamerBase):
    def __init__(self, status: ov_genai.StreamingStatus = ov_genai.StreamingStatus.RUNNING) -> None:
        super().__init__()
        self.tokens: list[int] = []
        self.status = status

    def write(self, token: int | list[int]) -> ov_genai.StreamingStatus:
        self.tokens.extend(token if isinstance(token, list) else [token])
        return self.status

    def end(self) -> None:
        pass


def _text_config(max_new_tokens: int) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    return config


def _speech_config(max_frames: int, **fields) -> ov_genai.OmniTalkerSpeechConfig:
    config = ov_genai.OmniTalkerSpeechConfig()
    config.max_new_tokens = max_frames
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def _waveform(result: ov_genai.OmniDecodedResults) -> np.ndarray:
    assert len(result.speech_result.waveforms) == 1
    waveform = np.array(result.speech_result.waveforms[0].data, dtype=np.float32)
    assert waveform.shape[:2] == (1, 1)
    assert np.isfinite(waveform).all()
    return waveform.reshape(-1)


class TestSpeechConfigFields:
    def test_defaults(self) -> None:
        config = ov_genai.OmniTalkerSpeechConfig()
        assert config.output_mode == ov_genai.SpeechOutputMode.INTERLEAVED
        assert config.cp_repetition_penalty is None

    @pytest.mark.parametrize(
        "field, value",
        [("output_mode", ov_genai.SpeechOutputMode.SPEECH_ONLY), ("cp_repetition_penalty", 1.3)],
    )
    def test_round_trip(self, field: str, value: object) -> None:
        config = ov_genai.OmniTalkerSpeechConfig()
        setattr(config, field, value)
        expected = pytest.approx(value) if isinstance(value, float) else value
        assert getattr(config, field) == expected


@pytest.mark.parametrize("case", ["text", "image", "video", "audio", "image_audio", "image_video"])
def test_text_matches_optimum(
    vlm_pipe: ov_genai.VLMPipeline, optimum_reference: OptimumReference, media: Media, case: str
) -> None:
    inputs = _optimum_inputs(optimum_reference, case, media)
    expected = optimum_reference.model.generate(
        **inputs, max_new_tokens=TEXT_TOKENS, do_sample=False, generation_mode="text"
    )[0, inputs["input_ids"].shape[1] :].tolist()
    collector = TokenCollector()
    vlm_pipe.generate(
        PROMPT, generation_config=_text_config(TEXT_TOKENS), streamer=collector, **_genai_inputs(case, media)
    )
    assert collector.tokens == expected


def test_tagged_audio_matches_optimum(
    vlm_pipe: ov_genai.VLMPipeline, optimum_reference: OptimumReference, media: Media
) -> None:
    processor = optimum_reference.processor
    content = [{"type": "text", "text": "Listen: "}, {"type": "audio"}, {"type": "text", "text": " What is it?"}]
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    inputs = processor(text=[prompt], audio=[media.audio], sampling_rate=16000, return_tensors="pt")
    expected = optimum_reference.model.generate(
        **inputs, max_new_tokens=TEXT_TOKENS, do_sample=False, generation_mode="text"
    )[0, inputs["input_ids"].shape[1] :].tolist()
    collector = TokenCollector()
    vlm_pipe.generate(
        "Listen: <ov_genai_audio_0> What is it?",
        audios=[ov.Tensor(media.audio)],
        generation_config=_text_config(TEXT_TOKENS),
        streamer=collector,
    )
    assert collector.tokens == expected


def test_chat_history_matches_optimum(
    vlm_pipe: ov_genai.VLMPipeline, optimum_reference: OptimumReference, media: Media
) -> None:
    # An image turn, a fixed assistant reply and an audio turn: media from both user turns in one prompt.
    answer = "It is noise."
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "And this?"}]},
    ]
    processor = optimum_reference.processor
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=[Image.fromarray(media.image)],
        audio=[media.audio],
        sampling_rate=16000,
        return_tensors="pt",
    )
    expected = optimum_reference.model.generate(
        **inputs, max_new_tokens=TEXT_TOKENS, do_sample=False, generation_mode="text"
    )[0, inputs["input_ids"].shape[1] :].tolist()

    # Media indices are conversation-absolute: each call passes the media of its newest user message.
    history = ov_genai.ChatHistory()
    history.append({"role": "user", "content": "<ov_genai_image_0>What is this?"})
    vlm_pipe.generate(history, images=[ov.Tensor(media.image[None])], generation_config=_text_config(1))
    history.append({"role": "assistant", "content": answer})
    history.append({"role": "user", "content": "<ov_genai_audio_0>And this?"})
    collector = TokenCollector()
    vlm_pipe.generate(
        history, audios=[ov.Tensor(media.audio)], generation_config=_text_config(TEXT_TOKENS), streamer=collector
    )
    assert collector.tokens == expected


@pytest.mark.parametrize("max_frames", [1, 4, INTERLEAVED_AUDIO_FRAMES + 1])
def test_interleaved_matches_optimum_layout(
    omni_pipe: ov_genai.OmniPipeline, optimum_reference: OptimumReference, frame_samples: int, max_frames: int
) -> None:
    # The first text slots come before any speech frame, so they are greedy and match exactly.
    inputs = optimum_reference.model.preprocess_inputs(text=PROMPT, processor=optimum_reference.processor)
    slots, _ = optimum_reference.model.generate(
        **inputs, max_new_tokens=INTERLEAVED_TEXT_SLOTS + max_frames, do_sample=False, return_audio=True
    )
    expected_text = slots[0, inputs["input_ids"].shape[1] + 1 :][:INTERLEAVED_TEXT_SLOTS].tolist()
    collector = TokenCollector()
    result = omni_pipe.generate(
        PROMPT,
        text_config=_text_config(1000),
        talker_speech_config=_speech_config(max_frames),
        streamer=collector,
    )
    text_blocks = 1 + (max_frames - 1) // INTERLEAVED_AUDIO_FRAMES
    assert len(collector.tokens) == text_blocks * INTERLEAVED_TEXT_SLOTS
    assert collector.tokens[:INTERLEAVED_TEXT_SLOTS] == expected_text
    assert _waveform(result).size == max_frames * frame_samples


@pytest.mark.parametrize("chunk_frames", [1, 3])
def test_streamed_chunks_form_the_waveform(
    omni_pipe: ov_genai.OmniPipeline, frame_samples: int, chunk_frames: int
) -> None:
    chunks: list[np.ndarray] = []

    def on_chunk(chunk: ov.Tensor) -> ov_genai.StreamingStatus:
        chunks.append(np.array(chunk.data, dtype=np.float32).reshape(-1))
        return ov_genai.StreamingStatus.RUNNING

    result = omni_pipe.generate(
        PROMPT,
        text_config=_text_config(1000),
        talker_speech_config=_speech_config(7, audio_chunk_frames=chunk_frames),
        speech_streamer=on_chunk,
    )
    # One chunk per full group of frames, then the flush of the held-back ISTFT overlap.
    assert len(chunks) == 7 // chunk_frames + 1
    np.testing.assert_array_equal(np.concatenate(chunks), _waveform(result))
    assert _waveform(result).size == 7 * frame_samples


def test_streaming_matches_one_shot_decode(omni_pipe: ov_genai.OmniPipeline) -> None:
    # More frames than the 24 frames of left context each chunk is decoded with, so later chunks drop the start.
    max_frames = 30

    def speech(chunk_frames: int) -> np.ndarray:
        config = _speech_config(max_frames, audio_chunk_frames=chunk_frames, rng_seed=3)
        return _waveform(omni_pipe.generate(PROMPT, text_config=_text_config(1000), talker_speech_config=config))

    np.testing.assert_allclose(speech(2), speech(max_frames), atol=1e-5)


@pytest.mark.parametrize("field, value", [("cp_temperature", 0.5), ("cp_top_k", 4), ("cp_repetition_penalty", 2.0)])
def test_speech_sampling_fields_change_speech(omni_pipe: ov_genai.OmniPipeline, field: str, value: float) -> None:
    def speech(**fields) -> np.ndarray:
        config = _speech_config(8, rng_seed=0, **fields)
        return _waveform(omni_pipe.generate(PROMPT, text_config=_text_config(1000), talker_speech_config=config))

    assert not np.array_equal(speech(**{field: value}), speech())


def test_max_new_tokens_caps_text_and_speech_slots(omni_pipe: ov_genai.OmniPipeline, frame_samples: int) -> None:
    collector = TokenCollector()
    result = omni_pipe.generate(
        PROMPT,
        text_config=_text_config(INTERLEAVED_TEXT_SLOTS + 3),
        talker_speech_config=_speech_config(100),
        streamer=collector,
    )
    assert len(collector.tokens) == INTERLEAVED_TEXT_SLOTS
    assert _waveform(result).size == 3 * frame_samples


@pytest.mark.parametrize("status", [ov_genai.StreamingStatus.STOP, ov_genai.StreamingStatus.CANCEL])
def test_text_streamer_ends_speech_generation(
    omni_pipe: ov_genai.OmniPipeline, status: ov_genai.StreamingStatus
) -> None:
    # The interleaved answer opens with text, so no frame is generated before the streamer ends the turn.
    collector = TokenCollector(status)
    result = omni_pipe.generate(
        PROMPT, text_config=_text_config(1000), talker_speech_config=_speech_config(8), streamer=collector
    )
    assert len(collector.tokens) == 1
    assert result.speech_result.waveforms == []


@pytest.mark.parametrize(
    "status, expected_chunks", [(ov_genai.StreamingStatus.STOP, 2), (ov_genai.StreamingStatus.CANCEL, 1)]
)
def test_speech_streamer_ends_speech_generation(
    omni_pipe: ov_genai.OmniPipeline, frame_samples: int, status: ov_genai.StreamingStatus, expected_chunks: int
) -> None:
    chunks: list[np.ndarray] = []

    def on_chunk(chunk: ov.Tensor) -> ov_genai.StreamingStatus:
        chunks.append(np.array(chunk.data, dtype=np.float32).reshape(-1))
        return status

    result = omni_pipe.generate(
        PROMPT,
        text_config=_text_config(1000),
        talker_speech_config=_speech_config(10, audio_chunk_frames=2),
        speech_streamer=on_chunk,
    )
    # STOP keeps the two frames and flushes the held-back ISTFT overlap; CANCEL drops it.
    assert len(chunks) == expected_chunks
    waveform = _waveform(result)
    np.testing.assert_array_equal(np.concatenate(chunks), waveform)
    if status == ov_genai.StreamingStatus.STOP:
        assert waveform.size == 2 * frame_samples
    else:
        assert waveform.size < 2 * frame_samples


def test_seed_controls_speech(omni_pipe: ov_genai.OmniPipeline) -> None:
    def speech(seed: int) -> np.ndarray:
        config = _speech_config(4, rng_seed=seed)
        return _waveform(omni_pipe.generate(PROMPT, text_config=_text_config(1000), talker_speech_config=config))

    np.testing.assert_array_equal(speech(0), speech(0))
    assert not np.array_equal(speech(0), speech(1))


def test_speech_only_mode(omni_pipe: ov_genai.OmniPipeline, frame_samples: int) -> None:
    collector = TokenCollector()
    result = omni_pipe.generate(
        PROMPT,
        text_config=_text_config(1000),
        talker_speech_config=_speech_config(5, output_mode=ov_genai.SpeechOutputMode.SPEECH_ONLY),
        streamer=collector,
    )
    # Speech starts at once: no text precedes the first frame, and the frame cap ends the turn.
    assert collector.tokens == []
    assert _waveform(result).size == 5 * frame_samples


def test_chat_history_speech(omni_chat_pipe: ov_genai.OmniPipeline, frame_samples: int) -> None:
    # The usage from the LFM3-Omni Notes in the docs: the voice is set in the system prompt.
    history = ov_genai.ChatHistory()
    history.append({"role": "system", "content": "Perform TTS in American English using a female voice."})
    history.append({"role": "user", "content": "OpenVINO GenAI runs LFM3-Omni."})
    config = _speech_config(4, output_mode=ov_genai.SpeechOutputMode.SPEECH_ONLY)
    result = omni_chat_pipe.generate(history, text_config=_text_config(1000), talker_speech_config=config)
    assert _waveform(result).size == 4 * frame_samples


def test_chat_turn_after_speech_matches_text_turn(omni_chat_pipe: ov_genai.OmniPipeline) -> None:
    # The conv layers make every ChatHistory call decode the full history, so state left by speech must not leak.
    history = ov_genai.ChatHistory()
    history.append({"role": "user", "content": PROMPT})
    answer = omni_chat_pipe.generate(history, text_config=_text_config(1000), talker_speech_config=_speech_config(2))
    history.append({"role": "assistant", "content": answer.texts[0]})
    history.append({"role": "user", "content": "And then?"})
    text_only = _speech_config(1, return_audio=False)

    def next_turn() -> list[int]:
        collector = TokenCollector()
        omni_chat_pipe.generate(
            history, text_config=_text_config(TEXT_TOKENS), talker_speech_config=text_only, streamer=collector
        )
        return collector.tokens

    after_speech = next_turn()
    assert len(after_speech) == TEXT_TOKENS
    assert after_speech == next_turn()


def test_empty_audio_keeps_its_index(vlm_pipe: ov_genai.VLMPipeline, media: Media) -> None:
    # Both prompts expand to the same tags: the empty audio first, then the real one.
    def tokens(prompt: str, audios: list[ov.Tensor]) -> list[int]:
        history = ov_genai.ChatHistory()
        history.append({"role": "user", "content": prompt})
        collector = TokenCollector()
        vlm_pipe.generate(history, audios=audios, generation_config=_text_config(TEXT_TOKENS), streamer=collector)
        return collector.tokens

    empty = ov.Tensor(np.zeros(0, dtype=np.float32))
    audio = ov.Tensor(media.audio)
    first = tokens("<ov_genai_audio_0><ov_genai_audio_1> What is it?", [empty, audio])
    assert len(first) == TEXT_TOKENS
    assert first == tokens("<ov_genai_audio_1><ov_genai_audio_0> What is it?", [audio, empty])


def test_speech_only_first_frame_follows_the_prompt(omni_pipe: ov_genai.OmniPipeline) -> None:
    # The first frame comes from the prefill hidden state, which must be the one of the last prompt position.
    def first_frame(prompt: str) -> np.ndarray:
        config = _speech_config(1, output_mode=ov_genai.SpeechOutputMode.SPEECH_ONLY)
        return _waveform(omni_pipe.generate(prompt, text_config=_text_config(1000), talker_speech_config=config))

    assert not np.array_equal(first_frame(PROMPT), first_frame("Say something else."))


def test_top_k_zero_disables_top_k(omni_pipe: ov_genai.OmniPipeline) -> None:
    def text_tokens(**fields) -> list[int]:
        config = _text_config(1000)
        config.do_sample = True
        config.rng_seed = 7
        for name, value in fields.items():
            setattr(config, name, value)
        collector = TokenCollector()
        omni_pipe.generate(PROMPT, text_config=config, talker_speech_config=_speech_config(2), streamer=collector)
        return collector.tokens

    assert text_tokens(top_k=0) == text_tokens()


def test_rejects_echo(omni_pipe: ov_genai.OmniPipeline) -> None:
    config = _text_config(4)
    config.echo = True
    with pytest.raises(RuntimeError, match="echo"):
        omni_pipe.generate(PROMPT, text_config=config, talker_speech_config=_speech_config(4))


class FailingStreamer(ov_genai.StreamerBase):
    def write(self, token: int | list[int]) -> ov_genai.StreamingStatus:
        raise ValueError("streamer failure")

    def end(self) -> None:
        pass


class EndCounter(ov_genai.OmniSpeechStreamerBase):
    def __init__(self) -> None:
        super().__init__()
        self.ends = 0

    def write(self, chunk: ov.Tensor) -> ov_genai.StreamingStatus:
        return ov_genai.StreamingStatus.RUNNING

    def end(self) -> None:
        self.ends += 1


@pytest.mark.parametrize(
    "fields, streamer, error",
    [
        ({}, None, None),
        ({"speaker": "Ethan"}, None, "no speaker selection"),
        ({}, "failing", "streamer failure"),
    ],
    ids=["finished", "rejected_config", "failing_text_streamer"],
)
def test_speech_streamer_end_is_called_once(
    omni_pipe: ov_genai.OmniPipeline, fields: dict, streamer: str | None, error: str | None
) -> None:
    counter = EndCounter()
    kwargs = {"streamer": FailingStreamer()} if streamer == "failing" else {}

    def run() -> None:
        omni_pipe.generate(
            PROMPT,
            text_config=_text_config(1000),
            talker_speech_config=_speech_config(2, **fields),
            speech_streamer=counter,
            **kwargs,
        )

    if error is None:
        run()
    else:
        with pytest.raises(Exception, match=error):
            run()
    assert counter.ends == 1


def test_vlm_pipeline_ends_speech_streamer_on_rejected_config(vlm_pipe: ov_genai.VLMPipeline) -> None:
    # Called on VLMPipeline directly: PA rejects speech output, SDPA rejects the speaker.
    counter = EndCounter()
    with pytest.raises(RuntimeError):
        vlm_pipe.generate(
            PROMPT,
            generation_config=_text_config(4),
            talker_speech_config=_speech_config(2, speaker="Ethan"),
            speech_streamer=counter,
        )
    assert counter.ends == 1


def test_tool_call_stop_ends_the_turn(omni_pipe: ov_genai.OmniPipeline) -> None:
    collector = TokenCollector(ov_genai.StreamingStatus.TOOL_CALL_STOP)
    result = omni_pipe.generate(
        PROMPT, text_config=_text_config(1000), talker_speech_config=_speech_config(8), streamer=collector
    )
    assert len(collector.tokens) == 1
    assert result.finish_reasons == [ov_genai.GenerationFinishReason.TOOL_CALL]
    assert result.speech_result.waveforms == []


def test_speech_submodel_properties_are_accepted(lfm3_omni_model_path: Path, frame_samples: int) -> None:
    roles = ["depth_decoder", "audio_out_embeddings", "audio_decoder"]
    properties = {"MODEL_PROPERTIES": {role: {"INFERENCE_PRECISION_HINT": "f32"} for role in roles}}
    pipe = ov_genai.OmniPipeline(lfm3_omni_model_path, "CPU", **properties)
    result = pipe.generate(PROMPT, text_config=_text_config(1000), talker_speech_config=_speech_config(2))
    assert _waveform(result).size == 2 * frame_samples


def test_text_only_returns_no_speech(omni_pipe: ov_genai.OmniPipeline) -> None:
    result = omni_pipe.generate(
        PROMPT, text_config=_text_config(TEXT_TOKENS), talker_speech_config=_speech_config(4, return_audio=False)
    )
    assert result.speech_result.waveforms == []
    assert result.texts[0]


def test_has_no_talker(omni_pipe: ov_genai.OmniPipeline) -> None:
    assert omni_pipe.get_talker() is None


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"speaker": "Ethan"}, "no speaker selection"),
        ({"talker_top_k": 5}, "no separate talker"),
    ],
)
def test_rejects_talker_fields(omni_pipe: ov_genai.OmniPipeline, fields: dict, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        omni_pipe.generate(PROMPT, text_config=_text_config(4), talker_speech_config=_speech_config(4, **fields))


def test_rejects_talker_construction(lfm3_omni_model_path: Path) -> None:
    with pytest.raises(RuntimeError, match="LFM3-Omni has no separate talker"):
        ov_genai.Talker(lfm3_omni_model_path, "CPU")


def test_paged_attention_returns_text_only(omni_pa_pipe: ov_genai.OmniPipeline) -> None:
    result = omni_pa_pipe.generate(
        PROMPT, text_config=_text_config(TEXT_TOKENS), talker_speech_config=_speech_config(4, return_audio=False)
    )
    assert result.texts[0]
    with pytest.raises(RuntimeError, match="ATTENTION_BACKEND=SDPA"):
        omni_pa_pipe.generate(PROMPT, text_config=_text_config(4), talker_speech_config=_speech_config(4))


def test_speech_decode_recovers_after_failed_call(omni_pipe: ov_genai.OmniPipeline) -> None:
    # The interleaved answer opens with text slots, so the streamer fails after the prompt is tokenized.
    config = _speech_config(2)
    expected = _waveform(omni_pipe.generate(PROMPT, text_config=_text_config(1000), talker_speech_config=config))
    with pytest.raises(Exception, match="streamer failure"):
        omni_pipe.generate(
            PROMPT, text_config=_text_config(1000), talker_speech_config=config, streamer=FailingStreamer()
        )
    actual = _waveform(omni_pipe.generate(PROMPT, text_config=_text_config(1000), talker_speech_config=config))
    np.testing.assert_array_equal(actual, expected)
