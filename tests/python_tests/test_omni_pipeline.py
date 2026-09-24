# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the public OmniPipeline API.

Public surface under test:

    openvino_genai.OmniPipeline
    openvino_genai.OmniTalkerSpeechConfig
    openvino_genai.OmniSpeechStreamerBase

The pybind variant alias OmniSpeechStreamerVariant is intentionally not in
the smoke import — variant aliases are not re-exported in __init__.py per the
existing AudioStreamerVariant convention.

Two groups live here. The model-free group covers imports, config defaults, AnyMap
update and validate() invariants. The audio group drives a real Qwen3-Omni export
through VLMPipeline to cover `<ov_genai_audio_N>` placement; it is marked
real_models and needs $QWEN3_OMNI_OV_MODEL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import openvino
import pytest

import openvino_genai as ov_genai
from openvino_genai import (
    ChatHistory,
    ContinuousBatchingPipeline,
    GenerationConfig,
    SchedulerConfig,
    VLMDecodedResults,
    VLMPipeline,
    VideoMetadata,
)
from utils.media_tags import ModalityType, get_universal_tag


class TestOmniPipelineImports:
    """Smoke tests: do the new public symbols exist at all?"""

    def test_omni_pipeline_imports(self) -> None:
        """All new public symbols must be importable from the top-level module."""
        from openvino_genai import (  # noqa: F401
            OmniPipeline,
            OmniSpeechStreamerBase,
            OmniTalkerSpeechConfig,
        )

    def test_old_symbol_is_gone(self) -> None:
        """The pre-migration OmniSpeechGenerationConfig symbol must not exist anymore.

        The rename to OmniTalkerSpeechConfig is a clean break with no deprecation
        alias. If this attribute survives, somebody re-exported the old name against
        the migration contract.
        """
        assert not hasattr(ov_genai, "OmniSpeechGenerationConfig"), (
            "OmniSpeechGenerationConfig was renamed to OmniTalkerSpeechConfig and must not be re-exported as an alias."
        )


class TestOmniTalkerSpeechConfig:
    """OmniTalkerSpeechConfig — defaults, validate(), AnyMap update, MRO shape."""

    def test_no_generation_config_inheritance(self) -> None:
        """Standalone struct: MRO must NOT contain GenerationConfig.

        The whole point of the migration is to break the historical
        `OmniSpeechGenerationConfig : public GenerationConfig` inheritance so a future
        omni model with a non-LLM talker doesn't drag GenerationConfig fields it
        doesn't use. pybind11 inserts its own `pybind11_object` into the MRO above
        `object`; that's a wrapping artifact, not the load-bearing inheritance check.
        """
        assert ov_genai.GenerationConfig not in ov_genai.OmniTalkerSpeechConfig.__mro__, (
            "OmniTalkerSpeechConfig must NOT inherit from GenerationConfig — that's the whole point of the migration."
        )

        cfg = ov_genai.OmniTalkerSpeechConfig()
        assert not isinstance(cfg, ov_genai.GenerationConfig), (
            "OmniTalkerSpeechConfig instances must not be GenerationConfig instances."
        )

    def test_defaults(self) -> None:
        """Default ctor exposes the speech-side fields with sane defaults."""
        cfg = ov_genai.OmniTalkerSpeechConfig()

        assert cfg.return_audio is True, "return_audio default must be True"
        assert cfg.speaker == "", "speaker default must be empty (model default)"
        assert cfg.audio_chunk_frames == 4, "audio_chunk_frames default must be 4"
        assert cfg.rng_seed == 0, "rng_seed default must be 0"
        # talker_*/cp_* sampling overrides are std::optional<T> — exposed as None when unset.
        assert cfg.talker_temperature is None
        assert cfg.talker_top_k is None
        assert cfg.talker_repetition_penalty is None
        assert cfg.cp_temperature is None
        assert cfg.cp_top_k is None

    def test_direct_field_assignment(self) -> None:
        """Direct field assignment sets the speech-side fields."""
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.return_audio = False
        cfg.speaker = "any_voice_id"
        cfg.audio_chunk_frames = 2
        cfg.rng_seed = 7
        cfg.talker_temperature = 0.7
        cfg.talker_top_k = 20
        cfg.cp_temperature = 0.5
        cfg.cp_top_k = 10

        assert cfg.return_audio is False
        assert cfg.speaker == "any_voice_id"
        assert cfg.audio_chunk_frames == 2
        assert cfg.rng_seed == 7
        assert cfg.talker_temperature == pytest.approx(0.7)
        assert cfg.talker_top_k == 20
        assert cfg.cp_temperature == pytest.approx(0.5)
        assert cfg.cp_top_k == 10

    def test_max_new_tokens_field(self) -> None:
        """OmniTalkerSpeechConfig carries its own max_new_tokens (talker AR cap).

        Independent of GenerationConfig.max_new_tokens (which caps the thinker text
        decode). Both can be set simultaneously to different values when the caller
        constructs typed configs explicitly.
        """
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.max_new_tokens = 128
        assert cfg.max_new_tokens == 128


class TestOmniPipelineAccessors:
    """OmniPipeline getter/setter surface — methods must exist with the right signatures.

    No model is loaded here, so we only assert presence and signatures via the unbound
    method handle. End-to-end behavior (get/set round-trip on a live pipeline) lives in
    the temp/ scripts that load the MoE checkpoint.
    """

    def test_methods_exist(self) -> None:
        assert hasattr(ov_genai.OmniPipeline, "get_talker"), "OmniPipeline.get_talker() missing from public surface"

        # Speaker APIs live on the Talker, accessed via get_talker()
        for method in ("list_speakers", "get_speaker_embedding"):
            assert hasattr(ov_genai.TalkerBase, method), f"TalkerBase.{method}() missing from public surface"

    def test_speech_config_accessors_live_on_base(self) -> None:
        """get/set_speech_config are part of the TalkerBase interface every backend implements.

        TalkerBase declares them pure virtual, so the accessors and the property-bag generate()
        overload that seeds from them are part of the contract for every backend, not just the
        default Qwen3-Omni Talker, which stores the config itself.
        """
        for method in ("get_speech_config", "set_speech_config"):
            assert hasattr(ov_genai.TalkerBase, method), f"TalkerBase.{method}() missing from public surface"
            assert hasattr(ov_genai.Talker, method), f"Talker.{method}() missing from public surface"

    def test_talker_blob_ctor_signature(self) -> None:
        """Talker exposes the ModelsMap/device_mapping blob constructor (slide 10 spec).

        Calling with a bogus models_map must raise (missing submodels), not TypeError — that
        proves the overload resolves and reaches C++ construction rather than being absent.
        The disk-path constructor stays available alongside it.
        """
        empty_models_map: dict[str, object] = {}
        empty_device_mapping: dict[str, str] = {}
        with pytest.raises(Exception) as exc_info:
            ov_genai.Talker(empty_models_map, ov_genai.OmniTalkerSpeechConfig(), ".", empty_device_mapping)
        # Must not be a signature-resolution failure — the overload has to exist.
        assert not isinstance(exc_info.value, TypeError), f"blob ctor overload did not resolve: {exc_info.value}"


# The tiny CI model's tokenizer lacks the audio tokens, so audio dies in the merge assert; audio
# tests use a real export instead. `expected_audio_pads` is the upstream formula, which the
# encoder matches since the disjoint-window fix (CVS-193623).
AUDIO_SAMPLE_RATE = 16000
AUDIO_TONE_HZ = 440


def make_audio(seconds: float) -> openvino.Tensor:
    """Mono float32 PCM sine tone at 16 kHz — the layout Qwen3-Omni's feature extractor expects."""
    samples = np.arange(int(AUDIO_SAMPLE_RATE * seconds))
    return openvino.Tensor(np.sin(2 * np.pi * AUDIO_TONE_HZ * samples / AUDIO_SAMPLE_RATE).astype(np.float32))


def expected_audio_pads(num_frames: int, n_window: int = 50, pads_per_chunk: int = 13) -> int:
    """Upstream processor's audio pad count, in mel frames (100 frames per second at 16 kHz)."""
    chunk_len = n_window * 2
    leave = num_frames % chunk_len
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (num_frames // chunk_len) * pads_per_chunk


@pytest.fixture(scope="session")
def audio_05s_tensor() -> openvino.Tensor:
    return make_audio(0.5)


@pytest.fixture(scope="session")
def audio_1s_tensor() -> openvino.Tensor:
    return make_audio(1.0)


@pytest.fixture(scope="session")
def audio_2s_tensor() -> openvino.Tensor:
    return make_audio(2.0)


def test_expected_audio_pads_formula():
    """Pin the transcribed upstream formula against its published pad counts, with no model loaded."""
    assert expected_audio_pads(50) == 7
    assert expected_audio_pads(100) == 13
    assert expected_audio_pads(200) == 26
    assert expected_audio_pads(500) == 65
    assert expected_audio_pads(3000) == 390


def audio_frames(seconds: float) -> int:
    """Mel frames for a duration: Whisper hop_length=160 at 16 kHz gives 100 frames/second."""
    return int(AUDIO_SAMPLE_RATE * seconds) // 160


def test_audio_pads_per_duration():
    """Pin the pad count for the durations the audio tests use, with no model loaded.

    The encoder splits the mel spectrogram into disjoint windows of ``n_window * 2`` frames
    (CVS-193623), so the count matches the upstream processor exactly. A regression in the
    arithmetic is caught here without loading anything.
    """
    measured = {0.08: 1, 0.16: 2, 0.5: 7, 1.0: 13, 2.0: 26, 4.0: 52}
    for seconds, pads in measured.items():
        assert expected_audio_pads(audio_frames(seconds)) == pads, (
            f"formula disagrees with the expected pad count for {seconds}s audio"
        )


# Marked real_models because the CI tiny model cannot run audio at all: tiny-random-qwen3-omni
# declares audio_token_id=9 while its tokenizer maps <|AUDIO|> to 267 and has no <|audio_pad|>,
# so audio never matches. CVS-194799 tracks fixing the checkpoint; until then these need a real
# export supplied by the caller, and pytest.ini excludes them by default.
QWEN3_OMNI_OV_MODEL_ENV = "QWEN3_OMNI_OV_MODEL"


@pytest.fixture(scope="session")
def qwen3_omni_ov_model() -> str:
    """Path to a real Qwen3-Omni OV export, taken from $QWEN3_OMNI_OV_MODEL."""
    raw_path = os.environ.get(QWEN3_OMNI_OV_MODEL_ENV)
    if not raw_path:
        pytest.skip(f"Set ${QWEN3_OMNI_OV_MODEL_ENV} to a Qwen3-Omni OV export to run the audio tests")
    path = Path(raw_path)
    if not (path / "openvino_audio_encoder_model.xml").exists():
        pytest.skip(f"No Qwen3-Omni export with an audio encoder at {path}")
    return str(path)


@pytest.fixture(scope="session")
def qwen3_omni_n_window(qwen3_omni_ov_model: str) -> int:
    """audio_config.n_window of the model under test — sets the chunk width, so the pad count."""
    config = json.loads((Path(qwen3_omni_ov_model) / "config.json").read_text())
    thinker = config.get("thinker_config", config)
    n_window = thinker.get("audio_config", {}).get("n_window")
    assert n_window, "audio_config.n_window missing; cannot predict pad counts"
    return int(n_window)


@pytest.fixture(scope="session")
def synthetic_video_32x32_tensor() -> openvino.Tensor:
    """A deterministic 10-frame 32x32 RGB video.

    The audio+video tests pass the same video to both runs so its pads cancel and only the audio
    contributes to the difference. Content is therefore irrelevant — only that it never varies.
    """
    frames = np.zeros((10, 32, 32, 3), dtype=np.uint8)
    for index in range(frames.shape[0]):
        frames[index] = (index * 25) % 256
    return openvino.Tensor(frames)


# ----------------------------------------------------------------------------------------------
# T-C: audio placement on the prompt path. Token assertions are same-skeleton differentials: two
# runs, byte-identical prompt, only the tensor differs. Deleting a tag is not a valid baseline.
#
# Token counts are blind to where an expansion landed, so each placement test is paired with a
# native-tag equivalence test. Pad counts come from expected_audio_pads.
AUDIO_MAX_NEW_TOKENS = 20

# Index-independent by design: every audio uses the same triple, ordering is positional. This is
# the real export's tag; the tiny CI model has none of these tokens.
NATIVE_AUDIO_TAG = "<|audio_start|><|audio_pad|><|audio_end|>"

# The minimum-contribution control: short enough that the encoder emits exactly one pad, so
# `<|audio_start|><|audio_pad|><|audio_end|>` expands to a string identical to the native tag.
# That is the input a `find`-from-zero expansion loop mis-handles (plan R8).
ONE_PAD_AUDIO_SECONDS = 0.08


def audio_generation_config() -> GenerationConfig:
    """Greedy, fixed length — deterministic O1 so two formulations can be compared textually."""
    return GenerationConfig(max_new_tokens=AUDIO_MAX_NEW_TOKENS, do_sample=False, ignore_eos=True)


def audio_tag(index: int) -> str:
    return get_universal_tag(ModalityType.AUDIO, index)


def pads(seconds: float, n_window: int) -> int:
    return expected_audio_pads(audio_frames(seconds), n_window)


@pytest.fixture(scope="session")
def audio_8s_tensor() -> openvino.Tensor:
    return make_audio(8.0)


@pytest.fixture(scope="session")
def audio_1pad_tensor() -> openvino.Tensor:
    return make_audio(ONE_PAD_AUDIO_SECONDS)


@pytest.fixture(scope="session")
def audio_empty_tensor() -> openvino.Tensor:
    return openvino.Tensor(np.zeros(0, dtype=np.float32))


@pytest.fixture(scope="session")
def qwen3_omni_pipe_pa(qwen3_omni_ov_model: str) -> VLMPipeline:
    return VLMPipeline(qwen3_omni_ov_model, "CPU", ATTENTION_BACKEND="PA")


# Qwen3-Omni inherits the qwen3-vl SDPA defect: a text-only request fails with a [cpu]reshape
# error, no audio involved. The SDPA audio tests xfail only while that exact failure reproduces;
# any other error in them is a real failure.
QWEN3_OMNI_SDPA_DEFECT = "conflicts with the reshape pattern"


@pytest.fixture(scope="session")
def qwen3_omni_sdpa_text_only_error(qwen3_omni_ov_model: str) -> str | None:
    """The known SDPA error on a text-only request, or None once the defect is fixed."""
    # Own pipeline: a failed generate leaves the embedder state dirty for the next call on that pipe.
    pipe = VLMPipeline(qwen3_omni_ov_model, "CPU", ATTENTION_BACKEND="SDPA")
    history = ChatHistory()
    history.append({"role": "user", "content": "Describe"})
    try:
        pipe.generate(history, generation_config=GenerationConfig(max_new_tokens=1))
    except RuntimeError as error:
        if QWEN3_OMNI_SDPA_DEFECT not in str(error):
            raise
        return str(error).strip().splitlines()[-1]
    return None


@pytest.fixture
def qwen3_omni_sdpa_known_defect(qwen3_omni_sdpa_text_only_error: str | None) -> None:
    if qwen3_omni_sdpa_text_only_error is not None:
        pytest.xfail(f"qwen3-omni inherits the qwen3-vl SDPA defect: {qwen3_omni_sdpa_text_only_error}")


@pytest.fixture(scope="session")
def qwen3_omni_pipe_sdpa(qwen3_omni_ov_model: str) -> VLMPipeline:
    return VLMPipeline(qwen3_omni_ov_model, "CPU", ATTENTION_BACKEND="SDPA")


def audio_run(pipe: VLMPipeline, prompt: str, audios: list[openvino.Tensor]) -> VLMDecodedResults:
    return pipe.generate(prompt, audios=audios, generation_config=audio_generation_config())


def num_input_tokens(results: VLMDecodedResults) -> int:
    # A method, not a property: `perf_metrics.num_input_tokens` raises AttributeError.
    return results.perf_metrics.get_num_input_tokens()


def audio_encodings(results: VLMDecodedResults) -> int:
    return len(results.perf_metrics.vlm_raw_metrics.audio_encoding_durations)


def assert_pad_delta(hi: VLMDecodedResults, lo: VLMDecodedResults, expected_delta: int, what: str) -> None:
    __tracebackhide__ = True
    actual = num_input_tokens(hi) - num_input_tokens(lo)
    assert actual == expected_delta, (
        f"{what}: expected the token count to grow by {expected_delta} when only the audio tensor "
        f"changes under a byte-identical prompt, got {actual} "
        f"(hi={num_input_tokens(hi)}, lo={num_input_tokens(lo)})"
    )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_prepend_equals_explicit_tag_at_front(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    """The no-tag default must be exactly the same prompt as an explicit tag at the front.

    This is the in-repo form of the prepend-compatibility gate: it needs no committed golden and
    no absolute token count, because the two formulations must normalize to the same prompt.
    """
    implicit = audio_run(qwen3_omni_pipe_pa, "Describe", [audio_1s_tensor])
    explicit = audio_run(qwen3_omni_pipe_pa, audio_tag(0) + "Describe", [audio_1s_tensor])

    assert implicit.texts == explicit.texts
    assert num_input_tokens(implicit) == num_input_tokens(explicit)


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_universal_append(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """A trailing tag must consume its own audio: the pad budget tracks the tensor's duration."""
    prompt = "Describe this: " + audio_tag(0)
    hi = audio_run(qwen3_omni_pipe_pa, prompt, [audio_2s_tensor])
    lo = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1s_tensor])

    assert_pad_delta(
        hi,
        lo,
        pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window),
        "appended audio tag",
    )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_universal_append_matches_native_tag(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    """A trailing universal tag must resolve to the native tag in the same position.

    The paired placement half of `test_audio_universal_append`: token counts alone cannot see
    which span an audio landed in, but two spellings of the same skeleton can only agree if the
    universal tag was consumed at its own position instead of being left as literal text.
    """
    universal = audio_run(qwen3_omni_pipe_pa, "Describe this: " + audio_tag(0), [audio_1s_tensor])
    native = audio_run(qwen3_omni_pipe_pa, "Describe this: " + NATIVE_AUDIO_TAG, [audio_1s_tensor])

    assert universal.texts == native.texts
    assert num_input_tokens(universal) == num_input_tokens(native)


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_universal_interleaved(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """A tag surrounded by text on both sides must consume its audio there."""
    prompt = "text " + audio_tag(0) + " more text"
    hi = audio_run(qwen3_omni_pipe_pa, prompt, [audio_2s_tensor])
    lo = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1s_tensor])

    assert_pad_delta(
        hi,
        lo,
        pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window),
        "interleaved audio tag",
    )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_universal_interleaved_matches_native_tag(
    qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor
):
    """An interleaved universal tag must resolve to the native tag in the same position."""
    universal = audio_run(qwen3_omni_pipe_pa, "text " + audio_tag(0) + " more text", [audio_1s_tensor])
    native = audio_run(qwen3_omni_pipe_pa, "text " + NATIVE_AUDIO_TAG + " more text", [audio_1s_tensor])

    assert universal.texts == native.texts
    assert num_input_tokens(universal) == num_input_tokens(native)


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_two_distinct_indices(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """The second tag's span is sized by the second tensor, not by a shared aggregate."""
    prompt = "A " + audio_tag(0) + " B " + audio_tag(1)
    hi = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1s_tensor, audio_2s_tensor])
    lo = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1s_tensor, audio_1s_tensor])

    assert_pad_delta(
        hi,
        lo,
        pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window),
        "second of two audio tags",
    )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_two_universal_tags_match_native_tags(
    qwen3_omni_pipe_pa: VLMPipeline,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """Two universal tags must resolve to two native tags in the same two positions.

    The placement half of `test_audio_two_distinct_indices`: pad totals are conserved when an
    expansion lands in the wrong span, so only comparing two spellings of the same skeleton can
    show that each tag was consumed where it stands.
    """
    audios = [audio_1s_tensor, audio_2s_tensor]
    universal = audio_run(qwen3_omni_pipe_pa, "A " + audio_tag(0) + " B " + audio_tag(1), audios)
    native = audio_run(qwen3_omni_pipe_pa, "A " + NATIVE_AUDIO_TAG + " B " + NATIVE_AUDIO_TAG, audios)

    assert universal.texts == native.texts
    assert num_input_tokens(universal) == num_input_tokens(native)


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_two_indices_order_matters(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """Swapping the two indices must swap which audio lands in which run.

    Both orders must run without raising, and the totals must match. Completing at all is the
    load-bearing part: the runs are 13 and 25 pads long, so a merge that bound in document order
    instead of by index would try to copy the 13-token audio into the 25-token run and trip the
    run-length assert in merge_audio_embeddings(). Deterministic proof that the sequence itself
    reverses lives in MediaTagNormalization.ReversedIndicesReverseTheSequenceNotThePositions.

    Deliberately does NOT assert the generated text differs: both fixtures are 440 Hz sine tones,
    which carry nothing the model can tell apart, so it emits the same greedy continuation for
    either order. That assertion would be unprovable here rather than merely strict.
    """
    audios = [audio_1s_tensor, audio_2s_tensor]
    forward = audio_run(qwen3_omni_pipe_pa, "A " + audio_tag(0) + " B " + audio_tag(1), audios)
    swapped = audio_run(qwen3_omni_pipe_pa, "A " + audio_tag(1) + " B " + audio_tag(0), audios)

    assert num_input_tokens(forward) == num_input_tokens(swapped)

    # Same-skeleton differential: byte-identical prompt, only the second tensor's duration changes,
    # so the token delta must be exactly that audio's pad-count delta.
    shorter = audio_run(
        qwen3_omni_pipe_pa, "A " + audio_tag(0) + " B " + audio_tag(1), [audio_1s_tensor, audio_1s_tensor]
    )
    delta = pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window)
    assert_pad_delta(forward, shorter, delta, "second audio's own run grows with its duration")


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_duplicate_index_renders_twice(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """Referring to the same audio twice expands both occurrences — the delta doubles."""
    prompt = audio_tag(0) + " and " + audio_tag(0)
    hi = audio_run(qwen3_omni_pipe_pa, prompt, [audio_2s_tensor])
    lo = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1s_tensor])

    expected = 2 * (pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window))
    assert_pad_delta(hi, lo, expected, "duplicated audio index")


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_sub_second(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_05s_tensor: openvino.Tensor,
    audio_1s_tensor: openvino.Tensor,
):
    """Audio shorter than one encoder chunk still expands to its own pad count."""
    prompt = "Describe " + audio_tag(0)
    hi = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1s_tensor])
    lo = audio_run(qwen3_omni_pipe_pa, prompt, [audio_05s_tensor])

    assert_pad_delta(
        hi,
        lo,
        pads(1.0, qwen3_omni_n_window) - pads(0.5, qwen3_omni_n_window),
        "sub-second audio",
    )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_one_pad_audio_still_expands(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1pad_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """An audio worth exactly one pad must not swallow the following tag's expansion.

    A one-pad expansion leaves the native tag byte-identical to its unexpanded form, so an
    expansion loop that restarts its search from position zero writes the second audio's pads into
    the first tag's slot. O2 only sees that the second audio expanded at all; which span it landed
    in is pinned by `Qwen3OmniAudioExpansion.OnePadAudioDoesNotSwallowNextTag`.
    """
    assert pads(ONE_PAD_AUDIO_SECONDS, qwen3_omni_n_window) == 1, (
        "the one-pad control no longer yields a single pad on this model"
    )
    prompt = "A " + audio_tag(0) + " B " + audio_tag(1)
    hi = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1pad_tensor, audio_2s_tensor])
    lo = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1pad_tensor, audio_1pad_tensor])

    assert_pad_delta(hi, lo, pads(2.0, qwen3_omni_n_window) - 1, "one-pad audio")


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_empty_list_adds_no_pads(qwen3_omni_pipe_pa: VLMPipeline):
    """An empty `audios` list with no tag must be indistinguishable from passing no audio at all."""
    with_empty_list = audio_run(qwen3_omni_pipe_pa, "Describe", [])
    without_argument = qwen3_omni_pipe_pa.generate("Describe", generation_config=audio_generation_config())

    assert num_input_tokens(with_empty_list) == num_input_tokens(without_argument)
    assert audio_encodings(with_empty_list) == 0


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_empty_tensor_keeps_its_index(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_empty_tensor: openvino.Tensor,
):
    """An empty audio contributes zero pads while still occupying its own index.

    If the empty entry were dropped from the list instead, the second tag would bind to the first
    tensor and the delta would collapse to zero.
    """
    prompt = "A " + audio_tag(0) + " B " + audio_tag(1)
    hi = audio_run(qwen3_omni_pipe_pa, prompt, [audio_1s_tensor, audio_1s_tensor])
    lo = audio_run(qwen3_omni_pipe_pa, prompt, [audio_empty_tensor, audio_1s_tensor])

    assert_pad_delta(hi, lo, pads(1.0, qwen3_omni_n_window), "empty audio tensor")


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_tag_without_any_audio_rejected(qwen3_omni_pipe_pa: VLMPipeline):
    with pytest.raises(RuntimeError, match="Missing image/video/audio with index 0"):
        audio_run(qwen3_omni_pipe_pa, audio_tag(0), [])


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_index_out_of_range_rejected(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    with pytest.raises(RuntimeError, match="Missing image/video/audio with index 5"):
        audio_run(qwen3_omni_pipe_pa, audio_tag(5), [audio_1s_tensor])


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_mixed_universal_and_native_rejected(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    with pytest.raises(RuntimeError, match="Prompt cannot mix universal tags"):
        audio_run(qwen3_omni_pipe_pa, NATIVE_AUDIO_TAG + audio_tag(0), [audio_1s_tensor])


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_native_tag_count_mismatch_rejected(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    with pytest.raises(RuntimeError, match="The number of native media tags must match"):
        audio_run(qwen3_omni_pipe_pa, NATIVE_AUDIO_TAG * 2, [audio_1s_tensor])


# ----------------------------------------------------------------------------------------------
# T-D: audio across chat turns, multipart messages, and the encoder cache. Where turn 1 differs
# between runs, its generated reply enters turn 2 and cannot be predicted exactly.
STALE_TURN_MAX_NEW_TOKENS = 1


def audio_conversation(
    pipe: VLMPipeline,
    turns: list[tuple[str, list[openvino.Tensor]]],
    max_new_tokens: int = AUDIO_MAX_NEW_TOKENS,
) -> list:
    """Run a multi-turn chat, always leaving chat mode even if a turn raises."""
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, ignore_eos=True)
    pipe.start_chat()
    try:
        return [pipe.generate(prompt, audios=audios, generation_config=config) for prompt, audios in turns]
    finally:
        pipe.finish_chat()


def audio_history_run(pipe: VLMPipeline, messages: list[dict], audios: list[openvino.Tensor]):
    """One `generate` call over an explicit history — no generated reply to confound the counts."""
    history = ChatHistory()
    for message in messages:
        history.append(message)
    return pipe.generate(history, audios=audios, generation_config=audio_generation_config())


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_stale_across_turns(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_8s_tensor: openvino.Tensor,
):
    """Turn 1's audio must be counted once, not re-attached to turn 2.

    Both conversations use byte-identical prompts on both turns; only turn 1's tensor differs. Turn
    2 therefore differs by turn 1's pad count exactly once. If turn 1's audio leaks into turn 2 the
    difference doubles — two orders of magnitude outside the tolerance below.

    The +-2 band absorbs the single generated token that turn 1 contributes to turn 2's history,
    which differs between the two conversations because their audio differs.
    """
    short_pads = pads(1.0, qwen3_omni_n_window)
    long_pads = pads(8.0, qwen3_omni_n_window)

    def turns(audio: openvino.Tensor) -> list[tuple[str, list[openvino.Tensor]]]:
        return [("Describe " + audio_tag(0), [audio]), ("And now?", [])]

    with_short = audio_conversation(
        qwen3_omni_pipe_pa, turns(audio_1s_tensor), max_new_tokens=STALE_TURN_MAX_NEW_TOKENS
    )
    with_long = audio_conversation(qwen3_omni_pipe_pa, turns(audio_8s_tensor), max_new_tokens=STALE_TURN_MAX_NEW_TOKENS)

    measured = num_input_tokens(with_short[1]) - num_input_tokens(with_long[1])
    expected = short_pads - long_pads
    assert abs(measured - expected) <= 2, (
        f"turn 2 differs by {measured} tokens, expected {expected} +-2. A difference near "
        f"{2 * expected} means turn 1's audio was counted again on turn 2."
    )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_binds_only_to_its_message(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """An audio tagged in one user message must expand in that message only, not in every one.

    The pre-fix code held audio on the embedder and prepended a block to each user message it
    normalized, so a history with two user messages got the audio twice. The delta below is one
    audio's worth; two would be double.

    The tag sits in the LAST user message because that is where call-time media attaches, for
    every modality — `fill_messages_metadata` assigns `provided_*_indices` to
    `get_last_user_message_index()`. Tagging an earlier message in a single call refers to media
    that was never registered for it, and `verify_ids` rejects it. Earlier messages carry their own
    media only when the history was built up across turns, which
    `test_audio_stale_across_turns` covers.

    Assistant turns are supplied rather than generated, so no reply text can confound the counts.
    """
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Second " + audio_tag(0)},
    ]
    hi = audio_history_run(qwen3_omni_pipe_pa, messages, [audio_2s_tensor])
    lo = audio_history_run(qwen3_omni_pipe_pa, messages, [audio_1s_tensor])

    assert_pad_delta(
        hi,
        lo,
        pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window),
        "one audio across a two-user-message history, not one per user message",
    )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_multipart_matches_string_history(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    """An `{"type": "audio"}` part must be the same prompt as a universal tag in the same place."""
    multipart = audio_history_run(
        qwen3_omni_pipe_pa,
        [{"role": "user", "content": [{"type": "text", "text": "Describe "}, {"type": "audio"}]}],
        [audio_1s_tensor],
    )
    string_form = audio_history_run(
        qwen3_omni_pipe_pa,
        [{"role": "user", "content": "Describe " + audio_tag(0)}],
        [audio_1s_tensor],
    )

    assert multipart.texts == string_form.texts
    assert num_input_tokens(multipart) == num_input_tokens(string_form)


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_multipart_audio_first(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    """An audio part before the text part must keep that order."""
    multipart = audio_history_run(
        qwen3_omni_pipe_pa,
        [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Describe"}]}],
        [audio_1s_tensor],
    )
    string_form = audio_history_run(
        qwen3_omni_pipe_pa,
        [{"role": "user", "content": audio_tag(0) + "Describe"}],
        [audio_1s_tensor],
    )

    assert multipart.texts == string_form.texts
    assert num_input_tokens(multipart) == num_input_tokens(string_form)


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_multipart_audio_only(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    """A message with only an audio part used to fail format detection as an unknown schema."""
    multipart = audio_history_run(
        qwen3_omni_pipe_pa, [{"role": "user", "content": [{"type": "audio"}]}], [audio_1s_tensor]
    )
    string_form = audio_history_run(qwen3_omni_pipe_pa, [{"role": "user", "content": audio_tag(0)}], [audio_1s_tensor])

    assert multipart.texts == string_form.texts
    assert num_input_tokens(multipart) == num_input_tokens(string_form)


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_multipart_unsupported_type_still_throws(
    qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor
):
    """Adding an audio branch to the multipart parser must not swallow the unknown-type fallthrough."""
    with pytest.raises(RuntimeError, match="Unsupported content type in multipart message: hologram"):
        audio_history_run(
            qwen3_omni_pipe_pa,
            [{"role": "user", "content": [{"type": "text", "text": "Describe "}, {"type": "hologram"}]}],
            [audio_1s_tensor],
        )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_identical_tensor_not_reencoded(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    """Re-sending the same tensor on a later turn must hit the content-hash cache, not re-encode.

    Uses the ChatHistory overload on one persistent ChatHistory object, because that is where the
    cache lives: `VLMChatContext` registers media in the `VisionRegistry` and skips encoding on a
    hash hit. The `start_chat()` + prompt path accumulates `m_history_*` and re-encodes each turn
    instead — for images and video too, not just audio — and a fresh ChatHistory per call would
    release the registry refs and drop the entry.
    """
    history = ChatHistory()
    history.append({"role": "user", "content": "Describe " + audio_tag(0)})
    first = qwen3_omni_pipe_pa.generate(history, audios=[audio_1s_tensor], generation_config=audio_generation_config())

    history.append({"role": "assistant", "content": first.texts[0]})
    history.append({"role": "user", "content": "And this? " + audio_tag(1)})
    second = qwen3_omni_pipe_pa.generate(history, audios=[audio_1s_tensor], generation_config=audio_generation_config())

    assert audio_encodings(first) == 1
    assert audio_encodings(second) == 0, "an identical tensor must be served from the registry cache"


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_different_tensor_is_reencoded(
    qwen3_omni_pipe_pa: VLMPipeline,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """The control for the cache-hit test: a new tensor must actually be encoded."""
    turns = audio_conversation(
        qwen3_omni_pipe_pa,
        [
            ("Describe " + audio_tag(0), [audio_1s_tensor]),
            ("And this? " + audio_tag(1), [audio_2s_tensor]),
        ],
    )

    assert audio_encodings(turns[0]) == 1
    assert audio_encodings(turns[1]) == 1


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_index_absolute_across_turns(
    qwen3_omni_pipe_pa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """Indices are absolute across turns: turn 2's audio is `<ov_genai_audio_1>`.

    Turn 1 is identical in both conversations — same string, same tensor — so its reply and its
    pads cancel exactly and no tolerance is needed.
    """

    def conversation(turn2_audio: openvino.Tensor):
        return audio_conversation(
            qwen3_omni_pipe_pa,
            [
                ("Describe " + audio_tag(0), [audio_1s_tensor]),
                ("And now? " + audio_tag(1), [turn2_audio]),
            ],
            max_new_tokens=STALE_TURN_MAX_NEW_TOKENS,
        )

    hi = conversation(audio_2s_tensor)
    lo = conversation(audio_1s_tensor)

    assert_pad_delta(
        hi[1],
        lo[1],
        pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window),
        "audio attached on turn 2 under an absolute index",
    )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_older_turn_reference_rejected(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    """Referring back to a previous turn's audio is an inherited non-goal, and must say so."""
    with pytest.raises(RuntimeError, match="Referring to older images/videos/audios is not supported"):
        audio_conversation(
            qwen3_omni_pipe_pa,
            [
                ("Describe " + audio_tag(0), [audio_1s_tensor]),
                ("Again " + audio_tag(0), [audio_1s_tensor]),
            ],
            max_new_tokens=STALE_TURN_MAX_NEW_TOKENS,
        )


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_failed_turn_is_rolled_back(qwen3_omni_pipe_pa: VLMPipeline, audio_1s_tensor: openvino.Tensor):
    """A turn that throws after registering its audio must not shift the next turn's indices."""
    history = ChatHistory()
    history.append({"role": "user", "content": "Describe " + audio_tag(1)})
    with pytest.raises(RuntimeError):
        qwen3_omni_pipe_pa.generate(history, audios=[audio_1s_tensor], generation_config=audio_generation_config())

    history.pop()
    history.append({"role": "user", "content": "Describe " + audio_tag(0)})
    retried = qwen3_omni_pipe_pa.generate(
        history, audios=[audio_1s_tensor], generation_config=audio_generation_config()
    )
    fresh = audio_history_run(
        qwen3_omni_pipe_pa, [{"role": "user", "content": "Describe " + audio_tag(0)}], [audio_1s_tensor]
    )

    assert retried.texts == fresh.texts
    assert num_input_tokens(retried) == num_input_tokens(fresh)


@pytest.mark.real_models
@pytest.mark.vlm
def test_chat_turn_rolled_back_on_exception(qwen3_omni_pipe_pa: VLMPipeline):
    """A start_chat turn that throws after storing its video must not leave that video behind."""
    config = GenerationConfig(max_new_tokens=STALE_TURN_MAX_NEW_TOKENS, do_sample=False, ignore_eos=True)
    long_video = openvino.Tensor(np.zeros((8, 32, 32, 3), dtype=np.uint8))
    short_video = openvino.Tensor(np.zeros((4, 32, 32, 3), dtype=np.uint8))
    invalid_audio = openvino.Tensor(np.zeros((2, 16000), dtype=np.float32))

    def second_turn_tokens(fail_first: bool) -> int:
        qwen3_omni_pipe_pa.start_chat()
        try:
            qwen3_omni_pipe_pa.generate("Hi", generation_config=config)
            if fail_first:
                with pytest.raises(RuntimeError, match="1-D tensor"):
                    qwen3_omni_pipe_pa.generate(
                        "Describe", videos=[long_video], audios=[invalid_audio], generation_config=config
                    )
            return num_input_tokens(
                qwen3_omni_pipe_pa.generate("Describe", videos=[short_video], generation_config=config)
            )
        finally:
            qwen3_omni_pipe_pa.finish_chat()

    assert second_turn_tokens(fail_first=True) == second_turn_tokens(fail_first=False)


# ----------------------------------------------------------------------------------------------
# ContinuousBatchingPipeline with a batch of two, each item with its own audio. Batch size 1 is
# already covered: VLMPipeline with PA goes through the same code via the CB adapter.


@pytest.fixture(scope="session")
def qwen3_omni_cb(qwen3_omni_ov_model: str) -> ContinuousBatchingPipeline:
    return ContinuousBatchingPipeline(qwen3_omni_ov_model, SchedulerConfig(), "CPU")


def cb_audio_run(
    pipe: ContinuousBatchingPipeline, inputs: list[str] | list[ChatHistory], audios: list[openvino.Tensor]
) -> list[VLMDecodedResults]:
    """One audio per batch item."""
    return pipe.generate(
        inputs,
        images=[[] for _ in inputs],
        videos=[[] for _ in inputs],
        audios_batches=[[audio] for audio in audios],
        generation_config=[audio_generation_config() for _ in inputs],
    )


def assert_batch_matches_single_runs(batch: list[VLMDecodedResults], singles: list[VLMDecodedResults]) -> None:
    __tracebackhide__ = True
    for item, single in zip(batch, singles, strict=True):
        assert item.texts == single.texts
        assert num_input_tokens(item) == num_input_tokens(single)
    assert num_input_tokens(batch[0]) != num_input_tokens(batch[1]), "the two audios must differ in length"


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_cb_prompt_batch_keeps_audio_per_prompt(
    qwen3_omni_cb: ContinuousBatchingPipeline,
    audio_1s_tensor: openvino.Tensor,
    audio_8s_tensor: openvino.Tensor,
):
    prompt = "Describe " + audio_tag(0)
    audios = [audio_1s_tensor, audio_8s_tensor]
    batch = cb_audio_run(qwen3_omni_cb, [prompt, prompt], audios)
    singles = [cb_audio_run(qwen3_omni_cb, [prompt], [audio])[0] for audio in audios]
    assert_batch_matches_single_runs(batch, singles)


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_cb_chat_history_batch_keeps_audio_per_history(
    qwen3_omni_cb: ContinuousBatchingPipeline,
    audio_1s_tensor: openvino.Tensor,
    audio_8s_tensor: openvino.Tensor,
):
    messages = [{"role": "user", "content": "Describe " + audio_tag(0)}]
    audios = [audio_1s_tensor, audio_8s_tensor]
    batch = cb_audio_run(qwen3_omni_cb, [ChatHistory(messages), ChatHistory(messages)], audios)
    singles = [cb_audio_run(qwen3_omni_cb, [ChatHistory(messages)], [audio])[0] for audio in audios]
    assert_batch_matches_single_runs(batch, singles)


# ----------------------------------------------------------------------------------------------
# T-E: the SDPA `VLMPipeline` ChatHistory path. Its overload puts `audios` before
# `videos_metadata`, the opposite of `OmniPipeline`; both are vectors, so a swap compiles silently.
#
# Cross-backend checks assert on `texts` only: `num_input_tokens` counts different things on the
# two backends, so comparing it across them would fail for reasons unrelated to audio.


def audio_sdpa_history_run(
    pipe: VLMPipeline,
    prompt: str,
    audios: list[openvino.Tensor],
    videos: list[openvino.Tensor] | None = None,
):
    history = ChatHistory()
    history.append({"role": "user", "content": prompt})
    kwargs: dict[str, Any] = {"audios": audios, "generation_config": audio_generation_config()}
    if videos is not None:
        kwargs["videos"] = videos
        kwargs["videos_metadata"] = [VideoMetadata() for _ in videos]
    return pipe.generate(history, **kwargs)


@pytest.mark.real_models
@pytest.mark.vlm
@pytest.mark.usefixtures("qwen3_omni_sdpa_known_defect")
def test_vlm_chat_history_audio_is_encoded_sdpa(
    qwen3_omni_pipe_sdpa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """The SDPA ChatHistory path must encode audio instead of silently dropping it."""
    prompt = "Describe " + audio_tag(0)
    hi = audio_sdpa_history_run(qwen3_omni_pipe_sdpa, prompt, [audio_2s_tensor])
    lo = audio_sdpa_history_run(qwen3_omni_pipe_sdpa, prompt, [audio_1s_tensor])

    assert audio_encodings(hi) == 1
    assert audio_encodings(lo) == 1
    assert_pad_delta(
        hi,
        lo,
        pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window),
        "audio on the SDPA ChatHistory path",
    )


@pytest.mark.real_models
@pytest.mark.vlm
@pytest.mark.usefixtures("qwen3_omni_sdpa_known_defect")
def test_vlm_chat_history_audio_matches_pa_backend(
    qwen3_omni_pipe_sdpa: VLMPipeline,
    qwen3_omni_pipe_pa: VLMPipeline,
    audio_1s_tensor: openvino.Tensor,
):
    """Both backends decode greedily from the same weights, so the same history must yield the
    same text. A divergence means the two backends built different prompts."""
    prompt = "Describe " + audio_tag(0)
    sdpa = audio_sdpa_history_run(qwen3_omni_pipe_sdpa, prompt, [audio_1s_tensor])
    pa = audio_sdpa_history_run(qwen3_omni_pipe_pa, prompt, [audio_1s_tensor])

    assert sdpa.texts == pa.texts


@pytest.mark.real_models
@pytest.mark.vlm
@pytest.mark.usefixtures("qwen3_omni_sdpa_known_defect")
def test_vlm_chat_history_audio_only_no_video(
    qwen3_omni_pipe_sdpa: VLMPipeline,
    qwen3_omni_n_window: int,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """Audio with explicitly empty video arguments must still reach the audio encoder.

    Half of the argument-swap check: routing `audios` into the video slot would send a 1-D waveform
    through the vision tower and raise a shape error.
    """
    prompt = "Describe " + audio_tag(0)
    hi = audio_sdpa_history_run(qwen3_omni_pipe_sdpa, prompt, [audio_2s_tensor], videos=[])
    lo = audio_sdpa_history_run(qwen3_omni_pipe_sdpa, prompt, [audio_1s_tensor], videos=[])

    assert_pad_delta(
        hi,
        lo,
        pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window),
        "audio with explicitly empty videos",
    )


@pytest.mark.real_models
@pytest.mark.vlm
# Deliberately not xfailed: this passes on SDPA today. It is the control showing video works on
# that path, so an audio failure there is attributable rather than inherited.
def test_vlm_chat_history_video_only_no_audio(
    qwen3_omni_pipe_sdpa: VLMPipeline,
    synthetic_video_32x32_tensor: openvino.Tensor,
):
    """Video with an empty `audios` list must behave exactly as if `audios` were never passed.

    The other half of the argument-swap check, and the guard that adding audio plumbing to this
    overload does not perturb the video path.
    """
    prompt = "Describe " + get_universal_tag(ModalityType.VIDEO, 0)
    with_empty_audios = audio_sdpa_history_run(qwen3_omni_pipe_sdpa, prompt, [], videos=[synthetic_video_32x32_tensor])

    history = ChatHistory()
    history.append({"role": "user", "content": prompt})
    without_audios = qwen3_omni_pipe_sdpa.generate(
        history,
        videos=[synthetic_video_32x32_tensor],
        videos_metadata=[VideoMetadata()],
        generation_config=audio_generation_config(),
    )

    assert with_empty_audios.texts == without_audios.texts
    assert audio_encodings(with_empty_audios) == 0


@pytest.mark.real_models
@pytest.mark.vlm
@pytest.mark.usefixtures("qwen3_omni_sdpa_known_defect")
def test_vlm_chat_history_audio_and_video_together(
    qwen3_omni_pipe_sdpa: VLMPipeline,
    qwen3_omni_n_window: int,
    synthetic_video_32x32_tensor: openvino.Tensor,
    audio_1s_tensor: openvino.Tensor,
    audio_2s_tensor: openvino.Tensor,
):
    """Audio and video in one message must not mis-count each other.

    The video is identical in both runs, so its pads cancel and only the audio contributes to the
    delta. Generation *quality* for this combination is a documented limitation (audio pads are
    positioned as text tokens relative to vision tokens); this test only pins that it runs and
    counts correctly.
    """
    prompt = "Watch " + get_universal_tag(ModalityType.VIDEO, 0) + " and hear " + audio_tag(0)
    hi = audio_sdpa_history_run(qwen3_omni_pipe_sdpa, prompt, [audio_2s_tensor], videos=[synthetic_video_32x32_tensor])
    lo = audio_sdpa_history_run(qwen3_omni_pipe_sdpa, prompt, [audio_1s_tensor], videos=[synthetic_video_32x32_tensor])

    assert audio_encodings(hi) == 1
    assert audio_encodings(lo) == 1
    assert_pad_delta(
        hi,
        lo,
        pads(2.0, qwen3_omni_n_window) - pads(1.0, qwen3_omni_n_window),
        "audio alongside a video",
    )


@pytest.mark.real_models
@pytest.mark.vlm
@pytest.mark.usefixtures("qwen3_omni_sdpa_known_defect")
def test_vlm_prompt_path_audio_unchanged_sdpa(
    qwen3_omni_pipe_sdpa: VLMPipeline,
    qwen3_omni_pipe_pa: VLMPipeline,
    audio_1s_tensor: openvino.Tensor,
):
    """Wiring audio into the ChatHistory overload must not regress the prompt overload."""
    sdpa = audio_run(qwen3_omni_pipe_sdpa, "Describe", [audio_1s_tensor])
    pa = audio_run(qwen3_omni_pipe_pa, "Describe", [audio_1s_tensor])

    assert sdpa.texts == pa.texts
