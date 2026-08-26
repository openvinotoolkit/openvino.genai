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

Three tiers, following the other pipeline suites:

1. Model-free tests — imports, config defaults, field round-trips, and dependency
   injection of user-defined VLMPipelineBase / TalkerBase children. The DI tests
   drive the real C++ OmniPipeline composition path against Python-defined mocks,
   exercising virtual dispatch, the text vs speech branching, and the constructor
   capability guards end to end.

2. Real-model tests against `optimum-intel-internal-testing/tiny-random-qwen3-omni`,
   covering the path-based constructor, text-only and speech generation, the
   ChatHistory overload, generation-config sensitivity, ModelsMap/path equivalence,
   multimodal inputs, and the speaker APIs.

3. Full-model tests marked `real_models`, pointed at a complete Qwen3-Omni export by
   OMNI_REAL_MODEL_PATH. The tiny checkpoint cannot synthesize speech at all, so the
   tests that need a real waveform live here and are deselected by default.

The tiny-checkpoint tier needs a newer transformers than tests/python_tests/requirements.txt
pins, so the CI matrix entries install one per job: transformers 5.0.0 reads an unset
`use_sliding_window` in `Qwen3OmniMoeTalkerCodePredictorConfig.__init__` and the export dies
with an `AttributeError`, fixed in 5.1.0. The pinned optimum-intel already carries the
talker/code2wav export from huggingface/optimum-intel#1700. `omni_model_path` still detects
both gaps and skips with the reason, so running the suite against the repo-wide pins degrades
to the model-free tier instead of failing.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_tokenizers
import pytest
import transformers
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForVisualCausalLM
from optimum.utils.import_utils import is_transformers_version

import openvino_genai as ov_genai
from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir
from utils.network import retry_request

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.omni

OMNI_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-omni"

# Points at a full Qwen3-Omni OpenVINO export for the speech tests the tiny checkpoint cannot host.
OMNI_REAL_MODEL_ENV = "OMNI_REAL_MODEL_PATH"

# Written by the Talker export; without them OmniPipeline's path ctor cannot build the speech stage.
TALKER_ARTIFACTS = (
    "openvino_talker_model.xml",
    "openvino_code_predictor_model.xml",
    "openvino_code2wav_model.xml",
)

MEDIA_EDGE = 64
VIDEO_FRAMES = 4
AUDIO_SAMPLE_RATE = 16000
AUDIO_SAMPLES = AUDIO_SAMPLE_RATE

SPEECH_PROMPT = "Say hello."

TALKER_RNG_SEED = 1234
TALKER_MAX_NEW_TOKENS = 8

OPTIMUM_COMPARE_TOKENS = 6

SPEAKER_EMBEDDING_SCALE = -8.0

NO_WAVEFORM_XFAIL_REASON = (
    "tiny-random-qwen3-omni's talker role token ids are outside its tokenizer's range (im_start 151644, "
    "user 872, assistant 77091, against a max id of 769), so build_talker_input finds no segments and "
    "speech generation raises before reaching code2wav — see "
    "test_speech_generation_rejects_unmatched_role_tokens, which locks that error in. Same checkpoint "
    "limitation test_tools_llm_benchmark.py xfails its omni text_to_speech cases on, for both "
    "--optimum and --genai."
)

VIDEO_INPUT_XFAIL_REASON = (
    "tiny-random-qwen3-omni's processor_config.json declares video_processor.patch_size=14 (a stale "
    "Qwen2VL default) while image_processor and thinker_config.vision_config both say 16, so the video "
    "patch dim is 2*3*14*14=1176 against an encoder built for 1536 and the infer request fails "
    "shape.compatible(). Setting that one value to 16 makes video work on this checkpoint, and real "
    "Qwen3-Omni models already ship 16 — the encoder and the video path itself are fine."
)

OPTIMUM_IMAGE_XFAIL_REASON = (
    "GenAI and optimum agree exactly on generated token ids for a text-only prompt but diverge once an "
    "image is attached, while still agreeing on the 57-token input length — so they build the same "
    "sequence and merge different image embeddings into it. Undetermined whether that is a real "
    "preprocessing difference or this checkpoint's random weights making the greedy argmax flip on "
    "numerical noise: it cannot be arbitrated locally, because optimum only recognizes model_type "
    "qwen3_omni_moe and every full Qwen3-Omni export at hand is dense qwen3_omni."
)

AUDIO_INPUT_XFAIL_REASON = (
    "tiny-random-qwen3-omni is internally inconsistent for audio: thinker_config.audio_token_id is 9, "
    "while its tokenizer maps <|AUDIO|> to 267 and has no <|audio_pad|> token at all. GenAI injects "
    "<|audio_start|><|audio_pad|>...<|audio_end|> and merges audio features at audio_token_id positions, "
    "so nothing matches and merge_audio_embeddings() throws 'Audio token count mismatch: placed 0 "
    "embeddings'. Same root cause as the omni speech_to_text xfail in test_tools_llm_benchmark.py."
)


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
    method handle. End-to-end get/set behavior against injected pipelines is covered by
    TestOmniPipelineDependencyInjection.
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


class RecordingTalker(ov_genai.TalkerBase):
    """User-defined talker that counts invocations. Result fields are read-only from Python, so
    dispatch is proven by the call count rather than by round-tripping audio.
    """

    def __init__(self, speakers: list[str] | None = None) -> None:
        super().__init__()
        self.generate_calls = 0
        self.last_return_audio: bool | None = None
        self.last_speech_streamer: object | None = None
        self._speakers = speakers if speakers is not None else ["default_voice"]
        self._speech_config = ov_genai.OmniTalkerSpeechConfig()

    def generate(
        self,
        vlm_result: ov_genai.VLMDecodedResults,
        talker_speech_config: ov_genai.OmniTalkerSpeechConfig,
        speech_streamer: ov_genai.OmniSpeechStreamerBase | None = None,
    ) -> ov_genai.TalkerResults:
        self.generate_calls += 1
        self.last_return_audio = talker_speech_config.return_audio
        self.last_speech_streamer = speech_streamer
        return ov_genai.TalkerResults()

    # TalkerBase declares these pure virtual, so a Python backend has to store the config itself.
    def get_speech_config(self) -> ov_genai.OmniTalkerSpeechConfig:
        return self._speech_config

    def set_speech_config(self, config: ov_genai.OmniTalkerSpeechConfig) -> None:
        self._speech_config = config

    def list_speakers(self) -> list[str]:
        return self._speakers

    def get_speaker_embedding(self, name: str) -> ov.Tensor:
        return ov.Tensor(np.zeros((1, 1, 4), dtype=np.float32))


class RecordingVLM(ov_genai.VLMPipelineBase):
    """User-defined VLM (thinker) that counts generate() calls. Capability queries default to True
    so the ctor accepts the mock; tests flip them to exercise its guard assertions.
    """

    def __init__(self, audio_output: bool = True, hidden_states: bool = True) -> None:
        super().__init__()
        self.generate_calls = 0
        self.last_prompt: str | ov_genai.ChatHistory | None = None
        self.last_images: list[ov.Tensor] = []
        self.last_videos: list[ov.Tensor] = []
        self.last_audios: list[ov.Tensor] = []
        self.last_videos_metadata: list[ov_genai.VideoMetadata] = []
        self.last_max_new_tokens: int | None = None
        self._audio_output = audio_output
        self._hidden_states = hidden_states

    def generate(  # noqa: PLR0913
        self,
        prompt: str | ov_genai.ChatHistory,
        images: list[ov.Tensor] | None = None,
        videos: list[ov.Tensor] | None = None,
        audios: list[ov.Tensor] | None = None,
        videos_metadata: list[ov_genai.VideoMetadata] | None = None,
        generation_config: ov_genai.GenerationConfig | None = None,
        streamer: ov_genai.StreamerBase | Callable[[str], bool] | None = None,
    ) -> ov_genai.VLMDecodedResults:
        self.generate_calls += 1
        self.last_prompt = prompt
        self.last_images = list(images or [])
        self.last_videos = list(videos or [])
        self.last_audios = list(audios or [])
        self.last_videos_metadata = list(videos_metadata or [])
        self.last_max_new_tokens = generation_config.max_new_tokens if generation_config else None
        return ov_genai.VLMDecodedResults()

    def get_tokenizer(self) -> ov_genai.Tokenizer:
        raise RuntimeError("RecordingVLM.get_tokenizer() is not needed for the prompt-based path")

    def set_chat_template(self, chat_template: str) -> None:
        pass

    def get_generation_config(self) -> ov_genai.GenerationConfig:
        return ov_genai.GenerationConfig()

    def set_generation_config(self, config: ov_genai.GenerationConfig) -> None:
        pass

    def supports_hidden_states_collection(self) -> bool:
        return self._hidden_states

    def is_audio_output_enabled(self) -> bool:
        return self._audio_output


class TestCustomTalkerSubclass:
    """Users can define a TalkerBase child in Python and have its methods dispatched from C++."""

    def test_subclass_is_instantiable(self) -> None:
        """TalkerBase must expose a constructor so Python subclasses can be instantiated."""
        talker = RecordingTalker()
        assert isinstance(talker, ov_genai.TalkerBase)

    def test_overrides_are_dispatched(self) -> None:
        """Overridden list_speakers / get_speaker_embedding must call back into Python."""
        talker = RecordingTalker(speakers=["alice", "bob"])
        assert talker.list_speakers() == ["alice", "bob"]
        embedding = talker.get_speaker_embedding("alice")
        assert embedding.shape == [1, 1, 4]

    def test_property_bag_generate_reduces_to_typed_override(self) -> None:
        """The kwargs generate() resolves the property bag and dispatches to the typed Python override.

        Called through TalkerBase rather than the instance on purpose: a Python subclass's own
        `generate` shadows the binding, so only the base entry point exercises the C++ AnyMap
        overload and the trampoline reduction behind it.
        """
        talker = RecordingTalker()

        ov_genai.TalkerBase.generate(talker, ov_genai.VLMDecodedResults(), return_audio=False)

        assert talker.generate_calls == 1, "the property-bag overload must reach the typed Python override"
        assert talker.last_return_audio is False, "return_audio must survive the AnyMap round-trip"

    def test_property_bag_generate_falls_back_to_stored_config(self) -> None:
        """Fields omitted from the kwargs come from get_speech_config(), not from a default-constructed one."""
        talker = RecordingTalker()
        stored = ov_genai.OmniTalkerSpeechConfig()
        stored.return_audio = False
        talker.set_speech_config(stored)

        ov_genai.TalkerBase.generate(talker, ov_genai.VLMDecodedResults(), rng_seed=5)

        assert talker.last_return_audio is False, "an omitted field must fall back to the backend's stored config"

    def test_property_bag_generate_accepts_speech_streamer(self) -> None:
        """A speech_streamer callable passed as a kwarg reaches the typed override.

        The C++ AnyMap overload accepts a speech_streamer property, so the kwargs form has to as
        well or the two APIs diverge on the one argument that is specific to the talker.
        """
        talker = RecordingTalker()

        ov_genai.TalkerBase.generate(
            talker, ov_genai.VLMDecodedResults(), return_audio=False, speech_streamer=lambda chunk: None
        )

        assert talker.generate_calls == 1, "a speech_streamer kwarg must not block the property-bag overload"
        assert talker.last_speech_streamer is not None, "the streamer must survive the AnyMap round-trip"

    def test_property_bag_generate_rejects_unknown_keys(self) -> None:
        """A typo in a kwarg must raise rather than being silently dropped."""
        talker = RecordingTalker()

        with pytest.raises(RuntimeError, match="is_omni_talker_speech_config_key"):
            ov_genai.TalkerBase.generate(talker, ov_genai.VLMDecodedResults(), bogus_key=1)

        assert talker.generate_calls == 0, "an invalid property bag must not reach the backend"


class TestCustomVLMSubclass:
    """Users can define a VLMPipelineBase child in Python and have its methods dispatched from C++."""

    def test_subclass_is_instantiable(self) -> None:
        vlm = RecordingVLM()
        assert isinstance(vlm, ov_genai.VLMPipelineBase)

    def test_capability_overrides_are_dispatched(self) -> None:
        vlm = RecordingVLM(audio_output=False, hidden_states=False)
        assert vlm.is_audio_output_enabled() is False
        assert vlm.supports_hidden_states_collection() is False

    def test_property_bag_generate_reduces_to_typed_override(self) -> None:
        """The kwargs generate() unpacks media and GenerationConfig fields into the typed override.

        Called through VLMPipelineBase for the same reason as the talker case: the subclass's own
        `generate` would otherwise shadow the binding under test.
        """
        vlm = RecordingVLM()
        media = ov.Tensor(np.zeros((1, 4, 4, 3), dtype=np.uint8))

        ov_genai.VLMPipelineBase.generate(vlm, "describe", images=[media], max_new_tokens=7)

        assert vlm.generate_calls == 1, "the property-bag overload must reach the typed Python override"
        assert vlm.last_prompt == "describe"
        assert len(vlm.last_images) == 1, "images must survive the AnyMap round-trip"
        assert vlm.last_max_new_tokens == 7, "a bare GenerationConfig field must be folded into the config"

    def test_property_bag_generate_accepts_chat_history(self) -> None:
        """The ChatHistory property-bag overload reduces the same way the prompt one does.

        max_new_tokens is what forces the AnyMap path: every media argument is also a named
        parameter of the typed binding, so a call passing only those resolves to the typed
        overload and never exercises generate(history, AnyMap) at all.
        """
        vlm = RecordingVLM()
        history = ov_genai.ChatHistory()
        history.append({"role": "user", "content": "describe"})

        ov_genai.VLMPipelineBase.generate(
            vlm, history, videos=[ov.Tensor(np.zeros((2, 4, 4, 3), dtype=np.uint8))], max_new_tokens=3
        )

        assert vlm.generate_calls == 1
        assert len(vlm.last_videos) == 1, "videos must survive the AnyMap round-trip"
        assert vlm.last_max_new_tokens == 3, "the ChatHistory AnyMap overload must fold in config fields"

    def test_property_bag_generate_rejects_audio_streamer(self) -> None:
        """audio_streamer has no typed generate() parameter to land in, so it must be rejected loudly.

        It is plumbing for the built-in Qwen3-Omni speech path; forwarding it to a Python subclass
        would silently drop the caller's streamer.
        """
        vlm = RecordingVLM()

        with pytest.raises(RuntimeError, match="audio_streamer"):
            ov_genai.VLMPipelineBase.generate(vlm, "describe", audio_streamer=lambda chunk: None)

        assert vlm.generate_calls == 0, "an unforwardable property must not reach the backend"


@dataclass(frozen=True)
class InjectedOmni:
    """The DI pipeline together with the mocks it was built from, so tests can read the recorded calls."""

    pipeline: ov_genai.OmniPipeline
    vlm: RecordingVLM
    talker: RecordingTalker


@pytest.fixture
def injected_omni() -> InjectedOmni:
    """Function-scoped on purpose: the mocks carry per-test call counters that must start at zero."""
    vlm, talker = RecordingVLM(), RecordingTalker()
    return InjectedOmni(ov_genai.OmniPipeline(vlm, talker), vlm, talker)


def _talker_speech_config(
    return_audio: bool,
    *,
    speaker: str | ov.Tensor | None = None,
    rng_seed: int | None = None,
    max_new_tokens: int | None = None,
) -> ov_genai.OmniTalkerSpeechConfig:
    """Keyword-only extras default to unset, so existing call sites keep the C++ defaults verbatim."""
    config = ov_genai.OmniTalkerSpeechConfig()
    config.return_audio = return_audio
    if speaker is not None:
        config.speaker = speaker
    if rng_seed is not None:
        config.rng_seed = rng_seed
    if max_new_tokens is not None:
        config.max_new_tokens = max_new_tokens
    return config


class TestOmniPipelineDependencyInjection:
    """OmniPipeline(vlm, talker) accepts user-defined children and orchestrates them via C++."""

    def test_construct_from_python_children(self, injected_omni: InjectedOmni) -> None:
        """The DI constructor accepts a Python-defined VLM and Talker and hands them back verbatim."""
        assert injected_omni.pipeline.get_vlm() is injected_omni.vlm, (
            "get_vlm() must return the exact injected instance"
        )
        assert injected_omni.pipeline.get_talker() is injected_omni.talker, (
            "get_talker() must return the exact injected instance"
        )

    def test_speech_path_invokes_both_stages(self, injected_omni: InjectedOmni) -> None:
        """With return_audio=True, generate() drives the VLM then the talker via virtual dispatch."""
        result = injected_omni.pipeline.generate(
            "describe this", talker_speech_config=_talker_speech_config(return_audio=True)
        )

        assert isinstance(result, ov_genai.OmniDecodedResults)
        assert injected_omni.vlm.generate_calls == 1, "the injected VLM must be driven exactly once"
        assert injected_omni.talker.generate_calls == 1, "the injected talker must be driven exactly once"
        assert injected_omni.vlm.last_prompt == "describe this", "the prompt must reach the Python VLM unchanged"
        assert injected_omni.talker.last_return_audio is True

    def test_text_only_path_skips_talker(self, injected_omni: InjectedOmni) -> None:
        """With return_audio=False, the talker must not be invoked at all."""
        injected_omni.pipeline.generate("just text", talker_speech_config=_talker_speech_config(return_audio=False))

        assert injected_omni.vlm.generate_calls == 1
        assert injected_omni.talker.generate_calls == 0, "text-only generation must short-circuit the talker"

    def test_media_arguments_do_not_leak_between_calls(self, injected_omni: InjectedOmni) -> None:
        """Media passed to one generate() call must not reappear in the next call that omits it.

        The binding declares images/videos/audios/videos_metadata with empty-list defaults, which
        pybind11 materializes once per overload; a call that omits them must still see an empty
        sequence rather than whatever the previous call supplied.
        """
        media = ov.Tensor(np.zeros((1, 4, 4, 3), dtype=np.uint8))
        text_only = _talker_speech_config(return_audio=False)
        injected_omni.pipeline.generate(
            "with media",
            images=[media],
            videos=[media],
            videos_metadata=[ov_genai.VideoMetadata()],
            audios=[media],
            talker_speech_config=text_only,
        )
        assert len(injected_omni.vlm.last_images) == 1, "the first call must reach the VLM with its media"

        injected_omni.pipeline.generate("without media", talker_speech_config=text_only)

        assert injected_omni.vlm.generate_calls == 2
        assert len(injected_omni.vlm.last_images) == 0, "images leaked from the previous generate() call"
        assert len(injected_omni.vlm.last_videos) == 0, "videos leaked from the previous generate() call"
        assert len(injected_omni.vlm.last_audios) == 0, "audios leaked from the previous generate() call"
        assert len(injected_omni.vlm.last_videos_metadata) == 0, "videos_metadata leaked from the previous call"

    def test_rejects_model_without_audio_output(self) -> None:
        """The constructor must reject a VLM whose is_audio_output_enabled() reports False."""
        vlm = RecordingVLM(audio_output=False)
        with pytest.raises(RuntimeError, match="is_audio_output_enabled"):
            ov_genai.OmniPipeline(vlm, RecordingTalker())

    def test_rejects_backend_without_hidden_states(self) -> None:
        """The constructor must reject a VLM that cannot collect hidden states the talker needs."""
        vlm = RecordingVLM(hidden_states=False)
        with pytest.raises(RuntimeError, match="supports_hidden_states_collection"):
            ov_genai.OmniPipeline(vlm, RecordingTalker())


def _export_tiny_omni_model(target_dir: Path) -> None:
    """Export the tiny Qwen3-Omni checkpoint to OpenVINO IR under ``target_dir``."""
    model_cached = snapshot_download(OMNI_MODEL_ID)  # required to avoid HF rate limits
    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    processor = retry_request(
        lambda: transformers.AutoProcessor.from_pretrained(
            model_cached,
            trust_remote_code=True,
            **align_with_optimum_cli,
        )
    )
    model = retry_request(
        lambda: OVModelForVisualCausalLM.from_pretrained(
            model_cached, compile=False, device="CPU", export=True, load_in_8bit=False
        )
    )

    tokenizer = processor.tokenizer
    tokenizer.save_pretrained(target_dir)
    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, target_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, target_dir / "openvino_detokenizer.xml")

    processor.save_pretrained(target_dir)
    model.save_pretrained(target_dir)


@pytest.fixture(scope="module")
def omni_model_path() -> Path:
    """Path to an exported tiny Qwen3-Omni model, or skip when the pinned deps cannot produce one.

    Two distinct gaps are reported separately so a future failure points at the right dependency: the
    export itself raising, and the export succeeding without any talker submodels. Only the known
    transformers 5.0.x config bug is turned into a skip — any other export failure propagates so a
    real regression cannot hide behind a skipped test.
    """
    model_dir = get_ov_cache_converted_models_dir() / OMNI_MODEL_ID.replace("/", "_")
    manager = AtomicDownloadManager(model_dir)

    if not manager.is_complete() and not (model_dir / "openvino_language_model.xml").exists():
        try:
            manager.execute(_export_tiny_omni_model)
        except AttributeError as error:
            # Transformers 5.0 reads an uninitialized use_sliding_window in this config; 5.1 fixed it.
            message = str(error)
            if not (
                is_transformers_version(">=", "5.0")
                and is_transformers_version("<", "5.1")
                and "Qwen3OmniMoeTalkerCodePredictorConfig" in message
                and "use_sliding_window" in message
            ):
                raise
            logger.info("Tiny Qwen3-Omni export hit the known transformers 5.0.x config bug: %s", message)
            pytest.skip(f"Cannot export {OMNI_MODEL_ID} with the pinned dependencies: AttributeError: {message}")

    missing = [name for name in TALKER_ARTIFACTS if not (model_dir / name).exists()]
    if missing:
        pytest.skip(
            f"{OMNI_MODEL_ID} exported without the talker stage (missing {', '.join(missing)}); "
            "the installed optimum-intel exported no talker stage; huggingface/optimum-intel#1700 is what "
            "added it, so a revision predating that one cannot produce these files."
        )

    return model_dir


@pytest.fixture(scope="module")
def omni_pipe(omni_model_path: Path) -> ov_genai.OmniPipeline:
    """Pipeline built once per module — loading the model per test dominates the suite runtime.

    Reused across the real-model tests the way test_vlm_pipeline.py reuses ov_pipe_model. Safe to
    share: none of these tests enters chat mode or mutates the stored configs, so no state carries
    between them.
    """
    return ov_genai.OmniPipeline(omni_model_path, "CPU")


def _text_config(max_new_tokens: int = 10) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    return config


def _sampling_text_config(rng_seed: int, max_new_tokens: int = 20) -> ov_genai.GenerationConfig:
    """Multinomial counterpart of _text_config, seeded so one run reproduces itself.

    The high temperature and ignore_eos come from test_llm_pipeline.py's rng_seed pair: they keep
    different seeds apart instead of collapsing them onto the same near-greedy sequence.
    """
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = True
    config.temperature = 1.3
    config.ignore_eos = True
    config.rng_seed = rng_seed
    return config


@pytest.fixture(scope="module")
def omni_image() -> ov.Tensor:
    """Deterministic RGB image as [H, W, 3] uint8 — the layout the image preprocessor expects."""
    ramp = np.linspace(0, 255, MEDIA_EDGE, dtype=np.uint8)
    frame = np.empty((MEDIA_EDGE, MEDIA_EDGE, 3), dtype=np.uint8)
    frame[..., 0] = ramp[None, :]
    frame[..., 1] = ramp[::-1][:, None]
    frame[..., 2] = 128
    return ov.Tensor(frame)


@pytest.fixture(scope="module")
def omni_video() -> ov.Tensor:
    """Deterministic video as [N, H, W, 3] uint8, one horizontal shift per frame so frames differ."""
    base = np.tile(np.linspace(0, 255, MEDIA_EDGE, dtype=np.uint8), (MEDIA_EDGE, 1))
    frames = [np.repeat(np.roll(base, 8 * index, axis=1)[..., None], 3, axis=2) for index in range(VIDEO_FRAMES)]
    return ov.Tensor(np.stack(frames))


@pytest.fixture(scope="module")
def omni_audio() -> ov.Tensor:
    """Deterministic mono PCM as a 1-D float32 tensor at 16 kHz — what the audio encoder validates."""
    seconds = np.arange(AUDIO_SAMPLES, dtype=np.float32) / AUDIO_SAMPLE_RATE
    return ov.Tensor(np.sin(2.0 * np.pi * 440.0 * seconds).astype(np.float32))


def _single_waveform(result: ov_genai.OmniDecodedResults) -> np.ndarray:
    """Flatten the one waveform a speech-enabled result must carry, rejecting empty/non-finite audio."""
    waveforms = result.speech_result.waveforms
    assert len(waveforms) == 1, f"return_audio=True must produce exactly one waveform, got {len(waveforms)}"
    waveform = np.array(waveforms[0].data, dtype=np.float32).reshape(-1)
    assert waveform.size > 0, "waveform must not be empty"
    assert np.isfinite(waveform).all(), "waveform must contain no NaN or Inf"
    return waveform


def _waveforms_differ(left: np.ndarray, right: np.ndarray) -> bool:
    """Whether two waveforms are different audio, by length or by samples."""
    return left.shape != right.shape or not np.allclose(left, right)


def _generate_speech(pipe: ov_genai.OmniPipeline, speaker: str | ov.Tensor) -> np.ndarray:
    """Generate speech for the fixed prompt/seed/token budget and return the waveform."""
    result = pipe.generate(
        SPEECH_PROMPT,
        text_config=_text_config(),
        talker_speech_config=_talker_speech_config(
            return_audio=True,
            speaker=speaker,
            rng_seed=TALKER_RNG_SEED,
            max_new_tokens=TALKER_MAX_NEW_TOKENS,
        ),
    )
    return _single_waveform(result)


@pytest.fixture(scope="module")
def real_omni_pipe() -> ov_genai.OmniPipeline:
    """Pipeline over a full Qwen3-Omni export, for tests the tiny checkpoint cannot host.

    Pointed at by OMNI_REAL_MODEL_PATH rather than a models/ list file: those hold HF ids consumed by
    the LLM suites' indirect fixture, which cannot express a pre-exported VLM+talker directory. Tests
    using this are also marked real_models, so pytest.ini's default addopts deselect them; the skip
    below only matters when somebody opts in without setting the variable.
    """
    raw_path = os.environ.get(OMNI_REAL_MODEL_ENV)
    if not raw_path:
        pytest.skip(f"set {OMNI_REAL_MODEL_ENV} to a full Qwen3-Omni OpenVINO export to run this")

    model_dir = Path(raw_path)
    if not (model_dir / "openvino_talker_model.xml").exists():
        pytest.skip(f"{OMNI_REAL_MODEL_ENV}={model_dir} has no talker export")

    return ov_genai.OmniPipeline(model_dir, "CPU")


class _CapturingStreamer(ov_genai.StreamerBase):
    """Records the raw token ids GenAI generates.

    Decoded text is useless as a comparison target on this checkpoint: its tokenizer covers ids
    0-769 while the model head emits the full Qwen vocab, so every generated id falls outside the
    tokenizer and renders as ''. Ids sidestep the broken tokenizer entirely.
    """

    def __init__(self) -> None:
        super().__init__()
        self.token_ids: list[int] = []

    def write(self, token: int | Sequence[int]) -> ov_genai.StreamingStatus:
        if isinstance(token, (list, tuple)):
            self.token_ids.extend(int(one) for one in token)
        else:
            self.token_ids.append(int(token))
        return ov_genai.StreamingStatus.RUNNING

    def end(self) -> None:
        pass


def _genai_generated_ids(pipe: ov_genai.OmniPipeline, prompt: str, image: np.ndarray | None) -> list[int]:
    """Token ids the thinker generates, harvested through a streamer."""
    streamer = _CapturingStreamer()
    pipe.generate(
        prompt,
        images=[ov.Tensor(image)] if image is not None else [],
        text_config=_text_config(max_new_tokens=OPTIMUM_COMPARE_TOKENS),
        talker_speech_config=_talker_speech_config(return_audio=False),
        streamer=streamer,
    )
    return streamer.token_ids


@dataclass(frozen=True)
class OptimumReference:
    """optimum-intel model plus the processor that builds its inputs."""

    model: OVModelForVisualCausalLM
    processor: transformers.ProcessorMixin


def _optimum_generated_ids(reference: OptimumReference, prompt: str, image: np.ndarray | None) -> list[int]:
    """Token ids optimum-intel generates for the same prompt, sliced off its echoed input."""
    model = reference.model
    inputs = model.preprocess_inputs(
        text=prompt, image=image, processor=reference.processor, tokenizer=None, config=model.config
    )
    output_ids = model.generate(**inputs, max_new_tokens=OPTIMUM_COMPARE_TOKENS, do_sample=False)
    input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
    return [out[len(inp) :].tolist() for inp, out in zip(input_ids, output_ids)][0]


@pytest.fixture(scope="module")
def optimum_reference(omni_model_path: Path) -> OptimumReference:
    """optimum-intel over the same export, as a reference implementation.

    Module-scoped: this is a second full model load on top of omni_pipe.
    """
    return OptimumReference(
        model=OVModelForVisualCausalLM.from_pretrained(omni_model_path, export=False),
        processor=transformers.AutoProcessor.from_pretrained(omni_model_path, trust_remote_code=True),
    )


def _load_models_map(model_dir: Path) -> dict[str, tuple[str, ov.Tensor]]:
    """Read every exported IR in ``model_dir`` into the ModelsMap form the blob ctors accept.

    Key convention is the one samples/python/visual_language_chat/encrypted_model_vlm.py uses:
    ``openvino_<name>_model.xml`` -> ``<name>``. A single glob covers both stages — the VLM picks
    up language/text_embeddings/vision_embeddings*, the talker picks up
    talker*/code_predictor/code2wav, and each ignores the keys it does not know.
    """
    models_map: dict[str, tuple[str, ov.Tensor]] = {}
    for xml_path in sorted(model_dir.glob("*.xml")):
        weights_path = xml_path.with_suffix(".bin")
        if "tokenizer" in xml_path.name or not weights_path.exists():
            continue
        name = xml_path.stem.removeprefix("openvino_").removesuffix("_model")
        models_map[name] = (
            xml_path.read_text(encoding="utf-8"),
            ov.Tensor(np.fromfile(weights_path, dtype=np.uint8)),
        )
    return models_map


@pytest.fixture(scope="module")
def omni_pipe_from_models_map(omni_model_path: Path) -> ov_genai.OmniPipeline:
    """The same checkpoint as omni_pipe, but with both stages built from in-memory IRs and injected.

    Module-scoped for the same reason as omni_pipe: this is a second full model load. The tokenizer
    is loaded separately rather than taken from omni_pipe.get_vlm(): get_tokenizer() hands back a
    new wrapper around the *same* C++ impl, so sharing it would couple the two pipelines that this
    test is supposed to compare independently. ATTENTION_BACKEND is requested explicitly because the
    path ctor reaches the continuous-batching backend by default — naming it here turns a failure to
    get it into a raise instead of a silent SDPA fallback that OmniPipeline would then reject with
    an unrelated-looking "speech output requires the continuous-batching backend".
    """
    models_map = _load_models_map(omni_model_path)
    vlm = ov_genai.VLMPipeline(
        models_map,
        ov_genai.Tokenizer(omni_model_path),
        omni_model_path,
        "CPU",
        ATTENTION_BACKEND="PA",
    )
    all_on_cpu: dict[str, str] = {}
    talker = ov_genai.Talker(models_map, ov_genai.OmniTalkerSpeechConfig(), omni_model_path, all_on_cpu)
    return ov_genai.OmniPipeline(vlm, talker)


class TestOmniPipelineRealModel:
    """OmniPipeline driven by an exported tiny Qwen3-Omni checkpoint.

    Covers what the mock-based DI tests cannot: the path-based constructor, the real
    thinker/talker composition, and the speaker APIs backed by actual model data.
    """

    def test_constructs_from_path(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """The path-based ctor builds both stages from a single model directory."""
        assert omni_pipe.get_vlm() is not None, "path ctor must build a VLM stage"
        assert omni_pipe.get_talker() is not None, "path ctor must build a talker stage"

    def test_models_map_ctor_matches_path_ctor_text(
        self, omni_pipe: ov_genai.OmniPipeline, omni_pipe_from_models_map: ov_genai.OmniPipeline
    ) -> None:
        """Building both stages from in-memory IRs decodes the same text as building from a path.

        Greedy decode keeps the two comparable token for token: handing the IRs over as strings and
        weight tensors changes where they are read from, never the graph that gets compiled.
        """
        text_config = _text_config()
        text_only = _talker_speech_config(return_audio=False)

        from_path = omni_pipe.generate("Describe this.", text_config=text_config, talker_speech_config=text_only)
        from_map = omni_pipe_from_models_map.generate(
            "Describe this.", text_config=text_config, talker_speech_config=text_only
        )

        assert from_map.texts == from_path.texts, (
            f"ModelsMap-built pipeline decoded {from_map.texts!r}, path-built decoded {from_path.texts!r}"
        )

        # This checkpoint decodes to '' whatever it is fed, so comparing texts alone cannot fail.
        # Seeded sampling gives a cumulative score that does discriminate between graphs.
        seeded = _sampling_text_config(rng_seed=42)
        path_scores = omni_pipe.generate("Describe this.", text_config=seeded, talker_speech_config=text_only).scores
        map_scores = omni_pipe_from_models_map.generate(
            "Describe this.", text_config=seeded, talker_speech_config=text_only
        ).scores

        assert map_scores == pytest.approx(path_scores), (
            f"seeded sampling diverged: ModelsMap-built scored {map_scores}, path-built scored {path_scores}, "
            "so the two constructions did not compile the same graph"
        )

    @pytest.mark.xfail(reason=NO_WAVEFORM_XFAIL_REASON, strict=True)
    def test_models_map_ctor_matches_path_ctor_speech(
        self, omni_pipe: ov_genai.OmniPipeline, omni_pipe_from_models_map: ov_genai.OmniPipeline
    ) -> None:
        """The ModelsMap-built talker synthesizes the same waveform as the path-built one.

        The talker samples with top-k, but it re-seeds its RNG from talker_speech_config.rng_seed on
        every call (default 0 on both sides here), so identical thinker hidden states yield an
        identical codec token stream and therefore an identical sample count. The samples themselves
        are compared with a tolerance rather than bit-exactly: the two stacks are independently
        compiled copies of the same IRs, so only the last float bits may drift.
        """
        text_config = _text_config()
        speech_on = _talker_speech_config(return_audio=True)

        from_path = omni_pipe.generate("Describe this.", text_config=text_config, talker_speech_config=speech_on)
        from_map = omni_pipe_from_models_map.generate(
            "Describe this.", text_config=text_config, talker_speech_config=speech_on
        )

        assert from_map.texts == from_path.texts, "the thinker stages must agree before waveforms can be compared"

        path_waveform = _single_waveform(from_path)
        map_waveform = _single_waveform(from_map)

        assert map_waveform.shape == path_waveform.shape, (
            f"waveform lengths diverged ({map_waveform.size} vs {path_waveform.size} samples), so the two talkers "
            "sampled different codec tokens"
        )
        assert np.allclose(map_waveform, path_waveform, rtol=0.0, atol=1e-4), (
            f"waveforms differ by up to {np.abs(map_waveform - path_waveform).max():.3g}, which is beyond "
            "independent-compilation float noise"
        )

    def test_generate_text_only(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """With return_audio=False the pipeline decodes text and emits no waveform."""
        result = omni_pipe.generate(
            "Describe this.",
            text_config=_text_config(),
            talker_speech_config=_talker_speech_config(return_audio=False),
        )

        assert isinstance(result, ov_genai.OmniDecodedResults)
        assert len(result.texts) == 1, "greedy decode must produce exactly one sequence"
        assert result.speech_result.waveforms == [], "return_audio=False must not produce waveforms"

    def test_max_new_tokens_changes_generated_length(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """Raising max_new_tokens makes the thinker emit strictly more tokens.

        ignore_eos is belt-and-braces: this checkpoint never emits EOS within 24 tokens, so the
        counts are the same without it, but a checkpoint that did would make both caps stop early
        at the same length and the comparison would say nothing.
        """
        short_config = _text_config(max_new_tokens=4)
        short_config.ignore_eos = True
        long_config = _text_config(max_new_tokens=24)
        long_config.ignore_eos = True
        text_only = _talker_speech_config(return_audio=False)

        short_result = omni_pipe.generate("Describe this.", text_config=short_config, talker_speech_config=text_only)
        long_result = omni_pipe.generate("Describe this.", text_config=long_config, talker_speech_config=text_only)

        short_tokens = short_result.perf_metrics.get_num_generated_tokens()
        long_tokens = long_result.perf_metrics.get_num_generated_tokens()

        assert 0 < short_tokens <= 4, f"max_new_tokens=4 must cap the decode, got {short_tokens} tokens"
        assert 0 < long_tokens <= 24, f"max_new_tokens=24 must cap the decode, got {long_tokens} tokens"
        assert short_tokens < long_tokens, (
            f"raising max_new_tokens from 4 to 24 must lengthen the decode, got {short_tokens} then {long_tokens}"
        )

    def test_rng_seed_steers_sampling(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """One rng_seed reproduces its own result, and the seeds do not all produce the same one.

        Asserted on scores, not texts: this checkpoint decodes to '' for every seed, so a text
        comparison cannot fail. The cumulative score does discriminate, which is what makes this
        the test that rng_seed actually reaches the sampler.

        The divergence check spans four seeds and only requires that they are not all identical,
        mirroring test_llm_pipeline.py's rng_seed pair — any single pair of seeds can collide.
        """
        text_only = _talker_speech_config(return_audio=False)
        seeded = _sampling_text_config(rng_seed=42)

        first = omni_pipe.generate("Describe this.", text_config=seeded, talker_speech_config=text_only)
        second = omni_pipe.generate("Describe this.", text_config=seeded, talker_speech_config=text_only)
        assert first.scores == pytest.approx(second.scores), (
            f"rng_seed=42 must reproduce its own scores, got {first.scores} then {second.scores}"
        )

        rng_seeds = (42, 123, 777, 2024)
        sampled = {
            tuple(
                omni_pipe.generate(
                    "Describe this.",
                    text_config=_sampling_text_config(rng_seed=seed),
                    talker_speech_config=text_only,
                ).scores
            )
            for seed in rng_seeds
        }
        assert len(sampled) > 1, (
            f"sampling with different rng_seeds {rng_seeds} produced the same scores for every seed"
        )

    @pytest.mark.xfail(reason=NO_WAVEFORM_XFAIL_REASON, strict=True)
    def test_generate_with_speech(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """With return_audio=True the talker produces a finite, non-empty waveform."""
        result = omni_pipe.generate(
            "Describe this.",
            text_config=_text_config(),
            talker_speech_config=_talker_speech_config(return_audio=True),
        )

        _single_waveform(result)

    def test_matches_optimum_text(self, omni_pipe: ov_genai.OmniPipeline, optimum_reference: OptimumReference) -> None:
        """Greedy decode must produce the same token ids as optimum-intel for a text-only prompt.

        Compared on ids rather than decoded text because this checkpoint's tokenizer only covers ids
        0-769 while the model head emits the full Qwen vocab, so both stacks decode to '' and a text
        comparison could not fail. Ids are what the two implementations actually disagree about.
        """
        prompt = "Describe."

        optimum_ids = _optimum_generated_ids(optimum_reference, prompt, None)
        genai_ids = _genai_generated_ids(omni_pipe, prompt, None)

        assert genai_ids, "the streamer must observe generated tokens"
        assert genai_ids == optimum_ids, (
            f"GenAI generated {genai_ids}, optimum generated {optimum_ids} for the same greedy prompt"
        )

    @pytest.mark.xfail(reason=OPTIMUM_IMAGE_XFAIL_REASON, strict=True)
    def test_matches_optimum_image(
        self, omni_pipe: ov_genai.OmniPipeline, optimum_reference: OptimumReference, omni_image: ov.Tensor
    ) -> None:
        """The same comparison with an image attached.

        Both stacks agree on the input length, so any divergence here is in the image embeddings
        rather than in how the prompt is built.
        """
        prompt = "Describe this image."
        image = np.array(omni_image.data, dtype=np.uint8).reshape(omni_image.shape)

        optimum_ids = _optimum_generated_ids(optimum_reference, prompt, image)
        genai_ids = _genai_generated_ids(omni_pipe, prompt, image)

        assert genai_ids == optimum_ids, (
            f"GenAI generated {genai_ids}, optimum generated {optimum_ids} for the same image prompt"
        )

    def test_speech_generation_rejects_unmatched_role_tokens(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """A checkpoint whose role token ids never appear in its token stream must raise, not go quiet.

        Speech used to warn and hand back an empty waveform list here, so a misconfigured checkpoint
        looked like a model that simply had nothing to say. The caller asked for audio; failing to
        segment the conversation is a configuration error and has to surface as one.

        This is the inverse of test_generate_with_speech: that one asserts the audio a healthy
        checkpoint owes us and xfails here, this one pins the diagnosis we give for a broken one.
        """
        with pytest.raises(RuntimeError, match="im_start_token_id") as excinfo:
            omni_pipe.generate(
                "Describe this.",
                text_config=_text_config(),
                talker_speech_config=_talker_speech_config(return_audio=True),
            )

        message = str(excinfo.value)
        assert "no talker input" in message, f"the error must say what failed, got: {message}"
        assert "config.json" in message, f"the error must point at the fix, got: {message}"

    def test_generate_from_chat_history(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """The ChatHistory overload accepts structured turns and leaves the caller's history intact."""
        history = ov_genai.ChatHistory()
        history.append({"role": "user", "content": "Describe this."})
        messages_before = history.get_messages()

        result = omni_pipe.generate(
            history,
            text_config=_text_config(),
            talker_speech_config=_talker_speech_config(return_audio=False),
        )

        assert len(result.texts) == 1
        assert history.get_messages() == messages_before, "ChatHistory messages should not be mutated after generate."

    @pytest.mark.parametrize(
        "modality",
        [
            pytest.param("image", id="image"),
            pytest.param("video", id="video", marks=pytest.mark.xfail(reason=VIDEO_INPUT_XFAIL_REASON, strict=True)),
            pytest.param("audio", id="audio", marks=pytest.mark.xfail(reason=AUDIO_INPUT_XFAIL_REASON, strict=True)),
        ],
    )
    def test_generate_from_chat_history_all_modalities(
        self,
        omni_pipe: ov_genai.OmniPipeline,
        omni_image: ov.Tensor,
        omni_video: ov.Tensor,
        omni_audio: ov.Tensor,
        modality: str,
    ) -> None:
        """Media attached to a ChatHistory turn reach the thinker instead of being silently dropped.

        Each modality expands the prompt with its own placeholder tokens, so the input token count
        must grow once media are attached — a modality that never reached preprocessing would leave
        it untouched. The count is the whole assertion for the vision modalities:
        vision_encoding_durations is appended unconditionally, so it is non-empty even for a
        text-only call and proves nothing. audio_encoding_durations is only populated when audio
        is actually encoded, so it is worth asserting on that branch.

        One modality per case so a failure names the modality that broke rather than the whole set.
        """

        def fresh_history() -> ov_genai.ChatHistory:
            # Reusing a ChatHistory whose first call had no media makes a later media call drop it.
            history = ov_genai.ChatHistory()
            history.append({"role": "user", "content": "Describe the attached media."})
            return history

        text_config = _text_config()
        talker_config = _talker_speech_config(return_audio=False)
        media: dict[str, list[ov.Tensor]] = {
            "images": [omni_image] if modality == "image" else [],
            "videos": [omni_video] if modality == "video" else [],
            "audios": [omni_audio] if modality == "audio" else [],
        }

        text_only = omni_pipe.generate(fresh_history(), text_config=text_config, talker_speech_config=talker_config)
        multimodal = omni_pipe.generate(
            fresh_history(), **media, text_config=text_config, talker_speech_config=talker_config
        )

        assert len(multimodal.texts) == 1, "greedy decode must produce exactly one sequence"
        assert multimodal.perf_metrics.get_num_input_tokens() > text_only.perf_metrics.get_num_input_tokens(), (
            f"{modality} placeholders must expand the prompt; an unchanged input token count means "
            "the media never reached preprocessing"
        )
        if modality == "audio":
            assert multimodal.perf_metrics.vlm_raw_metrics.audio_encoding_durations, (
                "the audio encoder must run when audios are passed"
            )

    def test_speaker_apis(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """list_speakers() reports the checkpoint's voices and each resolves to an embedding."""
        speakers = omni_pipe.get_talker().list_speakers()

        assert speakers, "the checkpoint declares speakers, so list_speakers() must not be empty"
        embedding = omni_pipe.get_talker().get_speaker_embedding(speakers[0])
        assert embedding.get_size() > 0, f"speaker {speakers[0]!r} must resolve to a non-empty embedding"

    @pytest.mark.real_models
    def test_altering_speaker_embedding_changes_speech(self, real_omni_pipe: ov_genai.OmniPipeline) -> None:
        """A modified speaker embedding yields different speech; an unmodified one reproduces it exactly.

        OmniTalkerSpeechConfig.speaker is a variant of name-or-tensor, and the tensor branch bypasses
        the name lookup and feeds the embedding straight into the talker prefix. The talker has no
        greedy path, so all three runs pin the same rng_seed. The control run proves the pipeline is
        reproducible under that seed, which is what lets the third run attribute its difference to the
        embedding rather than to sampling noise.

        Needs a full checkpoint: the tiny one cannot synthesize at all, so there is no waveform to
        compare. See NO_WAVEFORM_XFAIL_REASON.
        """
        omni_pipe = real_omni_pipe
        talker = omni_pipe.get_talker()
        speakers = talker.list_speakers()
        assert speakers, "the checkpoint declares speakers, so list_speakers() must not be empty"

        embedding = talker.get_speaker_embedding(speakers[0])
        altered_data = (np.array(embedding.data, dtype=np.float32) * SPEAKER_EMBEDDING_SCALE).astype(np.float32)
        altered = ov.Tensor(altered_data)

        baseline = _generate_speech(omni_pipe, speaker=embedding)
        control = _generate_speech(omni_pipe, speaker=embedding)
        perturbed = _generate_speech(omni_pipe, speaker=altered)

        assert np.array_equal(baseline, control), (
            "the same speaker embedding and rng_seed must reproduce the waveform sample for sample; "
            "without that the comparison below cannot attribute a difference to the embedding"
        )
        assert _waveforms_differ(perturbed, baseline), (
            f"scaling the speaker embedding by {SPEAKER_EMBEDDING_SCALE} left the waveform unchanged "
            f"({perturbed.size} samples), so the embedding is not reaching the talker"
        )

    @pytest.mark.real_models
    def test_distinct_speakers_produce_distinct_speech(self, real_omni_pipe: ov_genai.OmniPipeline) -> None:
        """Two different named speakers must not render the same audio for the same text and seed.

        Needs a full checkpoint, for the same reason as the embedding test above.
        """
        omni_pipe = real_omni_pipe
        speakers = omni_pipe.get_talker().list_speakers()
        if len(speakers) < 2:
            pytest.skip(f"the checkpoint declares a single speaker ({speakers}); nothing to compare")

        first = _generate_speech(omni_pipe, speaker=speakers[0])
        second = _generate_speech(omni_pipe, speaker=speakers[1])

        assert _waveforms_differ(first, second), (
            f"speakers {speakers[0]!r} and {speakers[1]!r} rendered identical audio ({first.size} samples); "
            "the speaker selection is not reaching the talker"
        )
