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

Three groups, following the other pipeline suites:

1. Model-free tests — imports, config defaults, field round-trips, and dependency
   injection of user-defined VLMPipelineBase / TalkerBase children. The DI tests
   drive the real C++ OmniPipeline composition path against Python-defined mocks,
   exercising virtual dispatch, the text vs speech branching, and the constructor
   capability guards end to end.

2. Real-model tests against `optimum-intel-internal-testing/tiny-random-qwen3-omni`,
   covering the path-based constructor, text-only and speech generation, the
   ChatHistory overload, generation-config sensitivity, ModelsMap/path equivalence,
   multimodal inputs, and the speaker APIs.

3. Audio tests that drive a real Qwen3-Omni export through VLMPipeline to cover
   `<ov_genai_audio_N>` placement. They are marked real_models and need $QWEN3_OMNI_OV_MODEL.

The tiny-checkpoint tier needs a newer transformers than tests/python_tests/requirements.txt
pins, so the CI matrix entries install one per job: transformers 5.0.0 reads an unset
`use_sliding_window` in `Qwen3OmniMoeTalkerCodePredictorConfig.__init__` and the export dies
with an `AttributeError`, fixed in 5.1.0. The pinned optimum-intel already carries the
talker/code2wav export from huggingface/optimum-intel#1700. `omni_model_path` skips on that
transformers gap with the reason, so running the suite against the repo-wide pins degrades to
the model-free tier instead of failing.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openvino
import openvino as ov
import openvino_tokenizers
import pytest
import transformers
from huggingface_hub import snapshot_download
from optimum.intel.openvino import OVModelForVisualCausalLM
from optimum.utils.import_utils import is_transformers_version

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
from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir
from utils.media_tags import ModalityType, get_universal_tag
from utils.network import retry_request

OMNI_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-omni"

FRAME_RESOLUTION = 64
VIDEO_FRAMES = 4
AUDIO_SAMPLE_RATE = 16000
AUDIO_SAMPLES = AUDIO_SAMPLE_RATE

OPTIMUM_COMPARE_TOKENS = 6

OPTIMUM_IMAGE_XFAIL_REASON = (
    "optimum-intel builds 1D position ids on prefill instead of the thinker's mRoPE positions, so once an "
    "image is attached its ids diverge from GenAI and HF transformers, which agree. Fixed by "
    "huggingface/optimum-intel#2042; drop this xfail once the pinned optimum-intel includes it."
)

# Real Qwen3-Omni role ids; the tiny tokenizer never produces them, so the talker finds no segments.
UNMATCHED_ROLE_TOKEN_IDS = {
    "im_start_token_id": 151644,
    "system_token_id": 8948,
    "user_token_id": 872,
    "assistant_token_id": 77091,
}


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

    def generate(
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

    def test_property_bag_generate_accepts_speech_streamer_object(self) -> None:
        """An OmniSpeechStreamerBase subclass reaches the typed override, not just a callable.

        The callable and the shared_ptr are separate alternatives of OmniSpeechStreamerVariant and
        take different branches of the AnyMap reader, so a callable-only test leaves half of it
        unexercised.
        """

        class Collector(ov_genai.OmniSpeechStreamerBase):
            def write(self, chunk: ov.Tensor) -> ov_genai.StreamingStatus:
                return ov_genai.StreamingStatus.RUNNING

            def end(self) -> None:
                pass

        talker = RecordingTalker()
        collector = Collector()

        ov_genai.TalkerBase.generate(
            talker, ov_genai.VLMDecodedResults(), return_audio=False, speech_streamer=collector
        )

        assert talker.generate_calls == 1, "a streamer object must not block the property-bag overload"
        assert talker.last_speech_streamer is collector, "the streamer object must survive the AnyMap round-trip"

    def test_property_bag_generate_accepts_whole_speech_config(self) -> None:
        """A full OmniTalkerSpeechConfig passed as a kwarg reaches the typed override.

        generate() documents talker_speech_config as a keyword and resolve_talker_properties()
        reads it back as OmniTalkerSpeechConfig, so py_object_to_any() has to convert the type.
        """
        talker = RecordingTalker()
        config = ov_genai.OmniTalkerSpeechConfig()
        config.return_audio = False

        ov_genai.TalkerBase.generate(talker, ov_genai.VLMDecodedResults(), talker_speech_config=config)

        assert talker.generate_calls == 1, "a talker_speech_config kwarg must reach the backend"
        assert talker.last_return_audio is False, "the passed config must win over the stored one"

    def test_property_bag_generate_rejects_unknown_keys(self) -> None:
        """A typo in a kwarg must raise rather than being silently dropped."""
        talker = RecordingTalker()

        with pytest.raises(RuntimeError, match="unrecognized property"):
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

    def test_property_bag_generate_rejects_speech_streamer(self) -> None:
        """speech_streamer has no typed generate() parameter to land in, so it must be rejected loudly.

        It is plumbing for the built-in Qwen3-Omni speech path; forwarding it to a Python subclass
        would silently drop the caller's streamer. kwargs_to_any_map() accepts the key, so the
        rejection has to come from unpack_config_map() rather than from the conversion failing.
        """
        vlm = RecordingVLM()

        with pytest.raises(RuntimeError, match="speech_streamer"):
            ov_genai.VLMPipelineBase.generate(vlm, "describe", speech_streamer=lambda chunk: None)

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
        with pytest.raises(RuntimeError, match="requires a Qwen3-Omni model with audio output enabled"):
            ov_genai.OmniPipeline(vlm, RecordingTalker())

    def test_rejects_backend_without_hidden_states(self) -> None:
        """The constructor must reject a VLM that cannot collect hidden states the talker needs."""
        vlm = RecordingVLM(hidden_states=False)
        with pytest.raises(RuntimeError, match="speech output requires the continuous-batching backend"):
            ov_genai.OmniPipeline(vlm, RecordingTalker())


def _export_tiny_omni_model(target_dir: Path) -> None:
    """Export the tiny Qwen3-Omni checkpoint to OpenVINO IR under ``target_dir``."""
    # Only the download needs retrying; both from_pretrained() calls below read the local path it returns.
    model_cached = retry_request(lambda: snapshot_download(OMNI_MODEL_ID))
    align_with_optimum_cli = {"padding_side": "left", "truncation_side": "left"}
    processor = transformers.AutoProcessor.from_pretrained(
        model_cached,
        trust_remote_code=True,
        **align_with_optimum_cli,
    )
    model = OVModelForVisualCausalLM.from_pretrained(
        model_cached, compile=False, device="CPU", export=True, load_in_8bit=False
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

    Only the known transformers 5.0.x config bug is turned into a skip — any other export failure
    propagates so a real regression cannot hide behind a skipped test.
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
            pytest.skip(
                f"Cannot export {OMNI_MODEL_ID}: hit the known transformers 5.0.x config bug, "
                f"fixed in 5.1.0. AttributeError: {message}"
            )

    return model_dir


@pytest.fixture(scope="module")
def omni_pipe(omni_model_path: Path) -> ov_genai.OmniPipeline:
    """Pipeline built once per module — loading the model per test dominates the suite runtime.

    Safe to share: none of these tests enters chat mode or mutates the stored configs, so no state
    carries between them.
    """
    return ov_genai.OmniPipeline(omni_model_path, "CPU")


def _text_config(max_new_tokens: int = 10) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    return config


def _sampling_text_config(rng_seed: int, max_new_tokens: int = 20) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = True
    # A high temperature and ignore_eos keep different seeds on different sequences instead of
    # collapsing them onto the same near-greedy one.
    config.temperature = 1.3
    config.ignore_eos = True
    config.rng_seed = rng_seed
    return config


@pytest.fixture(scope="module")
def omni_image() -> ov.Tensor:
    """Deterministic RGB image as [H, W, 3] uint8 — the layout the image preprocessor expects."""
    ramp = np.linspace(0, 255, FRAME_RESOLUTION, dtype=np.uint8)
    frame = np.empty((FRAME_RESOLUTION, FRAME_RESOLUTION, 3), dtype=np.uint8)
    frame[..., 0] = ramp[None, :]
    frame[..., 1] = ramp[::-1][:, None]
    frame[..., 2] = 128
    return ov.Tensor(frame)


@pytest.fixture(scope="module")
def omni_video() -> ov.Tensor:
    """Deterministic video as [N, H, W, 3] uint8, one horizontal shift per frame so frames differ."""
    base = np.tile(np.linspace(0, 255, FRAME_RESOLUTION, dtype=np.uint8), (FRAME_RESOLUTION, 1))
    frames = [np.repeat(np.roll(base, 8 * index, axis=1)[..., None], 3, axis=2) for index in range(VIDEO_FRAMES)]
    return ov.Tensor(np.stack(frames))


@pytest.fixture(scope="module")
def omni_audio() -> ov.Tensor:
    """Deterministic mono PCM as a 1-D float32 tensor at 16 kHz — what the audio encoder validates."""
    seconds = np.arange(AUDIO_SAMPLES, dtype=np.float32) / AUDIO_SAMPLE_RATE
    return ov.Tensor(np.sin(2.0 * np.pi * 440.0 * seconds).astype(np.float32))


def _extract_assert_single_waveform(result: ov_genai.OmniDecodedResults) -> np.ndarray:
    """Flatten the one waveform a speech-enabled result must carry, rejecting empty/non-finite audio."""
    waveforms = result.speech_result.waveforms
    assert len(waveforms) == 1, f"return_audio=True must produce exactly one waveform, got {len(waveforms)}"
    waveform = np.array(waveforms[0].data, dtype=np.float32).reshape(-1)
    assert waveform.size > 0, "waveform must not be empty"
    assert np.isfinite(waveform).all(), "waveform must contain no NaN or Inf"
    return waveform


class _CapturingStreamer(ov_genai.StreamerBase):
    """Records the raw token ids GenAI generates, so a mismatch points at the first diverging token."""

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


@pytest.fixture(scope="module")
def omni_pipe_with_unmatched_role_tokens(
    omni_model_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> ov_genai.OmniPipeline:
    """The exported checkpoint with only its role token ids intentionally broken in config.json.

    Copied, not symlinked: the pipeline refuses to load model files that resolve outside the model dir.
    """
    model_dir = tmp_path_factory.mktemp("omni_unmatched_role_tokens") / "model"
    shutil.copytree(omni_model_path, model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    config.update(UNMATCHED_ROLE_TOKEN_IDS)
    (model_dir / "config.json").write_text(json.dumps(config))
    return ov_genai.OmniPipeline(model_dir, "CPU")


@pytest.mark.skipif(
    sys.platform == "darwin" and platform.machine() == "arm64",
    reason="CVS-194249: OmniPipeline requires the continuous-batching backend, and PagedAttention is "
    "unavailable on macOS arm64.",
)
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

        # Greedy text can agree while the sampling path differs; seeded sampling also compares the scores.
        seeded = _sampling_text_config(rng_seed=42)
        path_scores = omni_pipe.generate("Describe this.", text_config=seeded, talker_speech_config=text_only).scores
        map_scores = omni_pipe_from_models_map.generate(
            "Describe this.", text_config=seeded, talker_speech_config=text_only
        ).scores

        assert map_scores == path_scores, (
            f"seeded sampling diverged: ModelsMap-built scored {map_scores}, path-built scored {path_scores}, "
            "so the two constructions did not compile the same graph"
        )

    def test_models_map_ctor_matches_path_ctor_speech(
        self, omni_pipe: ov_genai.OmniPipeline, omni_pipe_from_models_map: ov_genai.OmniPipeline
    ) -> None:
        """The ModelsMap-built talker synthesizes the same waveform as the path-built one.

        The talker samples with top-k, but it re-seeds its RNG from talker_speech_config.rng_seed on
        every call (default 0 on both sides here), so identical thinker hidden states yield an
        identical codec token stream. Both stacks run the same IRs on the same device, so the
        waveforms must match sample for sample, not just closely.
        """
        text_config = _text_config()
        speech_on = _talker_speech_config(return_audio=True)

        from_path = omni_pipe.generate("Describe this.", text_config=text_config, talker_speech_config=speech_on)
        from_map = omni_pipe_from_models_map.generate(
            "Describe this.", text_config=text_config, talker_speech_config=speech_on
        )

        assert from_map.texts == from_path.texts, "the thinker stages must agree before waveforms can be compared"

        path_waveform = _extract_assert_single_waveform(from_path)
        map_waveform = _extract_assert_single_waveform(from_map)

        assert map_waveform.shape == path_waveform.shape, (
            f"waveform lengths diverged ({map_waveform.size} vs {path_waveform.size} samples), so the two talkers "
            "sampled different codec tokens"
        )
        assert np.array_equal(map_waveform, path_waveform), (
            f"waveforms differ by up to {np.abs(map_waveform - path_waveform).max():.3g}; the ModelsMap and "
            "path talkers run the same IRs from the same seed, so they must agree sample for sample"
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
        """max_new_tokens sets the generated length exactly.

        ignore_eos is what makes the count exact rather than an upper bound: without it a checkpoint
        that emitted EOS early would stop short of the cap.
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

        assert short_tokens == 4, f"max_new_tokens=4 with ignore_eos must generate 4 tokens, got {short_tokens}"
        assert long_tokens == 24, f"max_new_tokens=24 with ignore_eos must generate 24 tokens, got {long_tokens}"

    def test_rng_seed_steers_sampling(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """One rng_seed reproduces its own result, and each seed produces a different one.

        Asserted on scores, not texts: on random weights two seeds can still decode to the same short
        text, while the cumulative score shows whether rng_seed actually reaches the sampler.
        """
        text_only = _talker_speech_config(return_audio=False)
        seeded = _sampling_text_config(rng_seed=42)

        first = omni_pipe.generate("Describe this.", text_config=seeded, talker_speech_config=text_only)
        second = omni_pipe.generate("Describe this.", text_config=seeded, talker_speech_config=text_only)
        assert first.scores == second.scores, (
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
        assert len(sampled) == len(rng_seeds), (
            f"sampling with different rng_seeds {rng_seeds} produced the same scores for some seeds"
        )

    def test_generate_with_speech(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """With return_audio=True the talker produces a finite, non-empty waveform."""
        result = omni_pipe.generate(
            "Describe this.",
            text_config=_text_config(),
            talker_speech_config=_talker_speech_config(return_audio=True),
        )

        _extract_assert_single_waveform(result)

    def test_matches_optimum_text(self, omni_pipe: ov_genai.OmniPipeline, optimum_reference: OptimumReference) -> None:
        """Greedy decode must produce the same token ids as optimum-intel for a text-only prompt.

        Compared on ids rather than decoded text, so a mismatch shows the first token the two stacks
        disagree on.
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

    def test_speech_generation_rejects_unmatched_role_tokens(
        self, omni_pipe_with_unmatched_role_tokens: ov_genai.OmniPipeline
    ) -> None:
        """A checkpoint whose role token ids never appear in its token stream must raise, not go quiet.

        Speech used to warn and hand back an empty waveform list here, so a misconfigured checkpoint
        looked like a model that simply had nothing to say. The caller asked for audio; failing to
        segment the conversation is a configuration error and has to surface as one.

        This is the inverse of test_generate_with_speech: that one asserts the audio a healthy
        checkpoint owes us, this one pins the diagnosis we give for a broken one.
        """
        with pytest.raises(RuntimeError, match="im_start_token_id") as excinfo:
            omni_pipe_with_unmatched_role_tokens.generate(
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
            pytest.param("video", id="video"),
            pytest.param("audio", id="audio"),
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

    def test_speaker_embedding_steers_speech(self, omni_pipe: ov_genai.OmniPipeline) -> None:
        """A speaker's own embedding reproduces its named voice, and an altered embedding changes the speech."""
        talker = omni_pipe.get_talker()
        speaker = talker.list_speakers()[0]
        embedding = talker.get_speaker_embedding(speaker)
        altered = ov.Tensor(np.array(embedding.data) + 1.0)

        def speak(voice: str | ov.Tensor) -> np.ndarray:
            result = omni_pipe.generate(
                "Describe this.",
                text_config=_text_config(),
                talker_speech_config=_talker_speech_config(return_audio=True, speaker=voice),
            )
            return _extract_assert_single_waveform(result)

        named = speak(speaker)
        assert np.array_equal(speak(embedding), named), (
            f"passing the embedding of {speaker!r} as a tensor must synthesize the same waveform as its name"
        )
        assert not np.array_equal(speak(altered), named), "an altered speaker embedding must change the speech"


# The tiny CI model's tokenizer lacks the audio tokens, so audio dies in the merge assert; audio
# tests use a real export instead. `expected_audio_pads` is the upstream formula, which the
# encoder matches since the disjoint-window fix (CVS-193623).
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
# Audio placement on the prompt path. Token assertions are same-skeleton differentials: two
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
# That is the input a `find`-from-zero expansion loop mis-handles.
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
# Audio across chat turns, multipart messages, and the encoder cache. Where turn 1 differs
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


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_cb_positional_images_stay_images(
    qwen3_omni_cb: ContinuousBatchingPipeline, audio_1s_tensor: openvino.Tensor
):
    """A positional second argument plus audios_batches must bind to images, never to videos."""
    image = openvino.Tensor(np.zeros((64, 64, 3), dtype=np.uint8))
    prompt = "Describe " + audio_tag(0)
    config = [audio_generation_config()]
    positional = qwen3_omni_cb.generate([prompt], [[image]], config, audios_batches=[[audio_1s_tensor]])
    by_keyword = qwen3_omni_cb.generate(
        [prompt], images=[[image]], videos=[[]], generation_config=config, audios_batches=[[audio_1s_tensor]]
    )
    assert positional[0].texts == by_keyword[0].texts
    assert num_input_tokens(positional[0]) == num_input_tokens(by_keyword[0])


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_cb_chat_history_leaves_prompt_calls_stateless(
    qwen3_omni_cb: ContinuousBatchingPipeline, audio_1s_tensor: openvino.Tensor
):
    """A ChatHistory call must not switch later prompt calls into chat mode."""
    prompt = "Describe " + audio_tag(0)
    cb_audio_run(qwen3_omni_cb, [ChatHistory([{"role": "user", "content": prompt}])], [audio_1s_tensor])
    first = cb_audio_run(qwen3_omni_cb, [prompt], [audio_1s_tensor])
    second = cb_audio_run(qwen3_omni_cb, [prompt], [audio_1s_tensor])
    assert first[0].texts == second[0].texts
    assert num_input_tokens(first[0]) == num_input_tokens(second[0])


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_cb_add_request_rejects_audio(
    qwen3_omni_cb: ContinuousBatchingPipeline, audio_1s_tensor: openvino.Tensor
):
    """add_request() has no audio path, so audio must raise instead of being dropped."""
    with pytest.raises(RuntimeError, match="not supported by add_request"):
        qwen3_omni_cb.add_request(0, "Describe", [], [], audio_generation_config(), audios=[audio_1s_tensor])


@pytest.mark.real_models
@pytest.mark.vlm
def test_audio_cb_add_request_rejects_audio_tag(qwen3_omni_cb: ContinuousBatchingPipeline):
    """Without this check the tag would reach the model as plain text."""
    with pytest.raises(RuntimeError, match="Missing image/video/audio with index 0"):
        qwen3_omni_cb.add_request(0, "Describe " + audio_tag(0), [], [], audio_generation_config())


# ----------------------------------------------------------------------------------------------
# The SDPA `VLMPipeline` ChatHistory path. Its overload puts `audios` before
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
