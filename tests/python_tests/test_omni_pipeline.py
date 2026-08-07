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

This file covers the parts of the surface that can be exercised without loading
a multi-GB Qwen3-Omni checkpoint: imports, config defaults, field round-trips,
and dependency injection of user-defined VLMPipelineBase / TalkerBase children.
The DI tests drive the real C++ OmniPipeline composition path against Python-
defined mocks — no model weights required — so they exercise virtual dispatch,
the text vs speech branching, and the constructor capability guards end to end.

Cross-config validation that requires a real loaded pipeline (actual audio
decode, speaker embeddings from a checkpoint, etc.) lives in the nightly
real-model suites.
"""

from __future__ import annotations

import numpy as np
import openvino as ov
import pytest

import openvino_genai as ov_genai


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
        assert cfg.audio_chunk_frames == 1, "audio_chunk_frames default must be 1"
        assert cfg.rng_seed == 0, "rng_seed default must be 0"
        # talker_*/cp_* sampling overrides are std::optional<T> — exposed as None when unset.
        assert cfg.talker_temperature is None
        assert cfg.talker_top_k is None
        assert cfg.talker_repetition_penalty is None
        assert cfg.cp_temperature is None
        assert cfg.cp_top_k is None
        assert cfg.cp_repetition_penalty is None

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


class RecordingTalker(ov_genai.TalkerBase):
    """User-defined talker that counts invocations. Result fields are read-only from Python, so
    dispatch is proven by the call count rather than by round-tripping audio.
    """

    def __init__(self, speakers: list[str] | None = None) -> None:
        super().__init__()
        self.generate_calls = 0
        self._speakers = speakers if speakers is not None else ["default_voice"]

    def generate(self, vlm_result, talker_speech_config, speech_streamer=None) -> ov_genai.TalkerResults:  # noqa: ANN001
        self.generate_calls += 1
        self.last_return_audio = talker_speech_config.return_audio
        return ov_genai.TalkerResults()

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
        self.last_prompt: str | None = None
        self._audio_output = audio_output
        self._hidden_states = hidden_states

    def generate(  # noqa: ANN001, PLR0913
        self,
        prompt,
        images=[],  # noqa: B006
        videos=[],  # noqa: B006
        audios=[],  # noqa: B006
        videos_metadata=[],  # noqa: B006
        generation_config=None,
        streamer=None,
    ) -> ov_genai.VLMDecodedResults:
        self.generate_calls += 1
        self.last_prompt = prompt
        return ov_genai.VLMDecodedResults()

    def get_tokenizer(self):  # noqa: ANN201
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


class TestCustomVLMSubclass:
    """Users can define a VLMPipelineBase child in Python and have its methods dispatched from C++."""

    def test_subclass_is_instantiable(self) -> None:
        vlm = RecordingVLM()
        assert isinstance(vlm, ov_genai.VLMPipelineBase)

    def test_capability_overrides_are_dispatched(self) -> None:
        vlm = RecordingVLM(audio_output=False, hidden_states=False)
        assert vlm.is_audio_output_enabled() is False
        assert vlm.supports_hidden_states_collection() is False


class TestOmniPipelineDependencyInjection:
    """OmniPipeline(vlm, talker) accepts user-defined children and orchestrates them via C++."""

    def test_construct_from_python_children(self) -> None:
        """The DI constructor accepts a Python-defined VLM and Talker and hands them back verbatim."""
        vlm, talker = RecordingVLM(), RecordingTalker()
        pipe = ov_genai.OmniPipeline(vlm, talker)
        assert pipe.get_vlm() is vlm, "get_vlm() must return the exact injected instance"
        assert pipe.get_talker() is talker, "get_talker() must return the exact injected instance"

    def test_speech_path_invokes_both_stages(self) -> None:
        """With return_audio=True, generate() drives the VLM then the talker via virtual dispatch."""
        vlm, talker = RecordingVLM(), RecordingTalker()
        pipe = ov_genai.OmniPipeline(vlm, talker)

        talker_config = ov_genai.OmniTalkerSpeechConfig()
        talker_config.return_audio = True
        result = pipe.generate("describe this", talker_speech_config=talker_config)

        assert isinstance(result, ov_genai.OmniDecodedResults)
        assert vlm.generate_calls == 1, "the injected VLM must be driven exactly once"
        assert talker.generate_calls == 1, "the injected talker must be driven exactly once"
        assert vlm.last_prompt == "describe this", "the prompt must reach the Python VLM unchanged"
        assert talker.last_return_audio is True

    def test_text_only_path_skips_talker(self) -> None:
        """With return_audio=False, the talker must not be invoked at all."""
        vlm, talker = RecordingVLM(), RecordingTalker()
        pipe = ov_genai.OmniPipeline(vlm, talker)

        talker_config = ov_genai.OmniTalkerSpeechConfig()
        talker_config.return_audio = False
        pipe.generate("just text", talker_speech_config=talker_config)

        assert vlm.generate_calls == 1
        assert talker.generate_calls == 0, "text-only generation must short-circuit the talker"

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
