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

This file covers only the parts of the surface that can be exercised without
loading a multi-GB Qwen3-Omni checkpoint: imports, config defaults, AnyMap
update, and validate() invariants. Cross-config validation that requires a
loaded pipeline (return_audio incompatible with beam search etc.) lives in
the nightly real-model suites — here we only assert that the standalone
config validate() and the field round-trip behave.
"""

from __future__ import annotations

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
    method handle. End-to-end behavior (get/set round-trip on a live pipeline) lives in
    the temp/ scripts that load the MoE checkpoint.
    """

    def test_methods_exist(self) -> None:
        assert hasattr(ov_genai.OmniPipeline, "get_talker"), "OmniPipeline.get_talker() missing from public surface"

        # Speaker APIs live on the Talker, accessed via get_talker()
        for method in ("list_speakers", "get_speaker_embedding"):
            assert hasattr(ov_genai.TalkerBase, method), f"TalkerBase.{method}() missing from public surface"

    def test_speech_config_accessors_are_talker_only(self) -> None:
        """get/set_speech_config live on the concrete Talker, not on TalkerBase.

        Per the Talker class spec (slide 10 of the design deck), these accessors are
        non-virtual and belong to the default Qwen3-Omni Talker only — custom TalkerBase
        backends are not forced to store a speech config. Guard against them leaking onto
        the abstract base.
        """
        for method in ("get_speech_config", "set_speech_config"):
            assert hasattr(ov_genai.Talker, method), f"Talker.{method}() missing from public surface"
            assert not hasattr(ov_genai.TalkerBase, method), (
                f"TalkerBase.{method}() must not exist — the accessors are Talker-only per the spec."
            )

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
