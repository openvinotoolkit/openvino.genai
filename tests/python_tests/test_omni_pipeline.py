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


SHORT_GENERATION_TOKENS = 24


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

        Clean break per CLAUDE 'No Unrequested Backward Compatibility'. If this
        attribute survives, somebody added a deprecation alias against the migration
        contract.
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

    def test_update_anymap_speech_fields(self) -> None:
        """update_generation_config(AnyMap) sets the speech-side fields."""
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.update_generation_config(
            return_audio=False,
            speaker="any_voice_id",
            audio_chunk_frames=2,
            rng_seed=7,
            talker_temperature=0.7,
            talker_top_k=20,
            cp_temperature=0.5,
            cp_top_k=10,
        )

        assert cfg.return_audio is False
        assert cfg.speaker == "any_voice_id"
        assert cfg.audio_chunk_frames == 2
        assert cfg.rng_seed == 7
        assert cfg.talker_temperature == pytest.approx(0.7)
        assert cfg.talker_top_k == 20
        assert cfg.cp_temperature == pytest.approx(0.5)
        assert cfg.cp_top_k == 10

    def test_update_anymap_max_new_tokens(self) -> None:
        """OmniTalkerSpeechConfig carries its own max_new_tokens (talker AR cap).

        Independent of GenerationConfig.max_new_tokens (which caps the thinker text
        decode). Both can be set simultaneously to different values when the caller
        constructs typed configs explicitly.
        """
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.update_generation_config(max_new_tokens=128)
        assert cfg.max_new_tokens == 128

    def test_validate_rejects_zero_chunk_frames(self) -> None:
        """audio_chunk_frames must be >= 1 when return_audio is true."""
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.return_audio = True
        cfg.audio_chunk_frames = 0

        with pytest.raises(RuntimeError, match=r"(?i)audio_chunk_frames|chunk"):
            cfg.validate()

    def test_validate_ignores_chunk_frames_when_audio_disabled(self) -> None:
        """return_audio=False relaxes audio-only invariants.

        Text-only generation through OmniPipeline must not trip the audio_chunk_frames
        check — the field is irrelevant when the talker isn't running.
        """
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.return_audio = False
        cfg.audio_chunk_frames = 0
        cfg.validate()  # Must not raise.

    def test_validate_rejects_negative_talker_temperature(self) -> None:
        """talker_temperature must be > 0 when set."""
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.return_audio = True
        cfg.talker_temperature = -0.1

        with pytest.raises(RuntimeError, match=r"(?i)talker_temperature"):
            cfg.validate()

    def test_validate_rejects_zero_talker_top_k(self) -> None:
        """talker_top_k must be >= 1 when set."""
        cfg = ov_genai.OmniTalkerSpeechConfig()
        cfg.return_audio = True
        cfg.talker_top_k = 0

        with pytest.raises(RuntimeError, match=r"(?i)talker_top_k"):
            cfg.validate()
