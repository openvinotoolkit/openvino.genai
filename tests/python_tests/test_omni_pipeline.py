# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the public OmniPipeline API.

Public surface under test:

    openvino_genai.OmniPipeline
    openvino_genai.OmniSpeechGenerationConfig
    openvino_genai.OmniSpeechStreamerBase

The pybind variant alias OmniSpeechStreamerVariant is intentionally not in
the smoke import — variant aliases are not re-exported in __init__.py per the
existing AudioStreamerVariant convention.

This file covers only the parts of the surface that can be exercised without
loading a multi-GB Qwen3-Omni checkpoint: imports, config defaults, AnyMap
update, and validate() invariants. End-to-end generation is exercised by the
nightly real-model suites.
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
            OmniSpeechGenerationConfig,
            OmniSpeechStreamerBase,
        )


class TestOmniSpeechGenerationConfig:
    """OmniSpeechGenerationConfig — defaults, validate(), AnyMap update."""

    def test_omni_speech_generation_config_defaults(self) -> None:
        """Default ctor exposes the three Omni fields and inherits GenerationConfig."""
        cfg = ov_genai.OmniSpeechGenerationConfig()

        assert cfg.return_audio is True, "return_audio default must be True"
        assert cfg.speaker == "", "speaker default must be empty (model default)"
        assert cfg.audio_chunk_frames == 1, "audio_chunk_frames default must be 1"

        # Pybind exposes the C++ inheritance via Python's MRO. isinstance check is the
        # robust probe — hasattr would still pass on a property descriptor.
        assert isinstance(cfg, ov_genai.GenerationConfig), "OmniSpeechGenerationConfig must subclass GenerationConfig"
        cfg.max_new_tokens = 64
        assert cfg.max_new_tokens == 64

    def test_omni_speech_generation_config_update_anymap(self) -> None:
        """update_generation_config(AnyMap) extracts Omni fields AND forwards inherited ones.

        read_anymap_param pulls the three Omni fields, then forwards the remaining map
        to GenerationConfig::update. Tests both halves: an Omni field and an inherited
        field land in one call.
        """
        cfg = ov_genai.OmniSpeechGenerationConfig()
        cfg.update_generation_config(
            return_audio=False,
            speaker="any_voice_id",
            audio_chunk_frames=2,
            max_new_tokens=99,
        )

        assert cfg.return_audio is False
        assert cfg.speaker == "any_voice_id"
        assert cfg.audio_chunk_frames == 2
        assert cfg.max_new_tokens == 99, "Inherited field must be forwarded to GenerationConfig::update"

    def test_omni_speech_generation_config_validate_rejects_beam_search(self) -> None:
        """return_audio=True with num_beams>1 is invariant-violating.

        Beam search produces multiple candidate sequences but the talker consumes a
        single hidden-state stream — they are incompatible.
        """
        cfg = ov_genai.OmniSpeechGenerationConfig()
        cfg.return_audio = True
        cfg.num_beams = 2

        with pytest.raises(RuntimeError, match=r"(?i)beam"):
            cfg.validate()

    def test_omni_speech_generation_config_validate_rejects_prompt_lookup(self) -> None:
        """return_audio=True with prompt_lookup_num_tokens>0 is invariant-violating.

        Sibling invariant to the beam-search one — same source line in generation_config.cpp.
        """
        cfg = ov_genai.OmniSpeechGenerationConfig()
        cfg.return_audio = True
        cfg.num_assistant_tokens = 4

        with pytest.raises(RuntimeError, match=r"(?i)assistant|prompt|lookup|speculative"):
            cfg.validate()

    def test_omni_speech_generation_config_validate_rejects_zero_chunk_frames(self) -> None:
        """audio_chunk_frames must be >= 1; zero is a streaming-granularity contradiction."""
        cfg = ov_genai.OmniSpeechGenerationConfig()
        cfg.audio_chunk_frames = 0

        with pytest.raises(RuntimeError, match=r"(?i)audio_chunk_frames|chunk"):
            cfg.validate()

    def test_omni_speech_generation_config_validate_passes_when_audio_disabled(self) -> None:
        """return_audio=False relaxes audio-only invariants; beam search must be allowed.

        Invariants like "no beam search" only apply when the talker is engaged. Text-only
        callers must be free to use beam search through the same config type — otherwise
        OmniPipeline cannot subsume LLM-style generation paths.
        """
        cfg = ov_genai.OmniSpeechGenerationConfig()
        cfg.return_audio = False
        cfg.num_beams = 2
        cfg.max_new_tokens = SHORT_GENERATION_TOKENS

        # Must not raise.
        cfg.validate()
