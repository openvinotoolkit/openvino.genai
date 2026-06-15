# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import openvino_genai as ov_genai


@pytest.mark.speech_generation
class TestSpeechGenerationConfigQwenFields:
    def test_qwen_fields_kwargs_constructor(self):
        config = ov_genai.SpeechGenerationConfig(
            speaker="ryan",
            instruct="speak slowly",
            non_streaming_mode=False,
            subtalker_dosample=False,
            subtalker_top_k=64,
            subtalker_top_p=0.95,
            subtalker_temperature=0.7,
            seed=42,
        )

        assert config.speaker == "ryan"
        assert config.instruct == "speak slowly"
        assert config.non_streaming_mode is False
        assert config.subtalker_dosample is False
        assert config.subtalker_top_k == 64
        assert config.subtalker_top_p == pytest.approx(0.95)
        assert config.subtalker_temperature == pytest.approx(0.7)
        assert config.seed == 42

    def test_qwen_fields_update_generation_config(self):
        config = ov_genai.SpeechGenerationConfig()
        config.update_generation_config(
            speaker="vivian",
            instruct="neutral style",
            non_streaming_mode=True,
            subtalker_dosample=True,
            subtalker_top_k=32,
            subtalker_top_p=0.9,
            subtalker_temperature=0.8,
            seed=7,
        )

        assert config.speaker == "vivian"
        assert config.instruct == "neutral style"
        assert config.non_streaming_mode is True
        assert config.subtalker_dosample is True
        assert config.subtalker_top_k == 32
        assert config.subtalker_top_p == pytest.approx(0.9)
        assert config.subtalker_temperature == pytest.approx(0.8)
        assert config.seed == 7
