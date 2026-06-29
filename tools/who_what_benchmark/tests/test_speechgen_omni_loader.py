# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from whowhatbench import model_loaders
from whowhatbench.speech_generation_evaluator import Qwen3OmniSpeechWrapper, GenAIOmniSpeechWrapper


class _OmniConfig:
    model_type = "qwen3_omni_moe"


class _FakeOmniModel:
    has_talker = True


class _FakeProcessor:
    pass


class _FakeAutoProcessor:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _FakeProcessor()


def test_load_speech_generation_model_hf_routes_omni(monkeypatch):
    import transformers

    monkeypatch.setattr(model_loaders, "_resolve_remote_code_and_config", lambda model_id: (False, _OmniConfig()))
    monkeypatch.setattr(model_loaders, "load_omni_hf_pipeline", lambda *args, **kwargs: _FakeOmniModel())
    monkeypatch.setattr(transformers, "AutoProcessor", _FakeAutoProcessor)

    model = model_loaders.load_speech_generation_model("dummy-omni", use_hf=True)

    assert isinstance(model, Qwen3OmniSpeechWrapper)
    assert model.model_type == "speech-generation"


def test_load_speech_generation_model_optimum_routes_omni(monkeypatch):
    import transformers
    import optimum.intel.openvino as ov_optimum

    monkeypatch.setattr(model_loaders, "_resolve_remote_code_and_config", lambda model_id: (False, _OmniConfig()))

    class _FakeOVModelForMultimodalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeOmniModel()

    monkeypatch.setattr(ov_optimum, "OVModelForMultimodalLM", _FakeOVModelForMultimodalLM, raising=False)
    monkeypatch.setattr(transformers, "AutoProcessor", _FakeAutoProcessor)

    model = model_loaders.load_speech_generation_model("dummy-omni", use_hf=False, use_genai=False)

    assert isinstance(model, Qwen3OmniSpeechWrapper)


def test_load_speech_generation_genai_pipeline_routes_omni(monkeypatch):
    import openvino_genai

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _OmniConfig()

    class _FakeOmniPipeline:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(model_loaders, "AutoConfig", _FakeAutoConfig)
    monkeypatch.setattr(openvino_genai, "OmniPipeline", _FakeOmniPipeline)

    model = model_loaders.load_speech_generation_genai_pipeline("dummy-omni-dir", "CPU")

    assert isinstance(model, GenAIOmniSpeechWrapper)


def test_qwen3_omni_speech_wrapper_rejects_speaker_embedding():
    wrapper = Qwen3OmniSpeechWrapper(model=object(), processor=object())

    with pytest.raises(ValueError):
        wrapper.generate("hello world", speaker_embedding=object())
