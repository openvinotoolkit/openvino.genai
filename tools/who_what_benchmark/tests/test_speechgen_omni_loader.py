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


@pytest.mark.parametrize("use_hf", [True, False], ids=["hf", "optimum"])
def test_load_speech_generation_model_routes_omni(monkeypatch, use_hf):
    import transformers

    monkeypatch.setattr(model_loaders, "_resolve_remote_code_and_config", lambda model_id: (False, _OmniConfig()))
    monkeypatch.setattr(transformers, "AutoProcessor", _FakeAutoProcessor)

    if use_hf:
        monkeypatch.setattr(model_loaders, "load_omni_hf_pipeline", lambda *args, **kwargs: _FakeOmniModel())
    else:
        import optimum.intel.openvino as ov_optimum

        class _FakeOVModelForMultimodalLM:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return _FakeOmniModel()

        monkeypatch.setattr(ov_optimum, "OVModelForMultimodalLM", _FakeOVModelForMultimodalLM, raising=False)

    model = model_loaders.load_speech_generation_model("dummy-omni", use_hf=use_hf, use_genai=False)

    assert isinstance(model, Qwen3OmniSpeechWrapper)
    assert model.model_type == "speech-generation"


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
    monkeypatch.setattr(openvino_genai, "OmniPipeline", _FakeOmniPipeline, raising=False)

    model = model_loaders.load_speech_generation_genai_pipeline("dummy-omni-dir", "CPU")

    assert isinstance(model, GenAIOmniSpeechWrapper)


def test_qwen3_omni_speech_wrapper_rejects_speaker_embedding():
    wrapper = Qwen3OmniSpeechWrapper(model=object(), processor=object())

    with pytest.raises(ValueError):
        wrapper.generate("hello world", speaker_embedding=object())


def test_qwen3_omni_wrappers_declare_text_prompts_file():
    hf_wrapper = Qwen3OmniSpeechWrapper(model=object(), processor=object())
    genai_wrapper = GenAIOmniSpeechWrapper.__new__(GenAIOmniSpeechWrapper)

    assert hf_wrapper.prompts_file == "text_prompts.yaml"
    assert genai_wrapper.prompts_file == "text_prompts.yaml"
