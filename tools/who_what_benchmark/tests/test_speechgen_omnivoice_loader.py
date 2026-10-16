# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

from whowhatbench import model_loaders
from whowhatbench.speech_generation_evaluator import OmniVoiceWrapper


class _FakeOmniVoiceModel:
    def eval(self):
        return self

    def generate(self, text, language=None, ref_audio=None):
        return [[0.0, 0.0, 0.0]]


def _install_fake_omnivoice_module(monkeypatch, model_cls=_FakeOmniVoiceModel):
    """Register a fake `omnivoice` module so `OmniVoiceWrapper` never needs the
    real (heavyweight, torch-version-pinned) package to be installed."""
    fake_module = types.ModuleType("omnivoice")

    class _FakeOmniVoiceFactory:
        @staticmethod
        def from_pretrained(model_id):
            return model_cls()

    fake_module.OmniVoice = _FakeOmniVoiceFactory
    monkeypatch.setitem(sys.modules, "omnivoice", fake_module)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("k2-fsa/OmniVoice", True),
        ("K2-FSA/OMNIVOICE-large", True),
        ("hexgrad/Kokoro-82M", False),
        ("microsoft/speecht5_tts", False),
    ],
)
def test_is_omnivoice_model_id_matches_by_name(model_id, expected):
    assert model_loaders._is_omnivoice_model_id(model_id) is expected


def test_is_omnivoice_model_id_matches_local_dir_with_audio_tokenizer(tmp_path):
    model_dir = tmp_path / "local-checkpoint"
    (model_dir / "audio_tokenizer").mkdir(parents=True)

    assert model_loaders._is_omnivoice_model_id(str(model_dir)) is True


def test_is_omnivoice_model_id_rejects_non_string():
    assert model_loaders._is_omnivoice_model_id(None) is False


def test_load_speech_generation_model_routes_omnivoice(monkeypatch):
    _install_fake_omnivoice_module(monkeypatch)

    model = model_loaders.load_speech_generation_model("k2-fsa/OmniVoice", use_hf=True, use_genai=False)

    assert isinstance(model, OmniVoiceWrapper)
    assert model.model_type == "speech-generation"
    assert model.prompts_file == "speech_generation_prompts.yaml"


def test_omnivoice_wrapper_rejects_non_default_torch_dtype(monkeypatch):
    _install_fake_omnivoice_module(monkeypatch)

    with pytest.raises(ValueError):
        OmniVoiceWrapper("k2-fsa/OmniVoice", torch_dtype="fp16")


def test_omnivoice_wrapper_rejects_speaker_embedding(monkeypatch):
    _install_fake_omnivoice_module(monkeypatch)

    wrapper = OmniVoiceWrapper("k2-fsa/OmniVoice")

    with pytest.raises(ValueError):
        wrapper.generate("hello world", speaker_embedding=object())


def test_omnivoice_wrapper_generate_returns_speech_result(monkeypatch):
    _install_fake_omnivoice_module(monkeypatch)

    wrapper = OmniVoiceWrapper("k2-fsa/OmniVoice")
    result = wrapper.generate("hello world", language="en")

    assert result.output_sample_rate == 24000
    assert result.speeches[0].data.dtype.name == "float32"


def test_omnivoice_wrapper_speaker_embedding_shape_is_unused_placeholder(monkeypatch):
    _install_fake_omnivoice_module(monkeypatch)

    wrapper = OmniVoiceWrapper("k2-fsa/OmniVoice")

    assert wrapper.get_speaker_embedding_shape() == (1, 1)
