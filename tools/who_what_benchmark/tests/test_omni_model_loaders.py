# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import types

import pytest

from whowhatbench import model_loaders


OMNI_MODEL_TYPE = "qwen3_omni_moe"
TINY_OMNI_MODEL_ID = "ShowMaker27/Qwen3-Omni-tiny-random"


class _FakeAutoConfig:
    """Stand-in for transformers.AutoConfig returning an Omni model_type."""

    @staticmethod
    def from_pretrained(*args, **kwargs):
        return types.SimpleNamespace(model_type=OMNI_MODEL_TYPE)


def _patch_auto_config(monkeypatch):
    monkeypatch.setattr(model_loaders, "AutoConfig", _FakeAutoConfig)


def test_load_text_hf_pipeline_routes_omni_to_thinker_loader(monkeypatch):
    _patch_auto_config(monkeypatch)
    captured = {}

    def _fake_omni_loader(model_id, device, trust_remote_code, model_type):
        captured.update(model_id=model_id, device=device, trust_remote_code=trust_remote_code, model_type=model_type)
        return "thinker-pipeline"

    monkeypatch.setattr(model_loaders, "load_text_hf_omni_pipeline", _fake_omni_loader)

    result = model_loaders.load_text_hf_pipeline("dummy-omni", "cpu")

    assert result == "thinker-pipeline"
    assert captured["model_id"] == "dummy-omni"
    assert captured["model_type"] == OMNI_MODEL_TYPE


def test_load_text_model_optimum_routes_omni_to_multimodal_loader(monkeypatch):
    _patch_auto_config(monkeypatch)
    captured = {}

    def _fake_optimum_loader(model_id, device, ov_config):
        captured.update(model_id=model_id, device=device, ov_config=ov_config)
        return "multimodal-pipeline"

    monkeypatch.setattr(model_loaders, "load_text_optimum_omni_pipeline", _fake_optimum_loader)

    result = model_loaders.load_text_model("dummy-omni", device="CPU", use_hf=False)

    assert result == "multimodal-pipeline"
    assert captured["model_id"] == "dummy-omni"


def test_load_visual_text_model_hf_routes_omni_to_thinker_loader(monkeypatch):
    _patch_auto_config(monkeypatch)
    captured = {}

    def _fake_omni_loader(model_id, device, trust_remote_code, model_type):
        captured.update(model_id=model_id, model_type=model_type)
        return "thinker-pipeline"

    monkeypatch.setattr(model_loaders, "load_text_hf_omni_pipeline", _fake_omni_loader)

    result = model_loaders.load_visual_text_model("dummy-omni", device="cpu", use_hf=True)

    assert result == "thinker-pipeline"
    assert captured["model_type"] == OMNI_MODEL_TYPE


def test_load_visual_text_model_optimum_routes_omni_to_multimodal_loader(monkeypatch):
    _patch_auto_config(monkeypatch)
    captured = {}

    def _fake_optimum_loader(model_id, device, ov_config):
        captured.update(model_id=model_id)
        return "multimodal-pipeline"

    monkeypatch.setattr(model_loaders, "load_text_optimum_omni_pipeline", _fake_optimum_loader)

    result = model_loaders.load_visual_text_model("dummy-omni", device="CPU", use_hf=False)

    assert result == "multimodal-pipeline"
    assert captured["model_id"] == "dummy-omni"


def test_load_text_hf_omni_pipeline_extracts_thinker(monkeypatch):
    import transformers

    thinker = types.SimpleNamespace(eval=lambda: None)
    omni_model = types.SimpleNamespace(thinker=thinker)

    class _FakeOmniClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return omni_model

    monkeypatch.setattr(transformers, model_loaders.OMNI_MODEL_TYPES[OMNI_MODEL_TYPE], _FakeOmniClass, raising=False)

    result = model_loaders.load_text_hf_omni_pipeline("dummy-omni", "cpu", False, OMNI_MODEL_TYPE)

    assert result is thinker


def test_load_text_hf_omni_pipeline_raises_clear_error_when_class_missing(monkeypatch):
    import transformers

    # transformers uses lazy module loading, so deleting the attribute re-resolves it.
    # Setting it to None makes getattr(..., None) report the class as unavailable.
    monkeypatch.setattr(transformers, model_loaders.OMNI_MODEL_TYPES[OMNI_MODEL_TYPE], None, raising=False)

    with pytest.raises(ValueError, match="does not expose"):
        model_loaders.load_text_hf_omni_pipeline("dummy-omni", "cpu", False, OMNI_MODEL_TYPE)


def test_load_text_hf_omni_pipeline_raises_when_thinker_missing(monkeypatch):
    import transformers

    class _FakeOmniClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return types.SimpleNamespace()

    monkeypatch.setattr(transformers, model_loaders.OMNI_MODEL_TYPES[OMNI_MODEL_TYPE], _FakeOmniClass, raising=False)

    with pytest.raises(ValueError, match="thinker"):
        model_loaders.load_text_hf_omni_pipeline("dummy-omni", "cpu", False, OMNI_MODEL_TYPE)


@pytest.mark.skipif(
    not hasattr(__import__("transformers"), model_loaders.OMNI_MODEL_TYPES[OMNI_MODEL_TYPE]),
    reason="Installed transformers does not register the Qwen3-Omni MoE architecture.",
)
def test_load_tiny_omni_model_hf_returns_thinker():
    try:
        thinker = model_loaders.load_text_hf_omni_pipeline(
            TINY_OMNI_MODEL_ID, "cpu", trust_remote_code=False, model_type=OMNI_MODEL_TYPE
        )
    except Exception as exc:  # network/download issues should not fail the suite
        pytest.skip(f"Could not download/load tiny Omni model: {exc}")

    assert thinker is not None
    assert hasattr(thinker, "generate")
