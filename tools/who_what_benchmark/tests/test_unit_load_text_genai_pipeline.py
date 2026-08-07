# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest


class FakeAdapterConfig:
    """Records .add() calls so tests can assert which adapters were attached."""

    def __init__(self):
        self.added = []

    def add(self, adapter, alpha):
        self.added.append((adapter, alpha))


class FakeAdapter:
    def __init__(self, path):
        self.path = path


class FakePipeline:
    """Generic stand-in returned by VLMPipeline/LLMPipeline, recording constructor args."""

    def __init__(self, calls_list, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        calls_list.append(self)


def _install_fake_openvino_genai(monkeypatch):
    """Stub the openvino_genai module with recorder classes for VLMPipeline/LLMPipeline/etc."""
    fake_module = types.ModuleType("openvino_genai")
    fake_module.vlm_calls = []
    fake_module.llm_calls = []

    fake_module.AdapterConfig = FakeAdapterConfig
    fake_module.Adapter = FakeAdapter
    fake_module.SchedulerConfig = lambda: types.SimpleNamespace()
    fake_module.draft_model = lambda path, device, **kw: {"path": path, "device": device, **kw}
    fake_module.VLMPipeline = lambda *args, **kwargs: FakePipeline(fake_module.vlm_calls, *args, **kwargs)
    fake_module.LLMPipeline = lambda *args, **kwargs: FakePipeline(fake_module.llm_calls, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)
    return fake_module


class FakeGenAIModelWrapper:
    """Stand-in for GenAIModelWrapper that avoids its real AutoConfig.from_pretrained side effect."""

    def __init__(self, model, model_dir, model_type):
        self.model = model
        self.model_dir = model_dir
        self.model_type = model_type


def _load_text_genai_pipeline(monkeypatch, *, is_vlm_export, device="CPU", **kwargs):
    from whowhatbench import model_loaders

    monkeypatch.setattr(model_loaders, "_is_vlm_export", lambda model_dir: is_vlm_export)
    monkeypatch.setattr(model_loaders, "GenAIModelWrapper", FakeGenAIModelWrapper)
    fake_openvino_genai = _install_fake_openvino_genai(monkeypatch)

    result = model_loaders.load_text_genai_pipeline("model_dir", device=device, **kwargs)
    return result, fake_openvino_genai


def test_vlm_export_uses_vlm_pipeline_with_no_adapters_or_cb_config(monkeypatch):
    result, fake_openvino_genai = _load_text_genai_pipeline(monkeypatch, is_vlm_export=True)

    assert len(fake_openvino_genai.vlm_calls) == 1
    assert len(fake_openvino_genai.llm_calls) == 0
    vlm_call = fake_openvino_genai.vlm_calls[0]
    assert vlm_call.args == ("model_dir",)
    assert vlm_call.kwargs == {"device": "CPU"}
    assert isinstance(result, FakeGenAIModelWrapper)
    assert result.model is vlm_call
    assert result.model_dir == "model_dir"
    assert result.model_type == "text"


def test_vlm_export_with_adapters_sets_adapters_kwarg(monkeypatch):
    _, fake_openvino_genai = _load_text_genai_pipeline(
        monkeypatch, is_vlm_export=True, adapters=["adapter_path"], alphas=[0.5],
    )

    vlm_call = fake_openvino_genai.vlm_calls[0]
    assert "adapters" in vlm_call.kwargs
    adapter_config = vlm_call.kwargs["adapters"]
    assert isinstance(adapter_config, FakeAdapterConfig)
    assert len(adapter_config.added) == 1
    added_adapter, added_alpha = adapter_config.added[0]
    assert isinstance(added_adapter, FakeAdapter)
    assert added_adapter.path == "adapter_path"
    assert added_alpha == 0.5


def test_vlm_export_with_cb_config_sets_scheduler_and_attention_backend(monkeypatch):
    _, fake_openvino_genai = _load_text_genai_pipeline(
        monkeypatch, is_vlm_export=True, cb_config={"cache_size": 2},
    )

    vlm_call = fake_openvino_genai.vlm_calls[0]
    assert vlm_call.kwargs["ATTENTION_BACKEND"] == "PA"
    scheduler_config = vlm_call.kwargs["scheduler_config"]
    assert scheduler_config.cache_size == 2


def test_non_vlm_export_uses_llm_pipeline(monkeypatch):
    result, fake_openvino_genai = _load_text_genai_pipeline(monkeypatch, is_vlm_export=False)

    assert len(fake_openvino_genai.llm_calls) == 1
    assert len(fake_openvino_genai.vlm_calls) == 0
    llm_call = fake_openvino_genai.llm_calls[0]
    assert llm_call.args == ("model_dir",)
    assert llm_call.kwargs["device"] == "CPU"
    # Without adapters/none_if_empty, an (empty) AdapterConfig instance is still passed.
    assert isinstance(llm_call.kwargs["adapters"], FakeAdapterConfig)
    assert "scheduler_config" not in llm_call.kwargs
    assert isinstance(result, FakeGenAIModelWrapper)
    assert result.model_type == "text"


def test_non_vlm_export_with_cb_config_uses_continuous_batching(monkeypatch):
    _, fake_openvino_genai = _load_text_genai_pipeline(
        monkeypatch, is_vlm_export=False, cb_config={"cache_size": 4},
    )

    llm_call = fake_openvino_genai.llm_calls[0]
    assert "scheduler_config" in llm_call.kwargs
    assert llm_call.kwargs["scheduler_config"].cache_size == 4
    assert "ATTENTION_BACKEND" not in llm_call.kwargs


def test_vlm_export_with_draft_model_on_npu_raises(monkeypatch):
    """A VLM export loaded via VLMPipeline must be blocked from NPU draft-model use,
    same as an explicit --model-type visual-text run."""
    with pytest.raises(RuntimeError, match="visual-text"):
        _load_text_genai_pipeline(
            monkeypatch,
            is_vlm_export=True,
            device="NPU",
            draft_model="nonexistent_draft_model_dir",
        )


def test_non_vlm_export_with_draft_model_on_npu_is_allowed(monkeypatch, tmp_path):
    """A plain text export on NPU with a draft model is not subject to the VLM restriction."""
    draft_model_dir = tmp_path / "draft_model"
    draft_model_dir.mkdir()

    _, fake_openvino_genai = _load_text_genai_pipeline(
        monkeypatch,
        is_vlm_export=False,
        device="NPU",
        draft_model=str(draft_model_dir),
    )

    llm_call = fake_openvino_genai.llm_calls[0]
    assert "draft_model" in llm_call.kwargs
