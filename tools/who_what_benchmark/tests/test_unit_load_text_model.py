# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import types


def test_load_text_model_vlm_export_uses_visual_text_model(monkeypatch):
    """A 'text' model-type whose directory looks like a VLM export must be routed to load_visual_text_model."""
    from whowhatbench import model_loaders

    calls = []
    sentinel = object()

    monkeypatch.setattr(model_loaders, "_is_vlm_export", lambda model_dir: True)

    def fake_load_visual_text_model(model_id, device, ov_config, **kwargs):
        calls.append((model_id, device, ov_config, kwargs))
        return sentinel

    monkeypatch.setattr(model_loaders, "load_visual_text_model", fake_load_visual_text_model)

    result = model_loaders.load_text_model(
        "vlm_export_dir",
        device="CPU",
        ov_config={"KEY": "VAL"},
        use_hf=False,
        use_genai=False,
        use_llamacpp=False,
        gguf_file=None,
    )

    assert result is sentinel
    assert calls == [("vlm_export_dir", "CPU", {"KEY": "VAL"}, {"gguf_file": None})]


def test_load_text_model_non_vlm_uses_causal_lm(monkeypatch):
    """A 'text' model-type whose directory is a plain text export must keep using OVModelForCausalLM."""
    from whowhatbench import model_loaders

    calls = []
    sentinel = object()

    monkeypatch.setattr(model_loaders, "_is_vlm_export", lambda model_dir: False)

    visual_text_calls = []
    monkeypatch.setattr(
        model_loaders,
        "load_visual_text_model",
        lambda *args, **kwargs: visual_text_calls.append((args, kwargs)),
    )

    class FakeOVModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, device=None, ov_config=None, **kwargs):
            calls.append((model_id, device, ov_config, kwargs))
            return sentinel

    fake_module = types.ModuleType("optimum.intel.openvino")
    fake_module.OVModelForCausalLM = FakeOVModelForCausalLM
    monkeypatch.setitem(sys.modules, "optimum.intel.openvino", fake_module)

    result = model_loaders.load_text_model(
        "plain_text_dir",
        device="CPU",
        ov_config={"KEY": "VAL"},
        use_hf=False,
        use_genai=False,
        use_llamacpp=False,
    )

    assert result is sentinel
    assert calls == [("plain_text_dir", "CPU", {"KEY": "VAL"}, {})]
    assert visual_text_calls == []
