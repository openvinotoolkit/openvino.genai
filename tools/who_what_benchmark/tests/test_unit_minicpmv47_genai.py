# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

from whowhatbench import model_loaders
from whowhatbench import wwb


def test_genai_wrapper_loads_export_config_without_remote_code(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minicpmv4_7",
                "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
                "auto_map": {
                    "AutoConfig": "configuration_minicpmv4_7.MiniCPMV4_7Config",
                },
            }
        ),
        encoding="utf-8",
    )

    def reject_auto_config(*args, **kwargs):
        raise ValueError("remote configuration code is absent from the OpenVINO export")

    monkeypatch.setattr(model_loaders.AutoConfig, "from_pretrained", reject_auto_config)

    wrapper = model_loaders.GenAIModelWrapper(object(), tmp_path, "visual-text")

    assert wrapper.config.model_type == "minicpmv4_7"


def test_genai_vlm_skips_transformers_processor(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type": "minicpmv4_7"}', encoding="utf-8")

    def reject_processor_load(*args, **kwargs):
        raise AssertionError("GenAI VLM preprocessing belongs to VLMPipeline")

    monkeypatch.setattr(wwb.AutoProcessor, "from_pretrained", reject_processor_load)

    processor, loaded_config = wwb.load_processor(SimpleNamespace(base_model=None, target_model=tmp_path, genai=True))

    assert processor is None
    assert loaded_config.model_type == "minicpmv4_7"


def test_minicpmv47_genai_uses_public_vlm_pipeline(tmp_path, monkeypatch):
    pipeline_calls = []
    (tmp_path / "config.json").write_text('{"model_type": "minicpmv4_7"}', encoding="utf-8")

    class FakeVLMPipeline:
        def __init__(self, model_dir, **kwargs):
            pipeline_calls.append((model_dir, kwargs))

    fake_genai = SimpleNamespace(
        AdapterConfig=lambda: object(),
        VLMPipeline=FakeVLMPipeline,
    )
    monkeypatch.setitem(sys.modules, "openvino_genai", fake_genai)

    wrapper = model_loaders.load_visual_text_genai_pipeline(tmp_path, device="CPU")

    assert isinstance(wrapper.model, FakeVLMPipeline)
    assert pipeline_calls == [(tmp_path, {"device": "CPU"})]
