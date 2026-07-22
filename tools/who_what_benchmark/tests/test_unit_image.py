# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_text2image_genai_npu_reshapes_before_compile(monkeypatch):
    """Test that NPU text-to-image pipeline reshapes to static shapes before compilation."""
    from whowhatbench import model_loaders

    calls = []

    class FakePipeline:
        def __init__(self, model_dir):
            calls.append(("init", model_dir))

        def get_generation_config(self):
            calls.append(("get_generation_config",))
            return SimpleNamespace(guidance_scale=7.5)

        def reshape(self, **kwargs):
            calls.append(("reshape", kwargs))

        def compile(self, device, **properties):
            calls.append(("compile", device, properties))

    class FakeGenAIModelWrapper:
        def __init__(self, model, model_dir, model_type):
            self.model = model
            self.model_dir = model_dir
            self.model_type = model_type

    monkeypatch.setitem(
        sys.modules,
        "openvino_genai",
        SimpleNamespace(Text2ImagePipeline=FakePipeline),
    )
    monkeypatch.setattr(model_loaders, "GenAIModelWrapper", FakeGenAIModelWrapper)

    wrapper = model_loaders.load_text2image_genai_pipeline(
        "model_dir",
        device="NPU",
        ov_config={"CACHE_DIR": "cache"},
        image_size=512,
    )

    assert wrapper.model_type == "text-to-image"
    assert calls == [
        ("init", "model_dir"),
        ("get_generation_config",),
        (
            "reshape",
            {"num_images_per_prompt": 1, "height": 512, "width": 512, "guidance_scale": 7.5},
        ),
        ("compile", "NPU", {"CACHE_DIR": "cache"}),
    ]


def test_text2image_genai_npu_requires_positive_image_size(monkeypatch):
    """Test that NPU text-to-image pipeline requires positive image-size."""
    from whowhatbench import model_loaders

    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(Text2ImagePipeline=object))

    with pytest.raises(ValueError, match="positive --image-size"):
        model_loaders.load_text2image_genai_pipeline("model_dir", device="NPU", ov_config={})


def test_genai_gen_image_uses_model_adapter_config():
    """Test that genai_gen_image forwards model adapter_config to generate()."""
    from whowhatbench.wwb import genai_gen_image

    calls = []

    class FakeModel:
        resolution = (512, 512)
        adapter_config = object()

        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return SimpleNamespace(data=np.zeros((1, 1, 1, 3), dtype=np.uint8))

    genai_gen_image(FakeModel(), "prompt", 4)

    assert calls[0][1]["adapters"] is FakeModel.adapter_config
