# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Focused unit tests for GLM-Edge-V (config.model_type == "glm") visual-text
support in WhoWhatBenchmark.

These tests cover the reusable integration behaviour that was missing:

  * the "glm" visual-text inputs preprocessor is registered, and
  * ``load_visual_text_model`` forces ``trust_remote_code`` for a config that
    declares custom multimodal remote code (``auto_map`` + ``vision_config``),
    so the vision-capable custom class is loaded instead of the built-in
    text-only GLM class.

They intentionally avoid downloading weights: the loader behaviour is checked
by patching ``AutoConfig`` / model factories with lightweight fakes.
"""

import types

import pytest


def test_glm_registered_in_inputs_preprocessors():
    from whowhatbench.inputs_preprocessors import MODEL_TYPE_TO_CLS_MAPPING
    from whowhatbench.inputs_preprocessors.glm import GlmEdgeVInputsPreprocessor

    assert "glm" in MODEL_TYPE_TO_CLS_MAPPING
    assert MODEL_TYPE_TO_CLS_MAPPING["glm"] is GlmEdgeVInputsPreprocessor


class _FakeVisionConfig:
    patch_size = 14
    image_size = 672


class _FakeGlmVLMConfig:
    """Mimics a GLM-Edge-V config: model_type "glm" with a vision_config and an
    auto_map pointing at custom remote modeling code."""

    model_type = "glm"
    vision_config = _FakeVisionConfig()
    auto_map = {
        "AutoConfig": "configuration_glm.GlmConfig",
        "AutoModel": "modeling_glm.GlmModel",
        "AutoModelForCausalLM": "modeling_glm.GlmForCausalLM",
    }


def test_loader_forces_remote_code_for_glm_vlm(monkeypatch):
    """A glm config with vision_config + auto_map must be loaded with
    trust_remote_code=True, otherwise the built-in text-only GLM class is used
    and pixel_values / the vision tower are silently dropped."""
    from whowhatbench import model_loaders

    fake_config = _FakeGlmVLMConfig()

    def fake_autoconfig_from_pretrained(model_id, trust_remote_code=False, **kwargs):
        # Built-in "glm" config resolves even without remote code.
        return fake_config

    captured = {}

    class _FakeModel:
        def __init__(self, config):
            self.config = config

        def eval(self):
            return self

    def make_from_pretrained(name):
        def _from_pretrained(model_id, device_map=None, **kwargs):
            # AutoModelForImageTextToText / AutoModelForVision2Seq do not know
            # about "glm" -> mimic transformers raising ValueError so the loader
            # falls through to the AutoModelForCausalLM branch.
            if name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
                raise ValueError("Unrecognized configuration class for glm")
            captured["cls"] = name
            captured["trust_remote_code"] = kwargs.get("trust_remote_code")
            return _FakeModel(fake_config)

        return _from_pretrained

    monkeypatch.setattr(
        model_loaders.AutoConfig, "from_pretrained", fake_autoconfig_from_pretrained
    )
    monkeypatch.setattr(
        model_loaders.AutoModel,
        "from_pretrained",
        make_from_pretrained("AutoModel"),
    )
    monkeypatch.setattr(
        model_loaders.AutoModelForCausalLM,
        "from_pretrained",
        make_from_pretrained("AutoModelForCausalLM"),
    )

    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        make_from_pretrained("AutoModelForImageTextToText"),
        raising=False,
    )
    if hasattr(transformers, "AutoModelForVision2Seq"):
        monkeypatch.setattr(
            transformers.AutoModelForVision2Seq,
            "from_pretrained",
            make_from_pretrained("AutoModelForVision2Seq"),
            raising=False,
        )

    model = model_loaders.load_visual_text_model(
        "some/glm-edge-v", device="CPU", use_hf=True, model_type="visual-text"
    )

    assert model is not None
    # The generation-capable class must be selected via AutoModelForCausalLM ...
    assert captured["cls"] == "AutoModelForCausalLM"
    # ... and remote code must be trusted so the custom VLM class is used.
    assert captured["trust_remote_code"] is True


def test_glm_preprocessor_builds_pixel_values(monkeypatch):
    """The glm preprocessor must place image placeholders via the chat template
    and attach pixel_values from a separately-loaded image processor."""
    import numpy as np
    import torch

    from whowhatbench.inputs_preprocessors.glm import GlmEdgeVInputsPreprocessor

    class _FakeTokenizer:
        chat_template = "dummy"

        def apply_chat_template(
            self, messages, add_generation_prompt=True, return_dict=True,
            tokenize=True, return_tensors="pt",
        ):
            # one image => must contain image content entries
            has_image = any(
                c.get("type") == "image"
                for m in messages
                for c in m["content"]
            )
            assert has_image
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    class _FakeImageProcessor:
        image_mean = [0.5, 0.5, 0.5]

        def __call__(self, images):
            return types.SimpleNamespace(
                pixel_values=np.zeros((1, 1, 1, 3, 4, 4), dtype="float32")
            )

    class _FakeImage:
        pass

    config = _FakeGlmVLMConfig()
    config._name_or_path = "some/glm-edge-v"

    pp = GlmEdgeVInputsPreprocessor()
    # Inject the fake image processor so no download is attempted.
    pp._image_processor = _FakeImageProcessor()

    inputs = pp.preprocess_inputs(
        "What is shown?",
        image=_FakeImage(),
        processor=_FakeTokenizer(),
        tokenizer=_FakeTokenizer(),
        config=config,
        video=None,
    )

    assert "input_ids" in inputs
    assert "attention_mask" in inputs
    assert "pixel_values" in inputs
    assert inputs["pixel_values"].shape[-3:] == (3, 4, 4)


def test_glm_preprocessor_rejects_video():
    from whowhatbench.inputs_preprocessors.glm import GlmEdgeVInputsPreprocessor

    pp = GlmEdgeVInputsPreprocessor()

    class _FakeTokenizer:
        chat_template = "dummy"

    with pytest.raises(ValueError):
        pp.preprocess_inputs(
            "text",
            image=None,
            processor=_FakeTokenizer(),
            tokenizer=_FakeTokenizer(),
            config=_FakeGlmVLMConfig(),
            video=object(),
        )
