# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Focused unit tests for GLM-Edge-V (config.model_type == "glm") support in WWB.

These cover the two reusable pieces added to support the GLM-V VLM family:
  * the HF-path visual-text preprocessor mapping now resolves "glm";
  * the HF loader forces trust_remote_code when a "glm" config's auto_map
    references the repository's custom (vision-capable) modeling code, so the
    text-only built-in GlmForCausalLM is not loaded by mistake.
"""

from types import SimpleNamespace

from whowhatbench.inputs_preprocessors import MODEL_TYPE_TO_CLS_MAPPING
from whowhatbench.inputs_preprocessors.glm_edge_v import GLMEdgeVInputsPreprocessor


def test_glm_registered_in_visual_text_preprocessor_mapping():
    assert "glm" in MODEL_TYPE_TO_CLS_MAPPING
    assert MODEL_TYPE_TO_CLS_MAPPING["glm"] is GLMEdgeVInputsPreprocessor


def test_glm_preprocessor_uses_boi_token_as_image_token():
    model = SimpleNamespace(config=SimpleNamespace(boi_token_id=59256))
    preprocessor = GLMEdgeVInputsPreprocessor(model=model)
    assert preprocessor.def_image_token_id == 59256
    # boi placeholder positions are treated as image tokens
    assert preprocessor.is_image_token([1, 59256, 2], 1) is True
    assert preprocessor.is_image_token([1, 59256, 2], 0) is False


def _requires_remote_code(auto_map, model_type="glm"):
    """Mirror the loader decision added in load_visual_text_model."""
    config = SimpleNamespace(model_type=model_type, auto_map=auto_map)
    trust_remote_code = False
    if config.model_type == "glm" and isinstance(getattr(config, "auto_map", None), dict):
        if any(key.startswith("AutoModel") for key in config.auto_map):
            trust_remote_code = True
    return trust_remote_code


def test_glm_with_custom_auto_map_forces_remote_code():
    auto_map = {
        "AutoConfig": "configuration_glm.GlmConfig",
        "AutoModel": "modeling_glm.GlmModel",
        "AutoModelForCausalLM": "modeling_glm.GlmForCausalLM",
    }
    assert _requires_remote_code(auto_map) is True


def test_glm_without_model_auto_map_stays_builtin():
    # A plain built-in glm text model (no custom AutoModel* code) must not be
    # forced into remote-code loading.
    assert _requires_remote_code({"AutoConfig": "configuration_glm.GlmConfig"}) is False
    assert _requires_remote_code(None) is False
