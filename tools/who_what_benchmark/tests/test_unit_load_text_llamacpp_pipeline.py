# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import types
import os


class FakeLlama:
    """Records the constructor args/kwargs it was called with."""

    last_call = None

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.kwargs = kwargs
        FakeLlama.last_call = self


def _install_fake_llama_cpp(monkeypatch):
    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    FakeLlama.last_call = None
    return fake_module


def test_load_text_llamacpp_pipeline_with_gguf_file_joins_model_dir_and_file(monkeypatch):
    """When gguf_file is provided, Llama must be loaded from model_dir/gguf_file, not model_dir alone."""
    from whowhatbench import model_loaders

    _install_fake_llama_cpp(monkeypatch)

    model = model_loaders.load_text_llamacpp_pipeline("model_dir", gguf_file="model.gguf")

    assert isinstance(model, FakeLlama)
    assert model.model_path == os.path.join("model_dir", "model.gguf")


def test_load_text_llamacpp_pipeline_without_gguf_file_uses_model_dir(monkeypatch):
    """Without gguf_file, Llama must be loaded directly from model_dir."""
    from whowhatbench import model_loaders

    _install_fake_llama_cpp(monkeypatch)

    model = model_loaders.load_text_llamacpp_pipeline("model_dir")

    assert isinstance(model, FakeLlama)
    assert model.model_path == "model_dir"


def test_load_text_llamacpp_pipeline_passes_n_ctx_when_provided(monkeypatch):
    """llamacpp_n_ctx must be forwarded to Llama as an integer n_ctx kwarg."""
    from whowhatbench import model_loaders

    _install_fake_llama_cpp(monkeypatch)

    model = model_loaders.load_text_llamacpp_pipeline("model_dir", llamacpp_n_ctx="4096")

    assert model.kwargs == {"n_ctx": 4096}


def test_load_text_llamacpp_pipeline_omits_n_ctx_when_not_provided(monkeypatch):
    """Without llamacpp_n_ctx, no n_ctx kwarg should be forwarded to Llama."""
    from whowhatbench import model_loaders

    _install_fake_llama_cpp(monkeypatch)

    model = model_loaders.load_text_llamacpp_pipeline("model_dir")

    assert model.kwargs == {}
