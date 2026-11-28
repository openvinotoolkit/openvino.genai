# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from pathlib import Path

import openvino_genai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "llm_bench"))

get_lora_config = importlib.import_module("llm_bench_utils.ov_utils").get_lora_config


class FakeAdapter:
    def __init__(self, path):
        self.path = path


class FakeAdapterConfig:
    class Mode:
        MODE_AUTO = "auto"
        MODE_FUSE = "fuse"
        MODE_DYNAMIC = "dynamic"
        MODE_STATIC = "static"
        MODE_STATIC_RANK = "static_rank"

    def __init__(self, mode=Mode.MODE_AUTO):
        self.mode = mode
        self.adapters = []

    def add(self, adapter, alpha):
        self.adapters.append((adapter, alpha))

    def __bool__(self):
        return bool(self.adapters)


@pytest.fixture
def fake_lora_api(monkeypatch):
    monkeypatch.setattr(openvino_genai, "Adapter", FakeAdapter)
    monkeypatch.setattr(openvino_genai, "AdapterConfig", FakeAdapterConfig)


def test_get_lora_config_keeps_all_adapters(tmp_path, fake_lora_api):
    adapter_paths = [tmp_path / "adapter1", tmp_path / "adapter2"]
    for path in adapter_paths:
        path.touch()

    config = get_lora_config([str(path) for path in adapter_paths], ["0.5", "1.5"])

    assert len(config.adapters) == 2
    assert [alpha for _, alpha in config.adapters] == [0.5, 1.5]


def test_get_lora_config_uses_static_rank_mode(tmp_path, fake_lora_api):
    adapter_path = tmp_path / "adapter"
    adapter_path.touch()

    config = get_lora_config([str(adapter_path)], ["1.0"], "static_rank")

    assert config.mode == FakeAdapterConfig.Mode.MODE_STATIC_RANK
