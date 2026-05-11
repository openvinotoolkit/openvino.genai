# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RED-phase tests for ``whowhatbench.scenario.gt_cache``.

The module under test does not exist yet — these tests are expected to fail
until ``GTCache`` is implemented. They encode the cache-key contract from the
plan: GT-affecting parameters change the key, target-only parameters do not.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

from whowhatbench.scenario.gt_cache import GTCache
from whowhatbench.scenario.schema import Scenario


def make_scenario(**overrides: Any) -> Scenario:
    """Build a minimal valid Scenario, with optional top-level overrides."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "test-scenario",
        "models": {
            "base": {"path": "org/base", "backend": "hf"},
            "target_genai": {"path": "/ov/target", "backend": "genai", "device": "CPU"},
            "target_gpu": {"path": "/ov/target", "backend": "genai", "device": "GPU.0"},
        },
        "datasets": {
            "ds_builtin": {"type": "builtin"},
            "ds_hf": {
                "type": "huggingface",
                "path": "squad",
                "split": "validation[:32]",
                "field": "question",
            },
        },
        "tasks": [
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
            }
        ],
    }
    data.update(overrides)
    return Scenario.model_validate(data)


def _components(scenario: Scenario, dataset_id: str = "ds_builtin") -> dict[str, Any]:
    """Pack the four arguments GTCache.key() expects in one place."""
    task = scenario.tasks[0]
    base_model = scenario.models[task.base]
    dataset = scenario.datasets[dataset_id]
    return {
        "task": task,
        "base_model": base_model,
        "dataset": dataset,
        "scenario": scenario,
    }


def _key(cache: GTCache, scenario: Scenario, dataset_id: str = "ds_builtin") -> str:
    return cache.key(**_components(scenario, dataset_id=dataset_id))


def test_key_is_16_hex_chars(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario = make_scenario()
    key = _key(cache, scenario)
    assert isinstance(key, str)
    assert len(key) == 16
    assert all(ch in string.hexdigits for ch in key)


def test_key_stability(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario = make_scenario()
    assert _key(cache, scenario) == _key(cache, scenario)


def test_key_differs_on_base_path(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario_a = make_scenario()
    scenario_b = make_scenario()
    scenario_b.models["base"].path = "org/different-base"
    assert _key(cache, scenario_a) != _key(cache, scenario_b)


def test_key_differs_on_task_type(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario_a = make_scenario()
    scenario_b = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text-chat",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
            }
        ]
    )
    assert _key(cache, scenario_a) != _key(cache, scenario_b)


def test_key_differs_on_dataset(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario = make_scenario()
    key_builtin = _key(cache, scenario, dataset_id="ds_builtin")
    key_hf = _key(cache, scenario, dataset_id="ds_hf")
    assert key_builtin != key_hf


def test_key_differs_on_num_samples(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario_a = make_scenario()
    scenario_b = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
                "num_samples": 16,
            }
        ]
    )
    assert _key(cache, scenario_a) != _key(cache, scenario_b)


def test_key_differs_on_seed(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario_a = make_scenario()
    scenario_b = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
                "generation": {"seed": 1234},
            }
        ]
    )
    assert _key(cache, scenario_a) != _key(cache, scenario_b)


def test_key_differs_on_max_new_tokens(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario_a = make_scenario()
    scenario_b = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
                "generation": {"max_new_tokens": 999},
            }
        ]
    )
    assert _key(cache, scenario_a) != _key(cache, scenario_b)


def test_key_does_not_differ_on_device(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario_a = make_scenario()
    scenario_b = make_scenario()
    scenario_b.models["target_genai"].device = "GPU.0"
    assert _key(cache, scenario_a) == _key(cache, scenario_b)


def test_key_does_not_differ_on_ov_config(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario_a = make_scenario()
    scenario_b = make_scenario()
    scenario_b.models["target_genai"].ov_config = {"CACHE_DIR": "/tmp"}
    assert _key(cache, scenario_a) == _key(cache, scenario_b)


def test_cache_miss_on_empty_dir(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario = make_scenario()
    key = _key(cache, scenario)
    assert cache.get(key) is None


def test_cache_hit_after_put(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario = make_scenario()
    key = _key(cache, scenario)

    src_csv = tmp_path / "_gt.csv"
    src_csv.write_text("prompt,answer\nhello,world\n", encoding="utf-8")

    cache.put(key, src_csv)

    cached = cache.get(key)
    assert cached is not None
    assert isinstance(cached, Path)
    assert cached.exists()
    assert cached.is_file()


def test_meta_json_written_on_put(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    scenario = make_scenario()
    key = _key(cache, scenario)

    src_csv = tmp_path / "_gt.csv"
    src_csv.write_text("prompt,answer\nhello,world\n", encoding="utf-8")
    cache.put(key, src_csv)

    meta_path = cache.meta_path(key)
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    # Meta must be a JSON object describing the key components, so the cache
    # is auditable. Exact field names are intentionally not asserted; only the
    # structural promise that it's a dict and not, say, a bare string or list.
    assert isinstance(payload, dict)
    assert payload, "meta JSON must not be empty"


def test_get_returns_none_for_missing_key(tmp_path: Path) -> None:
    cache = GTCache(tmp_path)
    assert cache.get("0123456789abcdef") is None


def test_cache_dir_created_if_not_exists(tmp_path: Path) -> None:
    sub = tmp_path / "subdir"
    assert not sub.exists()
    GTCache(sub)
    assert sub.exists()
    assert sub.is_dir()
