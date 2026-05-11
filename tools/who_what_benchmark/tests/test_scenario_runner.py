# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RED-phase tests for ``whowhatbench.scenario.runner`` and
``whowhatbench.scenario.result_store``.

The modules under test do not exist yet — these tests are expected to fail
(import errors / attribute errors) until production code is implemented.
The assertions encode the contract described in the implementation plan.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from whowhatbench.scenario.result_store import ResultStore
from whowhatbench.scenario.runner import ScenarioRunner
from whowhatbench.scenario.schema import Scenario

# Two-target scenario shared by most runner tests.
_BASE_SCENARIO: dict[str, Any] = {
    "schema_version": 1,
    "name": "runner-test",
    "models": {
        "base": {"path": "org/base", "backend": "hf"},
        "t1": {"path": "/ov/t1", "backend": "genai"},
        "t2": {"path": "/ov/t2", "backend": "genai"},
    },
    "datasets": {"ds": {"type": "builtin"}},
    "tasks": [
        {
            "id": "chat",
            "type": "text",
            "base": "base",
            "targets": ["t1", "t2"],
            "dataset": "ds",
        }
    ],
}


@pytest.fixture
def simple_scenario() -> Scenario:
    return Scenario.model_validate(copy.deepcopy(_BASE_SCENARIO))


def _make_evaluator_mock(gt_csv_text: str = "prompt,answer\nq1,a1\n") -> MagicMock:
    """Build a mock evaluator that mimics the public surface used by the runner.

    - ``dump_gt(path)`` writes a trivial CSV.
    - ``score(...)`` returns the (per_question, aggregate) DataFrame pair.
    - ``get_generation_fn()`` returns ``None`` (runner falls back to defaults).
    - ``dump_predictions(path)`` writes a trivial CSV.
    """
    evaluator = MagicMock(name="Evaluator")

    def _dump_gt(path: Any) -> None:
        Path(path).write_text(gt_csv_text, encoding="utf-8")

    def _dump_predictions(path: Any) -> None:
        Path(path).write_text("prompt,answer\nq1,a1\n", encoding="utf-8")

    per_question = pd.DataFrame({"prompt": ["q1"], "similarity": [0.9]})
    aggregate = pd.DataFrame({"similarity_mean": [0.9]})

    evaluator.dump_gt.side_effect = _dump_gt
    evaluator.dump_predictions.side_effect = _dump_predictions
    evaluator.score.return_value = (per_question, aggregate)
    evaluator.get_generation_fn.return_value = None
    return evaluator


@pytest.fixture
def mocked_runner_deps():
    """Patch the runner's external dependencies and yield the mocks.

    The mocks live on the runner module path because that is where the runner
    looks up the names — patching the original module would not intercept the
    bound reference.
    """
    with (
        patch("whowhatbench.scenario.runner.load_model") as load_model_mock,
        patch("whowhatbench.scenario.runner.create_evaluator") as create_evaluator_mock,
    ):
        load_model_mock.return_value = MagicMock(name="LoadedModel")
        evaluator = _make_evaluator_mock()
        create_evaluator_mock.return_value = evaluator
        yield {
            "load_model": load_model_mock,
            "create_evaluator": create_evaluator_mock,
            "evaluator": evaluator,
        }


def test_run_returns_result_store(tmp_path: Path, simple_scenario: Scenario, mocked_runner_deps) -> None:
    runner = ScenarioRunner(simple_scenario, output_dir=tmp_path)
    store = runner.run()

    assert isinstance(store, ResultStore)
    results = store.all_results()
    assert len(results) == 2
    target_ids = {r.target_id for r in results}
    assert target_ids == {"t1", "t2"}


def test_gt_generated_once_for_two_targets(tmp_path: Path, simple_scenario: Scenario, mocked_runner_deps) -> None:
    runner = ScenarioRunner(simple_scenario, output_dir=tmp_path)
    runner.run()

    # Two targets share one base — GT must be produced exactly once.
    assert mocked_runner_deps["evaluator"].dump_gt.call_count == 1


def test_gt_cache_hit_on_second_target(tmp_path: Path, simple_scenario: Scenario, mocked_runner_deps) -> None:
    runner = ScenarioRunner(simple_scenario, output_dir=tmp_path)
    store = runner.run()

    results = store.all_results()
    # Iteration order over targets must be deterministic and follow scenario order.
    by_target = {r.target_id: r for r in results}
    assert by_target["t1"].gt_cache_hit is False
    assert by_target["t2"].gt_cache_hit is True


def test_only_task_ids_filters_tasks(tmp_path: Path, mocked_runner_deps) -> None:
    data = copy.deepcopy(_BASE_SCENARIO)
    data["tasks"] = [
        {
            "id": "task1",
            "type": "text",
            "base": "base",
            "targets": ["t1"],
            "dataset": "ds",
        },
        {
            "id": "task2",
            "type": "text",
            "base": "base",
            "targets": ["t2"],
            "dataset": "ds",
        },
    ]
    scenario = Scenario.model_validate(data)

    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    store = runner.run(only_task_ids=["task1"])

    results = store.all_results()
    assert len(results) == 1
    assert results[0].task_id == "task1"


def test_inline_dataset_passes_test_data(tmp_path: Path, mocked_runner_deps) -> None:
    data = copy.deepcopy(_BASE_SCENARIO)
    data["datasets"] = {
        "inline_ds": {"type": "inline", "prompts": ["hello", "world"]},
    }
    data["tasks"] = [
        {
            "id": "chat",
            "type": "text",
            "base": "base",
            "targets": ["t1"],
            "dataset": "inline_ds",
        }
    ]
    scenario = Scenario.model_validate(data)

    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    runner.run()

    create_evaluator_mock = mocked_runner_deps["create_evaluator"]
    assert create_evaluator_mock.call_count >= 1

    # Inline prompts must be forwarded as test_data — otherwise the evaluator
    # would silently fall back to a built-in dataset.
    # Robust: check test_data is present in either args or kwargs, and is non-None
    ce_call = create_evaluator_mock.call_args
    test_data = ce_call.kwargs.get("test_data") if ce_call.kwargs else None
    if test_data is None and ce_call.args:
        # fall back: test_data might be a positional arg beyond the first two
        # (create_evaluator(base_model, args, test_data=...))
        # check all args and kwargs for a list-like value
        for arg in ce_call.args[2:]:
            if hasattr(arg, "__iter__") and not isinstance(arg, str):
                test_data = arg
                break
    assert test_data is not None, "Expected test_data to be passed to create_evaluator for inline dataset"
    # The inline prompts defined in the scenario fixture are ["hello", "world"]
    prompts = list(test_data.get("prompts", [])) if isinstance(test_data, dict) else list(test_data)
    assert len(prompts) >= 1, f"test_data should contain at least one prompt, got {test_data!r}"


def test_csv_layout_after_flush(tmp_path: Path, simple_scenario: Scenario, mocked_runner_deps) -> None:
    runner = ScenarioRunner(simple_scenario, output_dir=tmp_path)
    store = runner.run()
    store.flush_csvs()

    for target_id in ("t1", "t2"):
        metrics_csv = tmp_path / "tasks" / "chat" / target_id / "metrics.csv"
        per_question_csv = tmp_path / "tasks" / "chat" / target_id / "metrics_per_question.csv"
        assert metrics_csv.is_file(), f"missing {metrics_csv}"
        assert per_question_csv.is_file(), f"missing {per_question_csv}"


def test_preexisting_gt_data_skips_base_load(tmp_path: Path, mocked_runner_deps) -> None:
    gt_csv = tmp_path / "preexisting_gt.csv"
    gt_csv.write_text("prompt,answer\nq1,a1\n", encoding="utf-8")

    data = copy.deepcopy(_BASE_SCENARIO)
    data["tasks"] = [
        {
            "id": "chat",
            "type": "text",
            "base": "base",
            "targets": ["t1"],
            "dataset": "ds",
            "gt_data": str(gt_csv),
        }
    ]
    scenario = Scenario.model_validate(data)

    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    runner.run()

    load_model_mock = mocked_runner_deps["load_model"]
    # Only the target should have been loaded; the base must be skipped because
    # GT already exists on disk.
    # More robust: assert no call contained the base model's path string
    base_path = scenario.models["base"].path
    all_call_strs = [str(call) for call in load_model_mock.call_args_list]
    assert not any(base_path in s for s in all_call_strs), (
        f"load_model was called with base path {base_path!r}; "
        f"pre-existing gt_data should have prevented it. Calls: {all_call_strs}"
    )


def test_target_model_loaded_for_each_target(tmp_path: Path, simple_scenario: Scenario, mocked_runner_deps) -> None:
    runner = ScenarioRunner(simple_scenario, output_dir=tmp_path)
    runner.run()

    load_model_mock = mocked_runner_deps["load_model"]
    # Two targets => at least two load_model calls for them. The base may
    # or may not be loaded depending on cache state, so assert with >=.
    target_load_count = sum(
        1
        for call in load_model_mock.call_args_list
        if any("/ov/t1" == arg or "/ov/t2" == arg for arg in (*call.args, *call.kwargs.values()))
    )
    assert target_load_count == 2


def test_target_csv_written_per_target(tmp_path: Path, simple_scenario: Scenario, mocked_runner_deps) -> None:
    """target.csv must be written by the runner via evaluator.dump_predictions()."""
    runner = ScenarioRunner(simple_scenario, output_dir=tmp_path)
    runner.run()

    for target_id in ["t1", "t2"]:
        target_csv = tmp_path / "tasks" / "chat" / target_id / "target.csv"
        assert target_csv.exists(), f"target.csv missing for target {target_id!r} at {target_csv}"
