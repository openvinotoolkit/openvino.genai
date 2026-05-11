# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RED-phase tests for the multi-task ``ScenarioRunner`` behaviour.

The runner module does not exist yet — these tests are expected to fail
(import errors / attribute errors) until production code is implemented.
The assertions encode the multi-task contract from the implementation plan:

    * GT is generated once per (base + dataset + seed) cache key, even when
      multiple tasks reuse the same key.
    * Different bases produce different cache keys → GT generated per base.
    * ``run(only_task_ids=...)`` filters tasks; unknown IDs raise.
    * Pre-existing ``gt_data`` skips loading the base model for that task.
    * Each (task, target) result lands in ``tasks/<task_id>/<target_id>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from whowhatbench.scenario.result_store import ResultStore  # noqa: F401 — RED import
from whowhatbench.scenario.runner import ScenarioRunner
from whowhatbench.scenario.schema import Scenario


def _make_multitask_scenario(shared_base: bool = True) -> Scenario:
    """Build a scenario with two tasks; base may be shared or distinct.

    When ``shared_base`` is ``True`` both tasks point at ``base_a`` so the GT
    cache key is identical and dump_gt should run exactly once. When ``False``
    the chat task switches to ``base_b`` so two distinct GT entries are needed.
    """
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "multitask-test",
        "models": {
            "base_a": {"path": "org/base_a", "backend": "hf"},
            "base_b": {"path": "org/base_b", "backend": "hf"},
            "t1": {"path": "/ov/t1", "backend": "genai"},
            "t2": {"path": "/ov/t2", "backend": "genai"},
        },
        "datasets": {
            "ds": {"type": "builtin"},
        },
        "tasks": [
            {
                "id": "task_text",
                "type": "text",
                "base": "base_a",
                "targets": ["t1"],
                "dataset": "ds",
            },
            {
                "id": "task_chat",
                "type": "text-chat",
                "base": "base_a" if shared_base else "base_b",
                "targets": ["t2"],
                "dataset": "ds",
            },
        ],
    }
    return Scenario.model_validate(data)


def _make_evaluator_mock(gt_csv_text: str = "prompt,answer\nq1,a1\n") -> MagicMock:
    """Mock evaluator with the public surface ScenarioRunner consumes."""
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
    """Patch the runner's external dependencies and yield the mocks."""
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


def test_two_tasks_same_base_gt_generated_once(tmp_path: Path, mocked_runner_deps) -> None:
    scenario = _make_multitask_scenario(shared_base=True)
    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    runner.run()

    # Both tasks share base_a + dataset + seed, so the GT cache key matches and
    # dump_gt must run exactly once.
    assert mocked_runner_deps["evaluator"].dump_gt.call_count == 1


def test_two_tasks_different_bases_gt_generated_twice(tmp_path: Path, mocked_runner_deps) -> None:
    scenario = _make_multitask_scenario(shared_base=False)
    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    runner.run()

    # Distinct bases produce distinct cache keys → GT must be regenerated.
    assert mocked_runner_deps["evaluator"].dump_gt.call_count == 2


def test_result_store_has_two_results(tmp_path: Path, mocked_runner_deps) -> None:
    scenario = _make_multitask_scenario(shared_base=True)
    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    store = runner.run()

    assert len(store.all_results()) == 2


def test_result_ids_correct(tmp_path: Path, mocked_runner_deps) -> None:
    scenario = _make_multitask_scenario(shared_base=True)
    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    store = runner.run()

    task_ids = {result.task_id for result in store.all_results()}
    assert task_ids == {"task_text", "task_chat"}


def test_only_task_ids_single(tmp_path: Path, mocked_runner_deps) -> None:
    scenario = _make_multitask_scenario(shared_base=True)
    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    store = runner.run(only_task_ids=["task_text"])

    results = store.all_results()
    assert len(results) == 1
    assert results[0].task_id == "task_text"


def test_only_task_ids_unknown_raises(tmp_path: Path, mocked_runner_deps) -> None:
    scenario = _make_multitask_scenario(shared_base=True)
    runner = ScenarioRunner(scenario, output_dir=tmp_path)

    with pytest.raises(ValueError) as exc_info:
        runner.run(only_task_ids=["nonexistent"])

    # The unknown id should be named in the error so users can fix the typo.
    assert "nonexistent" in str(exc_info.value)


def test_preexisting_gt_data_skips_base_load(tmp_path: Path, mocked_runner_deps) -> None:
    gt_csv = tmp_path / "preexisting_gt.csv"
    gt_csv.write_text("prompt,answer\nq1,a1\n", encoding="utf-8")

    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "preexisting-gt-test",
        "models": {
            "base_a": {"path": "org/base_a", "backend": "hf"},
            "base_b": {"path": "org/base_b", "backend": "hf"},
            "t1": {"path": "/ov/t1", "backend": "genai"},
            "t2": {"path": "/ov/t2", "backend": "genai"},
        },
        "datasets": {"ds": {"type": "builtin"}},
        "tasks": [
            {
                "id": "task_text",
                "type": "text",
                "base": "base_a",
                "targets": ["t1"],
                "dataset": "ds",
                "gt_data": str(gt_csv),
            },
            {
                "id": "task_chat",
                "type": "text-chat",
                "base": "base_b",
                "targets": ["t2"],
                "dataset": "ds",
            },
        ],
    }
    scenario = Scenario.model_validate(data)

    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    runner.run()

    # The first task ships pre-computed GT, so its base ("org/base_a") must
    # never be loaded. The second task has no shortcut and may still load
    # "org/base_b" — that's fine.
    load_model_mock = mocked_runner_deps["load_model"]
    base_path = scenario.models["base_a"].path
    all_call_strs = [str(call) for call in load_model_mock.call_args_list]
    assert not any(base_path in s for s in all_call_strs), (
        f"load_model was called with base path {base_path!r}; gt_data should have prevented it. Calls: {all_call_strs}"
    )


def test_output_subdir_structure(tmp_path: Path, mocked_runner_deps) -> None:
    scenario = _make_multitask_scenario(shared_base=True)
    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    store = runner.run()
    store.flush_csvs()

    # Layout: <output>/tasks/<task_id>/<target_id>/
    assert (tmp_path / "tasks" / "task_text" / "t1").is_dir()
    assert (tmp_path / "tasks" / "task_chat" / "t2").is_dir()


def test_task_with_two_targets_both_results_present(tmp_path: Path, mocked_runner_deps) -> None:
    """A single task with two targets must yield one result per target."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "two-targets-test",
        "models": {
            "base_a": {"path": "org/base_a", "backend": "hf"},
            "t1": {"path": "/ov/t1", "backend": "genai"},
            "t2": {"path": "/ov/t2", "backend": "genai"},
        },
        "datasets": {"ds": {"type": "builtin"}},
        "tasks": [
            {
                "id": "task_text",
                "type": "text",
                "base": "base_a",
                "targets": ["t1", "t2"],
                "dataset": "ds",
            }
        ],
    }
    scenario = Scenario.model_validate(data)

    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    store = runner.run()

    text_task_results = [r for r in store.all_results() if r.task_id == "task_text"]
    assert len(text_task_results) == 2
    target_ids = {r.target_id for r in text_task_results}
    assert target_ids == {"t1", "t2"}


def test_gt_cache_hit_tracked_in_result(tmp_path: Path, mocked_runner_deps) -> None:
    """Shared base → first task computes GT, second task hits the cache."""
    scenario = _make_multitask_scenario(shared_base=True)
    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    store = runner.run()

    results = store.all_results()
    by_task = {r.task_id: r for r in results}
    # The first task in scenario order processes its GT (cache miss)
    first_task_id = scenario.tasks[0].id
    second_task_id = scenario.tasks[1].id
    assert not by_task[first_task_id].gt_cache_hit, "First task should have a GT cache miss"
    assert by_task[second_task_id].gt_cache_hit, "Second task should reuse cached GT (cache hit)"
