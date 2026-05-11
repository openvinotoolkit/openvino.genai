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
from whowhatbench.scenario.runner import ScenarioRunner, _normalize_metrics
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


# ── F7: missing openvino_genai must surface as a clear ImportError ────────────


def test_text_to_image_evaluator_raises_import_error_when_unavailable() -> None:
    """When openvino_genai is unavailable the registry must contain a stub
    entry for 'text-to-image' that raises ``ImportError`` on instantiation —
    not ``KeyError`` (registry miss) or ``TypeError`` ('NoneType' is not callable).
    """
    from whowhatbench import EVALUATOR_REGISTRY, Text2ImageEvaluator

    if Text2ImageEvaluator is not None:
        pytest.skip("openvino_genai is available; F7 stub behaviour is not exercised")

    # The registry must still expose 'text-to-image' so callers don't get a
    # KeyError before the import diagnostics surface.
    assert "text-to-image" in EVALUATOR_REGISTRY, (
        "EVALUATOR_REGISTRY must contain a 'text-to-image' stub entry even when "
        "openvino_genai is unavailable; got keys: "
        f"{sorted(EVALUATOR_REGISTRY.keys())}"
    )

    stub_cls = EVALUATOR_REGISTRY["text-to-image"]
    with pytest.raises(ImportError) as exc_info:
        stub_cls()

    assert "openvino_genai" in str(exc_info.value), (
        f"ImportError message should mention 'openvino_genai'; got: {exc_info.value!r}"
    )


def test_create_evaluator_for_unavailable_type_raises_import_error(tmp_path: Path) -> None:
    """``create_evaluator`` must raise ``ImportError`` (not ``KeyError`` /
    ``ValueError`` / ``TypeError``) for task types whose evaluator module
    failed to import because openvino_genai is missing.
    """
    import argparse

    from whowhatbench import Text2ImageEvaluator
    from whowhatbench.wwb import create_evaluator

    if Text2ImageEvaluator is not None:
        pytest.skip("openvino_genai is available; F7 stub behaviour is not exercised")

    # Minimal Namespace covering every attribute the text-to-image branch in
    # ``create_evaluator`` reads. Values are placeholders — the call must fail
    # with ImportError before any of them is consumed.
    args = argparse.Namespace(
        model_type="text-to-image",
        gt_data=str(tmp_path / "gt.csv"),
        num_samples=1,
        image_size=64,
        num_inference_steps=1,
        empty_adapters=False,
        genai=False,
        seed=0,
        dataset=None,
        dataset_field=None,
        split=None,
    )

    with pytest.raises(ImportError) as exc_info:
        create_evaluator(base_model=None, args=args, test_data={"prompts": ["hello"]})

    assert "openvino_genai" in str(exc_info.value), (
        f"ImportError message should mention 'openvino_genai'; got: {exc_info.value!r}"
    )


def test_llamacpp_kwargs_passed_to_load_model(tmp_path: Path, mocked_runner_deps) -> None:
    """F2 — runner must forward llamacpp/gguf kwargs to load_model for the target.

    The legacy load_model signature accepts `use_llamacpp` and `gguf_file` as
    kwargs; dropping them silently routes a llamacpp model through the wrong
    backend path.
    """
    data = copy.deepcopy(_BASE_SCENARIO)
    data["models"] = {
        "base": {"path": "org/base", "backend": "hf"},
        "t1": {"path": "/weights.gguf", "backend": "llamacpp", "gguf_file": "/weights.gguf"},
    }
    data["tasks"] = [
        {
            "id": "chat",
            "type": "text",
            "base": "base",
            "targets": ["t1"],
            "dataset": "ds",
        }
    ]
    scenario = Scenario.model_validate(data)

    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    runner.run()

    load_model_mock = mocked_runner_deps["load_model"]
    target_calls = [
        call
        for call in load_model_mock.call_args_list
        if "/weights.gguf" in (call.args[1] if len(call.args) > 1 else call.kwargs.get("model_id", ""))
    ]
    assert target_calls, f"No load_model call for the target llamacpp model. Calls: {load_model_mock.call_args_list}"
    target_call = target_calls[0]

    use_llamacpp = target_call.kwargs.get("use_llamacpp")
    if use_llamacpp is None and len(target_call.args) >= 7:
        use_llamacpp = target_call.args[6]
    assert use_llamacpp is True, (
        f"load_model must be called with use_llamacpp=True for llamacpp backend. "
        f"args={target_call.args} kwargs={target_call.kwargs}"
    )

    gguf_file = target_call.kwargs.get("gguf_file")
    assert gguf_file == "/weights.gguf", (
        f"load_model must forward gguf_file kwarg. args={target_call.args} kwargs={target_call.kwargs}"
    )


def test_gt_args_use_base_backend_not_target(tmp_path: Path, mocked_runner_deps) -> None:
    """F3 — args namespace built for GT generation must reflect the BASE backend.

    When base=hf and target=genai, building args from the target id makes
    `args.hf=False, args.genai=True` even though the GT is produced by the hf
    base model. The evaluator branches on these flags, so the wrong backend
    code path is exercised during GT generation.
    """
    runner = ScenarioRunner(
        Scenario.model_validate(copy.deepcopy(_BASE_SCENARIO)),
        output_dir=tmp_path,
    )
    runner.run()

    create_evaluator_mock = mocked_runner_deps["create_evaluator"]
    # The GT-generation call passes the loaded base_model as the first positional arg;
    # later target-evaluation calls pass None.
    gt_calls = [
        call
        for call in create_evaluator_mock.call_args_list
        if (call.args and call.args[0] is not None) or (call.kwargs.get("base_model") is not None)
    ]
    assert gt_calls, (
        f"No GT-generation call to create_evaluator (first arg should be loaded base model). "
        f"Calls: {create_evaluator_mock.call_args_list}"
    )

    args_for_gt = gt_calls[0].args[1] if len(gt_calls[0].args) > 1 else gt_calls[0].kwargs.get("args")
    assert args_for_gt is not None, "create_evaluator GT call missing args namespace"

    assert args_for_gt.hf is True, (
        f"args_for_gt.hf must be True (base backend is hf), got {args_for_gt.hf!r}. "
        f"Runner is incorrectly building args from the target backend."
    )
    assert args_for_gt.genai is False, (
        f"args_for_gt.genai must be False (base backend is hf), got {args_for_gt.genai!r}. "
        f"Runner is incorrectly building args from the target backend."
    )


def test_base_model_llamacpp_kwargs_passed(tmp_path: Path, mocked_runner_deps) -> None:
    """F4 — base-model load_model call must also forward llamacpp/gguf kwargs.

    Mirrors F2 for the GT-generation path: the base model load drops the same
    kwargs, so a llamacpp base model gets loaded through the wrong code path.
    """
    data = copy.deepcopy(_BASE_SCENARIO)
    data["models"] = {
        "base": {"path": "/base_weights.gguf", "backend": "llamacpp", "gguf_file": "/base_weights.gguf"},
        "t1": {"path": "/ov/t1", "backend": "genai"},
    }
    data["tasks"] = [
        {
            "id": "chat",
            "type": "text",
            "base": "base",
            "targets": ["t1"],
            "dataset": "ds",
        }
    ]
    scenario = Scenario.model_validate(data)

    runner = ScenarioRunner(scenario, output_dir=tmp_path)
    runner.run()

    load_model_mock = mocked_runner_deps["load_model"]
    assert load_model_mock.call_args_list, "load_model was never called"

    base_calls = [
        call
        for call in load_model_mock.call_args_list
        if "/base_weights.gguf" in (call.args[1] if len(call.args) > 1 else call.kwargs.get("model_id", ""))
    ]
    assert base_calls, f"No load_model call for the base llamacpp model. Calls: {load_model_mock.call_args_list}"
    base_call = base_calls[0]

    use_llamacpp = base_call.kwargs.get("use_llamacpp")
    if use_llamacpp is None and len(base_call.args) >= 7:
        use_llamacpp = base_call.args[6]
    assert use_llamacpp is True, (
        f"Base load_model must be called with use_llamacpp=True. args={base_call.args} kwargs={base_call.kwargs}"
    )

    gguf_file = base_call.kwargs.get("gguf_file")
    assert gguf_file == "/base_weights.gguf", (
        f"Base load_model must forward gguf_file kwarg. args={base_call.args} kwargs={base_call.kwargs}"
    )


def test_run_continues_after_task_exception(tmp_path: Path) -> None:
    """F6 — a task that raises must not abort subsequent tasks.

    Without try/except in `run()`, an OOM on task_a's target load propagates and
    task_b never executes, leaving an incomplete result store with no signal to
    the caller about which tasks completed.
    """
    data = copy.deepcopy(_BASE_SCENARIO)
    data["tasks"] = [
        {
            "id": "task_a",
            "type": "text",
            "base": "base",
            "targets": ["t1"],
            "dataset": "ds",
        },
        {
            "id": "task_b",
            "type": "text",
            "base": "base",
            "targets": ["t2"],
            "dataset": "ds",
        },
    ]
    scenario = Scenario.model_validate(data)

    def _selective_load(model_type, model_id, *args, **kwargs):
        # Only the target for task_a explodes — base loads and task_b's target
        # must still proceed.
        if model_id == "/ov/t1":
            raise RuntimeError("OOM")
        return MagicMock(name="LoadedModel")

    with (
        patch("whowhatbench.scenario.runner.load_model", side_effect=_selective_load),
        patch("whowhatbench.scenario.runner.create_evaluator") as create_evaluator_mock,
    ):
        create_evaluator_mock.return_value = _make_evaluator_mock()
        runner = ScenarioRunner(scenario, output_dir=tmp_path)
        store = runner.run()

    task_ids = {r.task_id for r in store.all_results()}
    assert "task_b" in task_ids, f"task_b must run even after task_a's load_model raised. Got results for: {task_ids}"


def test_csv_loaded_once_per_task_not_per_target(tmp_path: Path, mocked_runner_deps) -> None:
    """F8 — CSV dataset materialisation must be hoisted out of the per-target loop.

    A 1-task / 3-target scenario should read the CSV exactly once. Re-reading
    inside `_run_one` triples disk I/O and parse cost without changing the
    test_data dict.
    """
    csv_path = tmp_path / "prompts.csv"
    csv_path.write_text("text\np1\np2\np3\n", encoding="utf-8")

    data = copy.deepcopy(_BASE_SCENARIO)
    data["models"] = {
        "base": {"path": "org/base", "backend": "hf"},
        "t1": {"path": "/ov/t1", "backend": "genai"},
        "t2": {"path": "/ov/t2", "backend": "genai"},
        "t3": {"path": "/ov/t3", "backend": "genai"},
    }
    data["datasets"] = {"ds": {"type": "csv", "path": str(csv_path), "field": "text"}}
    data["tasks"] = [
        {
            "id": "chat",
            "type": "text",
            "base": "base",
            "targets": ["t1", "t2", "t3"],
            "dataset": "ds",
        }
    ]
    scenario = Scenario.model_validate(data)

    with patch("whowhatbench.scenario.runner.pd.read_csv", wraps=pd.read_csv) as read_csv_spy:
        runner = ScenarioRunner(scenario, output_dir=tmp_path)
        runner.run()

    csv_calls = [call for call in read_csv_spy.call_args_list if str(csv_path) in str(call)]
    assert len(csv_calls) == 1, (
        f"CSV dataset must be materialised once per task, not once per target. Got {len(csv_calls)} calls: {csv_calls}"
    )


def test_csv_dataset_bounded_by_num_samples(tmp_path: Path, mocked_runner_deps) -> None:
    """F11 — CSV loader must honour task.num_samples to avoid reading huge files.

    Loading every row of a 5-row CSV when only 2 are needed is wasteful at
    small scale and OOM-prone at production scale (millions of rows).
    """
    csv_path = tmp_path / "prompts.csv"
    csv_path.write_text("text\np1\np2\np3\np4\np5\n", encoding="utf-8")

    data = copy.deepcopy(_BASE_SCENARIO)
    data["datasets"] = {"ds": {"type": "csv", "path": str(csv_path), "field": "text"}}
    data["tasks"] = [
        {
            "id": "chat",
            "type": "text",
            "base": "base",
            "targets": ["t1"],
            "dataset": "ds",
            "num_samples": 2,
        }
    ]
    scenario = Scenario.model_validate(data)

    with patch("whowhatbench.scenario.runner.pd.read_csv", wraps=pd.read_csv) as read_csv_spy:
        runner = ScenarioRunner(scenario, output_dir=tmp_path)
        runner.run()

    bounded_call = next(
        (call for call in read_csv_spy.call_args_list if str(csv_path) in str(call)),
        None,
    )
    assert bounded_call is not None, "pd.read_csv was never called for the CSV dataset"

    nrows = bounded_call.kwargs.get("nrows")
    assert nrows == 2, (
        f"CSV loader must pass nrows=task.num_samples to bound memory. "
        f"Got kwargs={bounded_call.kwargs} args={bounded_call.args}"
    )


def test_gt_cache_key_computed_once_per_task(tmp_path: Path, mocked_runner_deps) -> None:
    """F16 — GT cache key must be computed once per task, not per (task, target).

    The key only depends on base-model and dataset fields shared across targets,
    so recomputing it 3x for 3 targets is wasted work and signals that
    `_prepare_gt` is being invoked from the inner per-target loop instead of
    being lifted out.
    """
    data = copy.deepcopy(_BASE_SCENARIO)
    data["models"] = {
        "base": {"path": "org/base", "backend": "hf"},
        "t1": {"path": "/ov/t1", "backend": "genai"},
        "t2": {"path": "/ov/t2", "backend": "genai"},
        "t3": {"path": "/ov/t3", "backend": "genai"},
    }
    data["tasks"] = [
        {
            "id": "chat",
            "type": "text",
            "base": "base",
            "targets": ["t1", "t2", "t3"],
            "dataset": "ds",
        }
    ]
    scenario = Scenario.model_validate(data)

    with patch(
        "whowhatbench.scenario.runner.GTCache.key",
        autospec=True,
        side_effect=lambda self, *a, **kw: "deadbeefdeadbeef",
    ) as key_spy:
        runner = ScenarioRunner(scenario, output_dir=tmp_path)
        runner.run()

    assert key_spy.call_count == 1, (
        f"GTCache.key must be called once per task, not once per target. "
        f"Got {key_spy.call_count} calls. _prepare_gt is likely inside the per-target loop."
    )


def test_normalize_metrics_empty_list_no_crash() -> None:
    """F21 — _normalize_metrics must handle empty-list values without IndexError.

    A naive `v[0] if isinstance(v, list) else v` raises on `[]`. The function
    should either drop the key or keep the empty list — both are acceptable as
    long as it does not crash.
    """
    try:
        result = _normalize_metrics({"similarity_mean": []})
    except IndexError as exc:
        pytest.fail(f"_normalize_metrics crashed on empty list value: {exc!r}")

    # Either the key is dropped or its value is the empty list — both fine.
    if "similarity_mean" in result:
        assert result["similarity_mean"] == [], (
            f"empty-list value should be preserved or dropped, got {result['similarity_mean']!r}"
        )
