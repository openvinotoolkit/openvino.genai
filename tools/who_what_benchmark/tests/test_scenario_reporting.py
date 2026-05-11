# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RED-phase tests for ``whowhatbench.scenario.reporting``.

The reporting modules do not exist yet — these tests are expected to fail
(import errors / attribute errors) until production code is implemented.
The assertions encode the contract from the implementation plan.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from whowhatbench.scenario.reporting import write_reports
from whowhatbench.scenario.reporting.json_report import render_json
from whowhatbench.scenario.reporting.manifest import build_manifest
from whowhatbench.scenario.reporting.markdown import render_markdown
from whowhatbench.scenario.result_store import ResultStore, TaskResult
from whowhatbench.scenario.schema import Scenario

_BASE_SCENARIO: dict[str, Any] = {
    "schema_version": 1,
    "name": "test-report",
    "description": "Test scenario for report tests",
    "models": {
        "base": {"path": "org/base", "backend": "hf"},
        "llama_int4": {"path": "/ov/int4", "backend": "genai"},
        "llama_int8": {"path": "/ov/int8", "backend": "genai"},
    },
    "datasets": {"ds": {"type": "builtin"}},
    "tasks": [
        {
            "id": "chat_quality",
            "type": "text-chat",
            "base": "base",
            "targets": ["llama_int4", "llama_int8"],
            "dataset": "ds",
        }
    ],
}


@pytest.fixture
def sample_scenario() -> Scenario:
    return Scenario.model_validate(copy.deepcopy(_BASE_SCENARIO))


@pytest.fixture
def sample_store(tmp_path: Path) -> ResultStore:
    store = ResultStore()
    store.add(
        TaskResult(
            task_id="chat_quality",
            target_id="llama_int4",
            metrics={"similarity_mean": 0.912, "fdt_mean": 0.041, "sdt_mean": 0.087},
            per_question=[
                {"prompt": "q1", "similarity": 0.91},
                {"prompt": "q2", "similarity": 0.92},
            ],
            runtime_s=72.3,
            output_dir=tmp_path / "tasks" / "chat_quality" / "llama_int4",
            gt_cache_hit=False,
        )
    )
    store.add(
        TaskResult(
            task_id="chat_quality",
            target_id="llama_int8",
            metrics={"similarity_mean": 0.957, "fdt_mean": 0.018, "sdt_mean": 0.039},
            per_question=[{"prompt": "q1", "similarity": 0.96}],
            runtime_s=78.1,
            output_dir=tmp_path / "tasks" / "chat_quality" / "llama_int8",
            gt_cache_hit=True,
        )
    )
    return store


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_markdown_contains_summary_table(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    md = render_markdown(sample_store, sample_scenario)
    assert "chat_quality" in md
    assert "llama_int4" in md
    # 0.912 should be reported with at least 3 significant digits.
    assert "0.912" in md


def test_render_markdown_contains_scenario_name(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    md = render_markdown(sample_store, sample_scenario)
    assert "test-report" in md


def test_render_markdown_contains_description(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    md = render_markdown(sample_store, sample_scenario)
    assert "Test scenario for report tests" in md


def test_render_markdown_both_targets_in_table(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    md = render_markdown(sample_store, sample_scenario)
    assert "llama_int4" in md
    assert "llama_int8" in md


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


def test_render_json_has_tasks_key(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    payload = render_json(sample_store, sample_scenario, {})
    assert isinstance(payload["tasks"], list)
    assert len(payload["tasks"]) > 0


def test_render_json_task_has_results(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    payload = render_json(sample_store, sample_scenario, {})
    first_task = payload["tasks"][0]
    assert isinstance(first_task["results"], list)
    assert len(first_task["results"]) == 2


def test_render_json_metrics_correct(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    payload = render_json(sample_store, sample_scenario, {})
    first_task = payload["tasks"][0]

    int4_result = next(r for r in first_task["results"] if r["target_id"] == "llama_int4")
    assert int4_result["metrics"]["similarity_mean"] == pytest.approx(0.912)


def test_render_json_scenario_info(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    payload = render_json(sample_store, sample_scenario, {})
    assert payload["scenario"]["name"] == "test-report"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_build_manifest_has_required_keys(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    manifest = build_manifest(sample_scenario, None, sample_store)

    for key in ("scenario", "env", "tasks_timing"):
        assert key in manifest, f"missing key '{key}' in manifest: {sorted(manifest)}"


def test_build_manifest_scenario_sha256(sample_store: ResultStore, sample_scenario: Scenario, tmp_path: Path) -> None:
    scenario_path = tmp_path / "s.yaml"
    scenario_path.write_text(yaml.safe_dump(_BASE_SCENARIO), encoding="utf-8")

    manifest = build_manifest(sample_scenario, scenario_path, sample_store)
    sha = manifest["scenario"]["sha256"]
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_build_manifest_python_version(sample_store: ResultStore, sample_scenario: Scenario) -> None:
    manifest = build_manifest(sample_scenario, None, sample_store)
    expected = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert expected in manifest["env"]["python"]


def test_build_manifest_scenario_has_domain_keys(sample_store, sample_scenario) -> None:
    """Manifest scenario block must contain identity + hash fields."""
    manifest = build_manifest(sample_scenario, None, sample_store)
    scenario_block = manifest["scenario"]
    assert "name" in scenario_block
    assert "sha256" in scenario_block or "hash" in scenario_block
    assert scenario_block.get("name") == "test-report"


def test_build_manifest_env_has_python_and_wwb(sample_store, sample_scenario) -> None:
    """Manifest env block must expose python version and wwb version."""
    manifest = build_manifest(sample_scenario, None, sample_store)
    env = manifest["env"]
    assert "python" in env
    assert "wwb" in env or "whowhatbench" in env


# ---------------------------------------------------------------------------
# Aggregate report writer
# ---------------------------------------------------------------------------


def test_write_reports_creates_files(sample_store: ResultStore, sample_scenario: Scenario, tmp_path: Path) -> None:
    write_reports(sample_store, sample_scenario, tmp_path)

    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "run_manifest.json").is_file()


def test_write_reports_json_is_valid(sample_store: ResultStore, sample_scenario: Scenario, tmp_path: Path) -> None:
    write_reports(sample_store, sample_scenario, tmp_path)

    raw = (tmp_path / "report.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict)


def test_empty_store_produces_valid_reports(sample_scenario: Scenario, tmp_path: Path) -> None:
    empty_store = ResultStore()
    write_reports(empty_store, sample_scenario, tmp_path)

    md_text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert isinstance(md_text, str)
    assert len(md_text) > 0


def test_group_by_target_changes_table_order(sample_scenario, tmp_path) -> None:
    """group_by=target must produce a different ordering than the default task ordering."""
    # Build a scenario where natural insertion order is z_model before a_model
    scenario_data = {
        "schema_version": 1,
        "name": "order-test",
        "models": {
            "base": {"path": "org/base", "backend": "hf"},
            "z_model": {"path": "/ov/z", "backend": "genai"},
            "a_model": {"path": "/ov/a", "backend": "genai"},
        },
        "datasets": {"ds": {"type": "builtin"}},
        "tasks": [
            {
                "id": "task1",
                "type": "text",
                "base": "base",
                "targets": ["z_model", "a_model"],
                "dataset": "ds",
            }
        ],
        "report": {"group_by": "target"},
    }
    scenario = Scenario.model_validate(scenario_data)

    store = ResultStore()
    # Insert z_model first so natural order would put it before a_model
    store.add(
        TaskResult(
            task_id="task1",
            target_id="z_model",
            metrics={"similarity_mean": 0.8},
            per_question=[],
            runtime_s=1.0,
            output_dir=tmp_path / "z",
            gt_cache_hit=False,
        )
    )
    store.add(
        TaskResult(
            task_id="task1",
            target_id="a_model",
            metrics={"similarity_mean": 0.9},
            per_question=[],
            runtime_s=1.0,
            output_dir=tmp_path / "a",
            gt_cache_hit=False,
        )
    )

    md = render_markdown(store, scenario)
    z_pos = md.find("z_model")
    a_pos = md.find("a_model")
    assert z_pos != -1 and a_pos != -1
    # With group_by=target, a_model should appear before z_model
    # (alphabetical or insertion-order within group — either way different from default)
    assert a_pos < z_pos, (
        f"Expected a_model (pos {a_pos}) before z_model (pos {z_pos}) with group_by=target, but order was reversed"
    )


def test_render_markdown_escapes_html_in_description(tmp_path: Path) -> None:
    """Scenario.description must be escaped before being written to report.md.

    A scenario author (or someone interpolating values via ``${env.*}``) should
    not be able to inject raw HTML/script tags into the rendered Markdown.
    Reports are commonly rendered by web tooling (GitHub, GitLab, dashboards),
    so unescaped ``<script>`` and ``&`` are a stored-XSS vector.
    """
    scenario_data = copy.deepcopy(_BASE_SCENARIO)
    scenario_data["description"] = "<script>alert(1)</script> & foo"
    scenario = Scenario.model_validate(scenario_data)

    store = ResultStore()
    store.add(
        TaskResult(
            task_id="chat_quality",
            target_id="llama_int4",
            metrics={"similarity_mean": 0.9},
            per_question=[{"prompt": "q1", "similarity": 0.9}],
            runtime_s=1.0,
            output_dir=tmp_path / "tasks" / "chat_quality" / "llama_int4",
            gt_cache_hit=False,
        )
    )

    md = render_markdown(store, scenario)

    assert "<script>" not in md, "Raw <script> tag must be escaped in rendered markdown"
    assert "</script>" not in md, "Raw </script> tag must be escaped in rendered markdown"
    assert "&amp;" in md or "&#38;" in md, "Ampersand in description must be HTML-escaped"
