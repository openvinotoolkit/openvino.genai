# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, PackageLoader, select_autoescape

if TYPE_CHECKING:
    from whowhatbench.scenario.result_store import ResultStore, TaskResult
    from whowhatbench.scenario.schema import Scenario


# Headline metric per task type — used for the summary table.
# Keys MUST match the dict keys actually produced by each evaluator's
# scalar metric output (see ``whowhat_metrics.py`` and
# ``speech_generation_evaluator.py``). All current evaluators emit a single
# scalar under ``similarity`` except speech-generation, which emits
# ``overall similarity`` (with a space — preserved verbatim from
# ``OVERALL_SCORE_COL``).
_HEADLINE_METRIC: dict[str, str] = {
    "text": "similarity",
    "text-chat": "similarity",
    "visual-text": "similarity",
    "visual-text-chat": "similarity",
    "visual-video-text": "similarity",
    "text-to-image": "similarity",
    "image-to-image": "similarity",
    "image-inpainting": "similarity",
    "text-to-video": "similarity",
    "text-embedding": "similarity",
    "text-reranking": "similarity",
    "speech-generation": "overall similarity",
}

_TEMPLATE_NAME = "report.md.j2"


@lru_cache(maxsize=1)
def _get_environment() -> Environment:
    """Return a cached jinja2 Environment loading templates from the package.

    Autoescape is disabled by default for Markdown output; HTML escaping is
    applied explicitly via the ``| e`` filter at the call site where it
    matters (notably ``scenario.description``).
    """
    return Environment(
        loader=PackageLoader("whowhatbench", "scenario/reporting/templates"),
        autoescape=select_autoescape(disabled_extensions=("md", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def render_markdown(store: ResultStore, scenario: Scenario) -> str:
    """Return the full report.md contents as a string."""
    context = _build_context(store, scenario)
    template = _get_environment().get_template(_TEMPLATE_NAME)
    return template.render(**context)


def _build_context(store: ResultStore, scenario: Scenario) -> dict[str, Any]:
    """Build the jinja2 render context from store + scenario.

    All ordering, grouping, and metric formatting happen here so the template
    stays as a pure layout file (no Python expressions beyond simple loops).
    """
    results = store.all_results()
    group_by = scenario.report.group_by

    if group_by == "task":
        summary_rows = [_view_row(r, scenario) for r in results]
        task_groups = _group_by_task(results, scenario)
        target_groups = None
    else:
        sorted_results = sorted(results, key=lambda r: (r.target_id, r.task_id))
        summary_rows = [_view_row(r, scenario) for r in sorted_results]
        task_groups = None
        target_groups = _group_by_target(results, scenario)

    return {
        "scenario_name": scenario.name,
        "description": scenario.description,
        "group_by": group_by,
        "summary_rows": summary_rows,
        "task_groups": task_groups,
        "target_groups": target_groups,
    }


def _view_row(r: TaskResult, scenario: Scenario) -> dict[str, Any]:
    """Convert a TaskResult into a flat dict ready for template rendering."""
    task_cfg = next((t for t in scenario.tasks if t.id == r.task_id), None)
    task_type = task_cfg.type if task_cfg else "unknown"
    metric_name = _HEADLINE_METRIC.get(task_type, "similarity")
    return {
        "task_id": r.task_id,
        "task_type": task_type,
        "target_id": r.target_id,
        "metric_name": metric_name,
        "metric_value": _format_metric(r.metrics, metric_name, fallback_key="similarity"),
        "n_samples": len(r.per_question) if r.per_question else "—",
        "runtime_s": f"{r.runtime_s:.1f}",
    }


def _group_by_task(results: list[TaskResult], scenario: Scenario) -> list[dict[str, Any]]:
    """Group results by task_id (insertion order preserved)."""
    task_ids = list(dict.fromkeys(r.task_id for r in results))
    groups: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_results = [r for r in results if r.task_id == task_id]
        task_cfg = next((t for t in scenario.tasks if t.id == task_id), None)
        task_type = task_cfg.type if task_cfg else "unknown"
        metric_name = _HEADLINE_METRIC.get(task_type, "similarity")
        groups.append(
            {
                "task_id": task_id,
                "task_type": task_type,
                "metric_name": metric_name,
                "results": [
                    {
                        "target_id": r.target_id,
                        "metric_value": _format_metric(r.metrics, metric_name, fallback_key="similarity"),
                        "runtime_s": f"{r.runtime_s:.1f}",
                    }
                    for r in task_results
                ],
            }
        )
    return groups


def _group_by_target(results: list[TaskResult], scenario: Scenario) -> list[dict[str, Any]]:
    """Group results by target_id (alphabetical)."""
    target_ids = sorted({r.target_id for r in results})
    groups: list[dict[str, Any]] = []
    for target_id in target_ids:
        target_results = [r for r in results if r.target_id == target_id]
        rendered = []
        for r in target_results:
            task_cfg = next((t for t in scenario.tasks if t.id == r.task_id), None)
            task_type = task_cfg.type if task_cfg else "unknown"
            metric_name = _HEADLINE_METRIC.get(task_type, "similarity")
            rendered.append(
                {
                    "task_id": r.task_id,
                    "task_type": task_type,
                    "metric_name": metric_name,
                    "metric_value": _format_metric(r.metrics, metric_name, fallback_key="similarity"),
                    "runtime_s": f"{r.runtime_s:.1f}",
                }
            )
        groups.append({"target_id": target_id, "results": rendered})
    return groups


def _format_metric(metrics: dict[str, Any], key: str, fallback_key: str | None = None) -> str:
    value = metrics.get(key)
    if value is None and fallback_key is not None:
        value = metrics.get(fallback_key)
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
