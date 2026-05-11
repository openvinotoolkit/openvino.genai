# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whowhatbench.scenario.result_store import ResultStore, TaskResult
    from whowhatbench.scenario.schema import Scenario


# Headline metric per task type — used for the summary table
_HEADLINE_METRIC: dict[str, str] = {
    "text": "similarity_mean",
    "text-chat": "similarity_mean",
    "visual-text": "similarity_mean",
    "visual-text-chat": "similarity_mean",
    "visual-video-text": "similarity_mean",
    "text-to-image": "image_similarity_mean",
    "image-to-image": "image_similarity_mean",
    "image-inpainting": "image_similarity_mean",
    "text-to-video": "video_similarity_mean",
    "text-embedding": "embeds_similarity_mean",
    "text-reranking": "ndcg_mean",
    "speech-generation": "overall_similarity_mean",
}


def render_markdown(store: ResultStore, scenario: Scenario) -> str:
    """Return the full report.md contents as a string."""
    results = store.all_results()
    lines: list[str] = []

    lines.append(f"# WWB Scenario Report — {scenario.name}")
    lines.append("")
    if scenario.description:
        lines.append(f"> {scenario.description}")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    group_by = scenario.report.group_by

    if group_by == "task":
        lines.extend(_summary_table_by_task(results, scenario))
    else:
        lines.extend(_summary_table_by_target(results, scenario))

    lines.append("")

    if group_by == "task":
        task_ids = list(dict.fromkeys(r.task_id for r in results))
        for task_id in task_ids:
            task_results = [r for r in results if r.task_id == task_id]
            lines.extend(_task_detail_section(task_id, task_results, scenario))
    else:
        target_ids = sorted({r.target_id for r in results})
        for target_id in target_ids:
            target_results = [r for r in results if r.target_id == target_id]
            lines.extend(_target_detail_section(target_id, target_results, scenario))

    lines.append("---")
    lines.append("")
    lines.append("*See `run_manifest.json` for full environment and version details.*")
    lines.append("")

    return "\n".join(lines)


def _summary_table_by_task(results: list[TaskResult], scenario: Scenario) -> list[str]:
    lines: list[str] = []
    lines.append("| task | type | target | headline_metric | value | n_samples | runtime_s |")
    lines.append("|------|------|--------|-----------------|-------|-----------|-----------|")
    for r in results:
        task_cfg = next((t for t in scenario.tasks if t.id == r.task_id), None)
        task_type = task_cfg.type if task_cfg else "unknown"
        metric_name = _HEADLINE_METRIC.get(task_type, "similarity_mean")
        value = _format_metric(r.metrics, metric_name, fallback_key="similarity_mean")
        n = len(r.per_question) if r.per_question else "—"
        rt = f"{r.runtime_s:.1f}"
        lines.append(f"| {r.task_id} | {task_type} | {r.target_id} | {metric_name} | {value} | {n} | {rt} |")
    return lines


def _summary_table_by_target(results: list[TaskResult], scenario: Scenario) -> list[str]:
    lines: list[str] = []
    lines.append("| target | task | type | headline_metric | value | n_samples | runtime_s |")
    lines.append("|--------|------|------|-----------------|-------|-----------|-----------|")
    sorted_results = sorted(results, key=lambda r: (r.target_id, r.task_id))
    for r in sorted_results:
        task_cfg = next((t for t in scenario.tasks if t.id == r.task_id), None)
        task_type = task_cfg.type if task_cfg else "unknown"
        metric_name = _HEADLINE_METRIC.get(task_type, "similarity_mean")
        value = _format_metric(r.metrics, metric_name, fallback_key="similarity_mean")
        n = len(r.per_question) if r.per_question else "—"
        rt = f"{r.runtime_s:.1f}"
        lines.append(f"| {r.target_id} | {r.task_id} | {task_type} | {metric_name} | {value} | {n} | {rt} |")
    return lines


def _task_detail_section(task_id: str, task_results: list[TaskResult], scenario: Scenario) -> list[str]:
    lines: list[str] = []
    task_cfg = next((t for t in scenario.tasks if t.id == task_id), None)
    task_type = task_cfg.type if task_cfg else "unknown"
    lines.append(f"## Task: {task_id} ({task_type})")
    lines.append("")
    metric_name = _HEADLINE_METRIC.get(task_type, "similarity_mean")
    lines.append(f"| target | {metric_name} | runtime_s |")
    lines.append(f"|--------|{'-' * (len(metric_name) + 2)}|-----------|")
    for r in task_results:
        value = _format_metric(r.metrics, metric_name)
        lines.append(f"| {r.target_id} | {value} | {r.runtime_s:.1f} |")
    lines.append("")
    lines.append(f"*Worst examples: see `tasks/{task_id}/<target>/metrics_per_question.csv`*")
    lines.append("")
    return lines


def _target_detail_section(target_id: str, target_results: list[TaskResult], scenario: Scenario) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Target: {target_id}")
    lines.append("")
    for r in target_results:
        task_cfg = next((t for t in scenario.tasks if t.id == r.task_id), None)
        task_type = task_cfg.type if task_cfg else "unknown"
        metric_name = _HEADLINE_METRIC.get(task_type, "similarity_mean")
        value = _format_metric(r.metrics, metric_name)
        lines.append(f"- **{r.task_id}** ({task_type}): {metric_name} = {value}, runtime = {r.runtime_s:.1f}s")
    lines.append("")
    return lines


def _format_metric(metrics: dict, key: str, fallback_key: str | None = None) -> str:
    value = metrics.get(key)
    if value is None and fallback_key is not None:
        value = metrics.get(fallback_key)
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
