# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from whowhatbench.scenario.result_store import ResultStore, TaskResult
    from whowhatbench.scenario.schema import Scenario


def render_json(
    store: ResultStore,
    scenario: Scenario,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return the full report.json dict."""
    task_map: dict[str, list[TaskResult]] = {}
    for r in store.all_results():
        task_map.setdefault(r.task_id, []).append(r)

    tasks_out: list[dict[str, Any]] = []
    for task_cfg in scenario.tasks:
        results = task_map.get(task_cfg.id, [])
        task_out: dict[str, Any] = {
            "id": task_cfg.id,
            "type": task_cfg.type,
            "base": task_cfg.base,
            "dataset": task_cfg.dataset,
            "results": [],
        }
        for r in results:
            csv_base = f"tasks/{r.task_id}/{r.target_id}"
            task_out["results"].append(
                {
                    "target": r.target_id,
                    "metrics": r.metrics,
                    "n_samples": len(r.per_question),
                    "runtime_s": round(r.runtime_s, 3),
                    "gt_cache_hit": r.gt_cache_hit,
                    "csv_paths": {
                        "metrics": f"{csv_base}/metrics.csv",
                        "per_question": f"{csv_base}/metrics_per_question.csv",
                        "target": f"{csv_base}/target.csv",
                    },
                }
            )
        tasks_out.append(task_out)

    return {
        "schema_version": 1,
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
            "sha256": manifest.get("scenario", {}).get("sha256"),
            "path": manifest.get("scenario", {}).get("path"),
        },
        "env": manifest.get("env", {}),
        "tasks": tasks_out,
    }
