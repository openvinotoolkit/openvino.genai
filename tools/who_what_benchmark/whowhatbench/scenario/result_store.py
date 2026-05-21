# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TaskResult:
    """Single (task, target) evaluation outcome captured by the runner."""

    task_id: str
    target_id: str
    metrics: dict[str, Any]
    per_question: list[dict[str, Any]]
    runtime_s: float
    output_dir: Path
    gt_cache_hit: bool


class ResultStore:
    """In-memory collector of TaskResults with deferred CSV serialisation."""

    def __init__(self) -> None:
        self._results: list[TaskResult] = []

    def add(self, result: TaskResult) -> None:
        self._results.append(result)

    def all_results(self) -> list[TaskResult]:
        return list(self._results)

    def flush_csvs(self) -> None:
        for r in self._results:
            r.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([r.metrics]).to_csv(r.output_dir / "metrics.csv", index=False)
            pq = pd.DataFrame(r.per_question) if r.per_question else pd.DataFrame()
            pq.to_csv(r.output_dir / "metrics_per_question.csv", index=False)
