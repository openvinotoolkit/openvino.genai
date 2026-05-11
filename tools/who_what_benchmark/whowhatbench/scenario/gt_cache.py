# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from whowhatbench.scenario.schema import DatasetConfig, ModelConfig, Scenario, TaskConfig


class GTCache:
    """Content-addressed cache for ground-truth CSVs keyed by GT-affecting params."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        task: TaskConfig,
        base_model: ModelConfig,
        dataset: DatasetConfig,
        scenario: Scenario,
    ) -> str:
        """Compute the 16-char hex SHA-256 key over GT-affecting fields only."""
        components: list[str] = [
            base_model.path,
            base_model.backend.value,
            base_model.tokenizer or "",
            task.type,
            _dataset_repr(dataset),
            str(task.num_samples or ""),
            str(task.seed if task.seed is not None else scenario.defaults.seed),
            str(task.generation.max_new_tokens),
            str(task.generation.long_prompt),
            dataset.language,
        ]
        key_input = "\n".join(components)
        return hashlib.sha256(key_input.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Path]:
        p = self._dir / f"{key}.csv"
        return p if p.exists() else None

    def put(self, key: str, gt_path: Path) -> None:
        dest = self._dir / f"{key}.csv"
        gt_path = Path(gt_path)
        # Skip the copy if the caller already wrote into the cache directory directly
        # (avoids SameFileError on shutil.copy2).
        if gt_path.resolve() != dest.resolve():
            shutil.copy2(gt_path, dest)
        self._write_meta(key, gt_path)

    def meta_path(self, key: str) -> Path:
        return self._dir / f"{key}.meta.json"

    def _write_meta(self, key: str, source_path: Path) -> None:
        meta: dict[str, str] = {
            "key": key,
            "source_path": str(source_path),
        }
        self.meta_path(key).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _dataset_repr(dataset: DatasetConfig) -> str:
    if dataset.type.value == "builtin":
        return f"builtin:{dataset.language}"
    if dataset.type.value == "huggingface":
        return f"hf:{dataset.path}:{dataset.name or ''}:{dataset.split}:{dataset.field}"
    if dataset.type.value == "csv":
        return f"csv:{dataset.path}:{dataset.field}"
    if dataset.type.value == "inline":
        inline_data = dataset.prompts or dataset.passages or dataset.chats or []
        content = "\n".join(str(x) for x in inline_data)
        return f"inline:{hashlib.sha256(content.encode()).hexdigest()[:8]}"
    return f"unknown:{dataset.type.value}"
