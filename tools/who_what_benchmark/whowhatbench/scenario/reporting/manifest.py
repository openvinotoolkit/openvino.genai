# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from whowhatbench.scenario.result_store import ResultStore
    from whowhatbench.scenario.schema import Scenario


def build_manifest(
    scenario: "Scenario",
    scenario_path: Optional[Path],
    store: "ResultStore",
) -> dict[str, Any]:
    """Build the run_manifest.json dict."""
    return {
        "schema_version": 1,
        "scenario": _scenario_block(scenario, scenario_path),
        "env": _env_block(),
        "tasks_timing": _timing_block(store),
        "gt_cache_stats": _cache_stats(store),
    }


def _scenario_block(scenario: "Scenario", path: Optional[Path]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "name": scenario.name,
        "description": scenario.description,
    }
    if path is not None:
        raw = Path(path).read_bytes()
        block["sha256"] = hashlib.sha256(raw).hexdigest()
        block["path"] = str(path)
    else:
        block["sha256"] = None
        block["path"] = None
    return block


def _env_block() -> dict[str, Optional[str]]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "wwb": _pkg_version("whowhatbench"),
        "transformers": _pkg_version("transformers"),
        "datasets": _pkg_version("datasets"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "openvino": _pkg_version("openvino"),
        "openvino_genai": _pkg_version("openvino-genai"),
        "git_commit": _git_commit(),
    }


def _pkg_version(pkg: str) -> Optional[str]:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.SubprocessError, OSError):
        return None


def _timing_block(store: "ResultStore") -> list[dict[str, Any]]:
    return [
        {
            "task_id": r.task_id,
            "target_id": r.target_id,
            "runtime_s": round(r.runtime_s, 3),
        }
        for r in store.all_results()
    ]


def _cache_stats(store: "ResultStore") -> dict[str, int]:
    results = store.all_results()
    hits = sum(1 for r in results if r.gt_cache_hit)
    misses = len(results) - hits
    return {"hits": hits, "misses": misses}
