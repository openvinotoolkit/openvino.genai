"""Build filtered CI test matrices from smart-ci affected components.

Reads the ``AFFECTED_COMPONENTS`` JSON produced by the smart-ci action and writes
``wheel_tests`` and ``samples_tests`` JSON arrays to ``GITHUB_OUTPUT`` containing only
the entries whose components are affected. The test definitions are loaded from
``.github/workflows/test_matrices/*.yml``.
"""
import json
import os
from pathlib import Path
from typing import Any

import yaml

MATRICES_DIR = Path(__file__).resolve().parents[1] / "workflows" / "test_matrices"

affected: dict[str, Any] = json.loads(os.environ.get("AFFECTED_COMPONENTS") or "{}")


def is_affected(component: str) -> bool:
    value = affected.get(component)
    if isinstance(value, dict):
        return bool(value.get("test"))
    return bool(value)


def load_matrix(name: str) -> list[dict[str, Any]]:
    with open(MATRICES_DIR / f"{name}.yml", encoding="utf-8") as matrix_file:
        return yaml.safe_load(matrix_file)


def filter_matrix(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    for entry in entries:
        if any(is_affected(component) for component in entry["components"]):
            matrix.append({key: value for key, value in entry.items() if key != "components"})
    return matrix


wheel_tests = load_matrix("wheel_tests")
samples_tests = load_matrix("samples_tests")

with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
    output.write("wheel_tests=" + json.dumps(filter_matrix(wheel_tests), separators=(",", ":")) + "\n")
    output.write("samples_tests=" + json.dumps(filter_matrix(samples_tests), separators=(",", ":")) + "\n")
