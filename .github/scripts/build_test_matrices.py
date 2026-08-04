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


def is_affected(affected: dict[str, Any], component: str) -> bool:
    value = affected.get(component)
    if isinstance(value, dict):
        return bool(value.get("test"))
    return bool(value)


def load_matrix(name: str) -> list[dict[str, Any]]:
    with open(MATRICES_DIR / f"{name}.yml", encoding="utf-8") as matrix_file:
        return yaml.safe_load(matrix_file)


def filter_matrix(affected: dict[str, Any], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matrix = []
    for entry in entries:
        if any(is_affected(affected, component) for component in entry["components"]):
            matrix.append({key: value for key, value in entry.items() if key != "components"})
    return matrix


def print_matrix(name: str, entries: list[dict[str, Any]]) -> None:
    print(f"\n{name}: {len(entries)} job(s)")
    for entry in entries:
        print(f"  - {entry['name']}")


def main() -> None:
    affected: dict[str, Any] = json.loads(os.environ.get("AFFECTED_COMPONENTS") or "{}")

    wheel_tests = filter_matrix(affected, load_matrix("wheel_tests"))
    samples_tests = filter_matrix(affected, load_matrix("samples_tests"))

    print_matrix("wheel_tests", wheel_tests)
    print_matrix("samples_tests", samples_tests)

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write("wheel_tests=" + json.dumps(wheel_tests, separators=(",", ":")) + "\n")
        output.write("samples_tests=" + json.dumps(samples_tests, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
