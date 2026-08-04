"""Build filtered CI test matrices from smart-ci affected components.

Reads the ``AFFECTED_COMPONENTS`` JSON produced by the smart-ci action and writes one
JSON array per matrix definition file found under
``.github/workflows/test_matrices/<platform>/*.yml`` to ``GITHUB_OUTPUT``. Each output is
named after its file stem (e.g. ``wheel_tests``, ``samples_tests``) and contains only the
entries whose components are affected. ``<platform>`` is passed via ``--platform``.
"""
import argparse
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
    parser = argparse.ArgumentParser(description="Build filtered CI test matrices from smart-ci affected components.")
    parser.add_argument("--platform", required=True, help="Matrix subfolder to load definitions from (e.g. linux or windows).")
    platform = parser.parse_args().platform

    affected: dict[str, Any] = json.loads(os.environ.get("AFFECTED_COMPONENTS") or "{}")

    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for matrix_path in sorted((MATRICES_DIR / platform).glob("*.yml")):
            entries = filter_matrix(affected, yaml.safe_load(matrix_path.read_text(encoding="utf-8")))
            print_matrix(matrix_path.stem, entries)
            output.write(f"{matrix_path.stem}=" + json.dumps(entries, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
