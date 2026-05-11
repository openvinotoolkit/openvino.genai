# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from whowhatbench.scenario.result_store import ResultStore
    from whowhatbench.scenario.schema import Scenario


def write_reports(
    store: ResultStore,
    scenario: Scenario,
    output_dir: Path,
    scenario_path: Optional[Path] = None,
) -> None:
    """Write report.md, report.json, and run_manifest.json to output_dir."""
    from whowhatbench.scenario.reporting.json_report import render_json
    from whowhatbench.scenario.reporting.manifest import build_manifest
    from whowhatbench.scenario.reporting.markdown import render_markdown

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(scenario, scenario_path, store)

    md = render_markdown(store, scenario)
    (output_dir / "report.md").write_text(md, encoding="utf-8")

    report = render_json(store, scenario, manifest)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
