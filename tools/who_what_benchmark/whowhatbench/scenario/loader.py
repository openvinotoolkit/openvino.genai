# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import re
from pathlib import Path
from typing import Union

import yaml

from whowhatbench.scenario.schema import Scenario


def load_scenario(path: Union[str, Path]) -> Scenario:
    """Load and validate a YAML scenario file, applying defaults and interpolation.

    Steps:
        1. Read raw YAML text
        2. Interpolate ``${env.*}``, ``${scenario.name}``, ``${timestamp}``
        3. Parse as YAML
        4. Validate via Pydantic ``Scenario`` schema
        5. Apply scenario-level defaults to per-task fields
    """
    raw = Path(path).read_text(encoding="utf-8")
    interpolated = _interpolate(raw)
    data = yaml.safe_load(interpolated)
    scenario = Scenario.model_validate(data)
    _merge_defaults(scenario)
    return scenario


def _interpolate(text: str) -> str:
    def _replace_env(m: re.Match[str]) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            raise ValueError(f"Environment variable {var!r} referenced in scenario but not set")
        # Reject values that could inject new YAML structure when substituted into raw text
        if any(c in val for c in ("\n", "\r", "\x00")):
            raise ValueError(
                f"Environment variable {var!r} contains control characters (newline, carriage return, or NUL) "
                f"which could corrupt the YAML structure when interpolated. "
                f"Ensure the variable contains only single-line plain text."
            )
        return val

    text = re.sub(r"\$\{env\.([^}]+)\}", _replace_env, text)

    name_match = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
    scenario_name = name_match.group(1).strip().strip("\"'") if name_match else "unknown"
    text = text.replace("${scenario.name}", scenario_name)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    text = text.replace("${timestamp}", ts)

    return text


def _merge_defaults(scenario: Scenario) -> None:
    defaults = scenario.defaults
    for task in scenario.tasks:
        if task.num_samples is None and defaults.num_samples is not None:
            task.num_samples = defaults.num_samples
        if task.seed is None:
            task.seed = defaults.seed
        # Propagate the effective seed into GenerationParams so callers can use
        # task.generation.seed as the single authoritative seed value.
        if "seed" not in task.generation.model_fields_set:
            task.generation.seed = task.seed
        if task.data_encoder is None:
            task.data_encoder = defaults.data_encoder
