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
        return val

    text = re.sub(r"\$\{env\.([^}]+)\}", _replace_env, text)

    name_match = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
    scenario_name = name_match.group(1).strip().strip("\"'") if name_match else "unknown"
    text = text.replace("${scenario.name}", scenario_name)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    text = text.replace("${timestamp}", ts)

    return text


def _merge_defaults(scenario: Scenario) -> None:
    d = scenario.defaults
    for task in scenario.tasks:
        if task.num_samples is None and d.num_samples is not None:
            task.num_samples = d.num_samples
        if task.seed is None:
            task.seed = d.seed
        if task.data_encoder is None:
            task.data_encoder = d.data_encoder
