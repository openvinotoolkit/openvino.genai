# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from whowhatbench.scenario.loader import load_scenario

# Mirrors the fixture in test_scenario_schema.py; duplicated to keep loader
# tests self-contained when run in isolation.
MINIMAL_VALID = {
    "schema_version": 1,
    "name": "my-test",
    "models": {
        "base": {"path": "org/model", "backend": "hf"},
        "target": {"path": "/ov/model", "backend": "genai"},
    },
    "datasets": {
        "ds1": {"type": "builtin"},
    },
    "tasks": [
        {
            "id": "t1",
            "type": "text",
            "base": "base",
            "targets": ["target"],
            "dataset": "ds1",
        }
    ],
}


def _write_yaml(tmp_path: Path, data: dict, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_valid_yaml_file(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_VALID)
    scenario = load_scenario(path)
    assert scenario.name == "my-test"
    assert scenario.schema_version == 1
    assert scenario.tasks[0].id == "t1"


def test_env_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("WWB_MODEL_DIR", "/models")
    data = copy.deepcopy(MINIMAL_VALID)
    data["models"]["base"]["path"] = "${env.WWB_MODEL_DIR}/base"
    path = _write_yaml(tmp_path, data)

    scenario = load_scenario(path)
    assert scenario.models["base"].path == "/models/base"


def test_missing_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
    data = copy.deepcopy(MINIMAL_VALID)
    data["models"]["base"]["path"] = "${env.NONEXISTENT_VAR}/x"
    path = _write_yaml(tmp_path, data)

    with pytest.raises(ValueError) as exc_info:
        load_scenario(path)
    # The variable name should be mentioned in the error to aid debugging.
    assert "NONEXISTENT_VAR" in str(exc_info.value)


def test_scenario_name_interpolation(tmp_path):
    data = copy.deepcopy(MINIMAL_VALID)
    data["defaults"] = {"output_dir": "./_out/${scenario.name}"}
    path = _write_yaml(tmp_path, data)

    scenario = load_scenario(path)
    assert "${scenario.name}" not in scenario.defaults.output_dir
    assert "my-test" in scenario.defaults.output_dir


def test_timestamp_interpolation(tmp_path):
    data = copy.deepcopy(MINIMAL_VALID)
    data["defaults"] = {"output_dir": "${timestamp}/out"}
    path = _write_yaml(tmp_path, data)

    scenario = load_scenario(path)
    loaded_output_dir = scenario.defaults.output_dir

    # The placeholder must be replaced by a non-empty string matching a
    # timestamp-like format (digits + separators only, not arbitrary text)
    import re

    assert "${timestamp}" not in loaded_output_dir
    prefix = loaded_output_dir.split("/")[0]
    assert len(prefix) >= 8, f"Timestamp prefix too short: {prefix!r}"
    assert re.match(r"^[\d_T\-:Z+]+$", prefix), f"Timestamp prefix contains unexpected characters: {prefix!r}"


def test_bad_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\nname: [unclosed\n", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_scenario(path)


def test_unknown_schema_version_raises(tmp_path):
    data = copy.deepcopy(MINIMAL_VALID)
    data["schema_version"] = 99
    path = _write_yaml(tmp_path, data)

    with pytest.raises(ValidationError):
        load_scenario(path)


def test_defaults_merged_into_tasks(tmp_path):
    data = copy.deepcopy(MINIMAL_VALID)
    data["defaults"] = {"seed": 99}
    # Task has no seed; default should propagate.
    path = _write_yaml(tmp_path, data)

    scenario = load_scenario(path)
    assert scenario.tasks[0].seed == 99


def test_task_seed_overrides_default(tmp_path):
    data = copy.deepcopy(MINIMAL_VALID)
    data["defaults"] = {"seed": 99}
    data["tasks"][0]["seed"] = 77
    path = _write_yaml(tmp_path, data)

    scenario = load_scenario(path)
    assert scenario.tasks[0].seed == 77


def test_defaults_num_samples_merged(tmp_path):
    data = copy.deepcopy(MINIMAL_VALID)
    data["defaults"] = {"num_samples": 16}
    path = _write_yaml(tmp_path, data)

    scenario = load_scenario(path)
    assert scenario.tasks[0].num_samples == 16


def test_defaults_data_encoder_merged(tmp_path):
    data = copy.deepcopy(MINIMAL_VALID)
    data["defaults"] = {"data_encoder": "custom/encoder-v1"}
    path = _write_yaml(tmp_path, data)

    scenario = load_scenario(path)
    assert scenario.tasks[0].data_encoder == "custom/encoder-v1"


def test_path_as_string_or_path_object(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_VALID)

    from_str = load_scenario(str(path))
    from_path = load_scenario(path)

    assert from_str.name == "my-test"
    assert from_path.name == "my-test"
