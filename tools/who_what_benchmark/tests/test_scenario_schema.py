# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from pydantic import ValidationError
from whowhatbench.scenario.schema import (
    DatasetConfig,
    DefaultsConfig,
    ModelConfig,
    Scenario,
)

# Minimal valid scenario — used as a baseline that individual tests mutate.
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


def _scenario_dict(**overrides):
    """Return a deep copy of MINIMAL_VALID with top-level keys overridden."""
    data = copy.deepcopy(MINIMAL_VALID)
    data.update(overrides)
    return data


def test_valid_scenario_parses():
    scenario = Scenario.model_validate(MINIMAL_VALID)
    assert scenario.schema_version == 1
    assert scenario.name == "my-test"
    assert "base" in scenario.models
    assert "ds1" in scenario.datasets
    assert len(scenario.tasks) == 1


def test_schema_version_must_be_1():
    data = _scenario_dict(schema_version=2)
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_name_must_be_slugified():
    data = _scenario_dict(name="My Test!")
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_name_allows_hyphens_and_underscores():
    data = _scenario_dict(name="my-test_123")
    scenario = Scenario.model_validate(data)
    assert scenario.name == "my-test_123"


def test_unknown_field_raises():
    data = _scenario_dict()
    data["totally_unknown_field"] = 42
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_unknown_field_in_model_config():
    data = copy.deepcopy(MINIMAL_VALID)
    data["models"]["base"]["mystery_option"] = "x"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_missing_models_raises():
    data = copy.deepcopy(MINIMAL_VALID)
    del data["models"]
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_missing_datasets_raises():
    data = copy.deepcopy(MINIMAL_VALID)
    del data["datasets"]
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_missing_tasks_raises():
    data = copy.deepcopy(MINIMAL_VALID)
    del data["tasks"]
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_bad_ref_base_raises():
    data = copy.deepcopy(MINIMAL_VALID)
    data["tasks"][0]["base"] = "does_not_exist"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_bad_ref_target_raises():
    data = copy.deepcopy(MINIMAL_VALID)
    data["tasks"][0]["targets"] = ["does_not_exist"]
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_bad_ref_dataset_raises():
    data = copy.deepcopy(MINIMAL_VALID)
    data["tasks"][0]["dataset"] = "no_such_dataset"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_backend_enum_valid():
    data = copy.deepcopy(MINIMAL_VALID)
    data["models"]["base"]["backend"] = "genai"
    Scenario.model_validate(data)

    data["models"]["base"]["backend"] = "unknown"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_dataset_type_enum_valid():
    data = copy.deepcopy(MINIMAL_VALID)
    data["datasets"]["ds1"]["type"] = "builtin"
    Scenario.model_validate(data)

    data["datasets"]["ds1"]["type"] = "ftp"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


@pytest.mark.parametrize("ds_type", ["builtin", "huggingface", "csv", "inline"])
def test_all_dataset_types_valid(ds_type: str) -> None:
    from whowhatbench.scenario.schema import DatasetConfig

    DatasetConfig.model_validate({"type": ds_type})  # must not raise


def test_task_type_valid():
    data = copy.deepcopy(MINIMAL_VALID)
    data["tasks"][0]["type"] = "text-chat"
    Scenario.model_validate(data)

    data["tasks"][0]["type"] = "not-a-type"
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


@pytest.mark.parametrize(
    "task_type",
    [
        "text",
        "text-chat",
        "text-to-image",
        "text-to-video",
        "speech-generation",
        "visual-text",
        "visual-text-chat",
        "visual-video-text",
        "image-to-image",
        "image-inpainting",
        "text-embedding",
        "text-reranking",
    ],
)
def test_all_task_types_valid(task_type: str) -> None:
    d = copy.deepcopy(MINIMAL_VALID)
    d["tasks"][0]["type"] = task_type
    Scenario.model_validate(d)  # must not raise


def test_inline_dataset_with_prompts():
    cfg = DatasetConfig.model_validate({"type": "inline", "prompts": ["a", "b"]})
    assert cfg.prompts == ["a", "b"]


def test_empty_targets_raises():
    data = copy.deepcopy(MINIMAL_VALID)
    data["tasks"][0]["targets"] = []
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_defaults_config_defaults():
    defaults = DefaultsConfig()
    assert defaults.seed == 42
    assert defaults.device == "CPU"


def test_model_config_requires_path():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"backend": "hf"})


def test_model_config_requires_backend():
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"path": "org/model"})


def test_report_config_group_by_task() -> None:
    from whowhatbench.scenario.schema import ReportConfig

    r = ReportConfig.model_validate({"group_by": "task"})
    assert r.group_by == "task"


def test_report_config_group_by_target() -> None:
    from whowhatbench.scenario.schema import ReportConfig

    r = ReportConfig.model_validate({"group_by": "target"})
    assert r.group_by == "target"


def test_report_config_group_by_invalid() -> None:
    from whowhatbench.scenario.schema import ReportConfig

    with pytest.raises(ValidationError):
        ReportConfig.model_validate({"group_by": "invalid_value"})
