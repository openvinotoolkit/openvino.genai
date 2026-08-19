# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from whowhatbench.utils import get_json_config


def test_no_nested_config_key_is_returned_unchanged():
    raw = json.dumps({"INFERENCE_PRECISION_HINT": "f32"})
    assert get_json_config(raw) == {"INFERENCE_PRECISION_HINT": "f32"}


def test_config_key_with_non_dict_value_is_left_as_is():
    """A "config" key that isn't itself a dict (e.g. a string) is not a nesting wrapper."""
    raw = json.dumps({"config": "some_string_value"})
    assert get_json_config(raw) == {"config": "some_string_value"}


def test_reads_from_file(tmp_path):
    config_path = tmp_path / "ov_config.json"
    config_path.write_text(json.dumps({"config": {"DEVICE_PROPERTIES": {"CPU": {}}}}))
    result = get_json_config(str(config_path))
    assert result == {"DEVICE_PROPERTIES": {"CPU": {}}}


def test_empty_string_raises_value_error():
    with pytest.raises(ValueError):
        get_json_config("")


def test_none_raises_value_error():
    with pytest.raises(ValueError):
        get_json_config(None)
