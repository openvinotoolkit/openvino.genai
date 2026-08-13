# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from whowhatbench.utils import get_json_config


def test_flattens_nested_config_key():
    """A GenAI-style {"config": {...}} wrapper must be merged into the top level, so
    Optimum's plain ov_config consumers (which don't unwrap "config" themselves) see a flat map."""
    raw = json.dumps({"config": {"DEVICE_PROPERTIES": {"CPU": {"KV_CACHE_PRECISION": "u8"}}}})
    result = get_json_config(raw)
    assert result == {"DEVICE_PROPERTIES": {"CPU": {"KV_CACHE_PRECISION": "u8"}}}
    assert "config" not in result


def test_sibling_keys_win_over_nested_config_on_conflict():
    """Matches GenAI's insert-only-if-absent kwargs_to_any_map semantics: top-level keys
    override same-named keys nested under "config"."""
    raw = json.dumps({"MAX_PROMPT_LEN": 1, "config": {"MAX_PROMPT_LEN": 2, "OTHER": "x"}})
    result = get_json_config(raw)
    assert result == {"MAX_PROMPT_LEN": 1, "OTHER": "x"}


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
