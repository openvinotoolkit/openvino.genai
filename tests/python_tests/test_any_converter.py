# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pathlib

import pytest

_EXT_DIR = pathlib.Path(__file__).parent / "converter_ext"


def _find_converter_ext():
    for directory in (_EXT_DIR / "build", _EXT_DIR):
        if directory.is_dir():
            matches = sorted(directory.glob("converter_test_ext*.so"))
            if matches:
                return matches[0]
    return None


_ext_path = _find_converter_ext()
if _ext_path is None:
    pytest.skip("converter_test_ext native extension is not built", allow_module_level=True)

# A load or link error here is a real failure, not a skip.
_spec = importlib.util.spec_from_file_location("converter_test_ext", _ext_path)
ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ext)


def test_string():
    result = ext.conv_string()
    assert type(result) is str
    assert result == "HAPPY"


def test_cstr_stored_as_string_and_converts():
    # ov::Any(const char*) normalizes to std::string, so the conversion yields a str.
    assert ext.cstr_stored_type_is_string() is True
    result = ext.conv_cstr()
    assert type(result) is str
    assert result == "Speech"


def test_bool_is_bool_not_int():
    result = ext.conv_bool()
    assert type(result) is bool
    assert result is True


def test_int():
    result = ext.conv_int()
    assert type(result) is int
    assert result == -42


def test_int64():
    result = ext.conv_int64()
    assert type(result) is int
    assert result == 9223372036854775807


def test_size_t():
    result = ext.conv_size_t()
    assert type(result) is int
    assert result == 7


def test_float():
    result = ext.conv_float()
    assert type(result) is float
    assert result == pytest.approx(1.5)


def test_double():
    result = ext.conv_double()
    assert type(result) is float
    assert result == pytest.approx(2.5)


def test_nested_map():
    result = ext.conv_nested_map()
    assert result == {"nested": {"emotion": "HAPPY"}, "flag": True}
    assert type(result["nested"]) is dict
    assert type(result["flag"]) is bool


def test_vec_string():
    result = ext.conv_vec_string()
    assert type(result) is list
    assert result == ["a", "b"]
    assert all(type(item) is str for item in result)


def test_vec_int64():
    result = ext.conv_vec_int64()
    assert result == [1, 2, 3]
    assert all(type(item) is int for item in result)


def test_vec_double():
    result = ext.conv_vec_double()
    assert result == pytest.approx([1.5, 2.5])
    assert all(type(item) is float for item in result)


def test_vec_float():
    result = ext.conv_vec_float()
    assert result == pytest.approx([1.5, 2.5])
    assert all(type(item) is float for item in result)


def test_vec_any_recurses_per_element():
    result = ext.conv_vec_any()
    assert result == ["x", 5, True]
    assert type(result[0]) is str
    assert type(result[1]) is int
    assert type(result[2]) is bool


def test_empty_any_is_none():
    assert ext.conv_empty_any() is None


def test_empty_map_is_empty_dict():
    result = ext.conv_empty_map()
    assert type(result) is dict
    assert result == {}


def test_features_example_shape():
    result = ext.conv_features_example()
    assert result == [{"emotion": "HAPPY", "event": "Speech"}]


def test_unsupported_type_raises():
    with pytest.raises(Exception):
        ext.conv_unsupported()
