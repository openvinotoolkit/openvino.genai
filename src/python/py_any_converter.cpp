// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "py_any_converter.hpp"

#include "openvino/core/except.hpp"

namespace ov::genai::pybind::utils {

py::object any_to_py_object(const ov::Any& any) {
    if (any.empty()) {
        return py::none();
    }
    if (any.is<std::string>()) {
        return py::cast(any.as<std::string>());
    }
    if (any.is<bool>()) {
        return py::cast(any.as<bool>());
    }
    if (any.is<int>()) {
        return py::cast(any.as<int>());
    }
    if (any.is<int64_t>()) {
        return py::cast(any.as<int64_t>());
    }
    if (any.is<size_t>()) {
        return py::cast(any.as<size_t>());
    }
    if (any.is<float>()) {
        return py::cast(any.as<float>());
    }
    if (any.is<double>()) {
        return py::cast(any.as<double>());
    }
    if (any.is<ov::AnyMap>()) {
        return any_map_to_py_object(any.as<ov::AnyMap>());
    }
    if (any.is<std::vector<std::string>>()) {
        return py::cast(any.as<std::vector<std::string>>());
    }
    if (any.is<std::vector<int64_t>>()) {
        return py::cast(any.as<std::vector<int64_t>>());
    }
    if (any.is<std::vector<double>>()) {
        return py::cast(any.as<std::vector<double>>());
    }
    if (any.is<std::vector<float>>()) {
        return py::cast(any.as<std::vector<float>>());
    }
    if (any.is<std::vector<ov::Any>>()) {
        py::list items;
        for (const ov::Any& item : any.as<std::vector<ov::Any>>()) {
            items.append(any_to_py_object(item));
        }
        return items;
    }
    OPENVINO_THROW("Cannot convert ov::Any to a Python object: unsupported stored type '",
                   any.type_info().name(),
                   "'");
}

py::dict any_map_to_py_object(const ov::AnyMap& any_map) {
    py::dict result;
    for (const auto& [key, value] : any_map) {
        result[py::str(key)] = any_to_py_object(value);
    }
    return result;
}

}  // namespace ov::genai::pybind::utils
