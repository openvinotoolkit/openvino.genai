// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/any.hpp"

namespace py = pybind11;

namespace ov::genai::pybind::utils {

/**
 * @brief Convert a single ov::Any to a Python object.
 *
 * Supports strings, booleans, integer and floating-point scalars, nested ov::AnyMap,
 * and selected vector types. An empty ov::Any maps to None. Throws for any
 * other stored type so unmodeled metadata is never silently dropped.
 */
py::object any_to_py_object(const ov::Any& any);

/**
 * @brief Convert an ov::AnyMap to a Python dict, converting each value with any_to_py_object.
 */
py::dict any_map_to_py_object(const ov::AnyMap& any_map);

}  // namespace ov::genai::pybind::utils
