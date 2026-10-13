// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "py_any_converter.hpp"

namespace py = pybind11;
using ov::genai::pybind::utils::any_map_to_py_object;
using ov::genai::pybind::utils::any_to_py_object;

namespace {

struct UnsupportedType {};

// Verify the stored type before exercising the converter.
template <typename T>
py::object convert_checked(const ov::Any& any) {
    OPENVINO_ASSERT(any.is<T>(), "test input does not store the intended C++ type");
    return any_to_py_object(any);
}

}  // namespace

PYBIND11_MODULE(converter_test_ext, m) {
    m.def("conv_string", [] {
        return convert_checked<std::string>(ov::Any(std::string("HAPPY")));
    });
    m.def("conv_cstr", [] {
        return any_to_py_object(ov::Any("Speech"));
    });
    m.def("cstr_stored_type_is_string", [] {
        return ov::Any("Speech").is<std::string>();
    });
    m.def("conv_bool", [] {
        return convert_checked<bool>(ov::Any(true));
    });
    m.def("conv_int", [] {
        return convert_checked<int>(ov::Any(static_cast<int>(-42)));
    });
    m.def("conv_int64", [] {
        return convert_checked<int64_t>(ov::Any(std::numeric_limits<int64_t>::max()));
    });
    m.def("conv_size_t", [] {
        return convert_checked<size_t>(ov::Any(static_cast<size_t>(7)));
    });
    m.def("conv_float", [] {
        return convert_checked<float>(ov::Any(1.5f));
    });
    m.def("conv_double", [] {
        return convert_checked<double>(ov::Any(2.5));
    });
    m.def("conv_nested_map", [] {
        ov::AnyMap inner{{"emotion", std::string("HAPPY")}};
        ov::AnyMap outer{{"nested", inner}, {"flag", true}};
        return any_map_to_py_object(outer);
    });
    m.def("conv_vec_string", [] {
        return convert_checked<std::vector<std::string>>(ov::Any(std::vector<std::string>{"a", "b"}));
    });
    m.def("conv_vec_int64", [] {
        return convert_checked<std::vector<int64_t>>(ov::Any(std::vector<int64_t>{1, 2, 3}));
    });
    m.def("conv_vec_double", [] {
        return convert_checked<std::vector<double>>(ov::Any(std::vector<double>{1.5, 2.5}));
    });
    m.def("conv_vec_float", [] {
        return convert_checked<std::vector<float>>(ov::Any(std::vector<float>{1.5f, 2.5f}));
    });
    m.def("conv_vec_any", [] {
        std::vector<ov::Any> values;
        values.emplace_back(std::string("x"));
        values.emplace_back(static_cast<int64_t>(5));
        values.emplace_back(true);
        return convert_checked<std::vector<ov::Any>>(ov::Any(values));
    });
    m.def("conv_empty_any", [] {
        return any_to_py_object(ov::Any());
    });
    m.def("conv_empty_map", [] {
        return any_map_to_py_object(ov::AnyMap{});
    });
    m.def("conv_features_example", [] {
        std::vector<ov::AnyMap> features{
            ov::AnyMap{{"emotion", std::string("HAPPY")}, {"event", std::string("Speech")}}};
        py::list out;
        for (const auto& entry : features) {
            out.append(any_map_to_py_object(entry));
        }
        return out;
    });
    m.def("conv_unsupported", [] {
        return any_to_py_object(ov::Any(UnsupportedType{}));
    });
}
