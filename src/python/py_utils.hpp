// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/json_container.hpp"

namespace py = pybind11;
using ov::genai::StreamerBase;

namespace ov::genai::pybind::utils {

// When StreamerVariant is used utf-8 decoding is done by pybind and can lead to exception on incomplete texts.
// Therefore strings decoding should be handled with PyUnicode_DecodeUTF8(..., "replace") to not throw errors.
using PyBindStreamerVariant = std::variant<
    std::function<std::optional<uint16_t>(std::string)>,
    std::shared_ptr<StreamerBase>,
    std::monostate>;

ov::genai::StructuredOutputConfig::CompoundGrammar py_obj_to_structural_tag(const py::object& py_obj);

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

py::list handle_utf8(const std::vector<std::string>& decoded_res);

py::str handle_utf8(const std::string& text);

ov::AnyMap properties_to_any_map(const std::map<std::string, py::object>& properties);

ov::AnyMap kwargs_to_any_map(const py::kwargs& kwargs);

std::filesystem::path ov_tokenizers_module_path();

ov::genai::GenerationConfig update_config_from_kwargs(ov::genai::GenerationConfig config, const py::kwargs& kwargs);

ov::genai::StreamerVariant pystreamer_to_streamer(const PyBindStreamerVariant& py_streamer);

ov::AnyMap py_object_to_any_map(const py::object& py_obj);

ov::genai::JsonContainer py_object_to_json_container(const py::object& obj);

py::object json_container_to_py_object(const ov::genai::JsonContainer& container);

}  // namespace ov::genai::pybind::utils
