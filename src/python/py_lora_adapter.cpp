// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "./utils.hpp"

namespace py = pybind11;
namespace utils = ov::genai::pybind::utils;


void init_lora_adapter(py::module_& m) {
    py::class_<ov::genai::Adapter>(m, "Adapter", "Inmutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.")
        .def(py::init([](
            const std::string& path
        ) {
            return ov::genai::Adapter(path);
        }),
        py::arg("path"), "path", 
        R"(
            Inmutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
            path (str): Path.
        )"); 
        // TODO: operator bool() const 

    auto adapter_config = py::class_<ov::genai::AdapterConfig>(m, "AdapterConfig", "Adapter config.");
    py::enum_<ov::genai::AdapterConfig::Mode>(adapter_config, "Mode")
        .value("MODE_AUTO", ov::genai::AdapterConfig::Mode::MODE_AUTO)
        .value("MODE_DYNAMIC", ov::genai::AdapterConfig::Mode::MODE_DYNAMIC)
        .value("MODE_STATIC_RANK", ov::genai::AdapterConfig::Mode::MODE_STATIC_RANK)
        .value("MODE_STATIC", ov::genai::AdapterConfig::Mode::MODE_STATIC)
        .value("MODE_FUSE", ov::genai::AdapterConfig::Mode::MODE_FUSE);

    adapter_config.def(py::init([](
         ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(mode);
        }),
        py::arg("mode") = ov::genai::AdapterConfig::Mode::MODE_AUTO);

    adapter_config.def(py::init([](
        const ov::genai::Adapter& adapter, 
        float alpha,
         ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(adapter, alpha, mode);
        }),
        py::arg("adapter"),
        py::arg("alpha"),
        py::arg("mode") = ov::genai::AdapterConfig::Mode::MODE_AUTO);
    
    adapter_config.def(py::init([](
        const ov::genai::Adapter& adapter, 
        ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(adapter, mode);
        }),
        py::arg("adapter"),
        py::arg("mode") = ov::genai::AdapterConfig::Mode::MODE_AUTO);
    
    adapter_config.def(py::init([](
        const std::vector<ov::genai::Adapter>& adapters, 
        ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(adapters, mode);
        }),
        py::arg("adapters"),
        py::arg("mode") = ov::genai::AdapterConfig::Mode::MODE_AUTO);

    adapter_config.def(py::init([](
        const std::vector<std::pair<ov::genai::Adapter, float>>& adapters, 
        ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(adapters, mode);
        }),
        py::arg("adapters"),
        py::arg("mode") = ov::genai::AdapterConfig::Mode::MODE_AUTO);

    //  TODO: Need bindings for following methods

    // template <typename AT, typename std::enable_if<std::is_constructible<Adapter, AT>::value, bool>::type = true>
    // AdapterConfig (const std::initializer_list<AT>& adapters, Mode mode = MODE_AUTO) : AdapterConfig(std::vector<Adapter>(adapters), mode) {}
    // AdapterConfig (const std::initializer_list<std::pair<Adapter, float>>& adapters, Mode mode = MODE_AUTO) : AdapterConfig(std::vector<std::pair<Adapter, float>>(adapters), mode) {}
    // operator bool() const 

    adapter_config.def("set_alpha", &ov::genai::AdapterConfig::set_alpha);
    adapter_config.def("get_alpha", &ov::genai::AdapterConfig::get_alpha);
    adapter_config.def("remove", &ov::genai::AdapterConfig::remove);
    adapter_config.def("get_adapters", &ov::genai::AdapterConfig::get_adapters);
    adapter_config.def("add", static_cast<ov::genai::AdapterConfig& (ov::genai::AdapterConfig::*)(const ov::genai::Adapter&, float)>(&ov::genai::AdapterConfig::add));
    adapter_config.def("add", static_cast<ov::genai::AdapterConfig& (ov::genai::AdapterConfig::*)(const ov::genai::Adapter&)>(&ov::genai::AdapterConfig::add));
}