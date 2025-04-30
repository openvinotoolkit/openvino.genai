// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "py_utils.hpp"

namespace py = pybind11;

void init_lora_adapter(py::module_& m) {
    py::class_<ov::genai::Adapter>(m, "Adapter", "Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.")
        .def(py::init<>())
        .def(py::init([](
            const std::filesystem::path& path
        ) {
            return ov::genai::Adapter(path);
        }),
        py::arg("path"), "path",
        R"(
            Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
            path (os.PathLike): Path to adapter file in safetensors format.
        )")
        .def(py::init([](
            const ov::Tensor& safetensor
        ) {
            return ov::genai::Adapter(safetensor);
        }),
        py::arg("safetensor"), "ov::Tensor with pre-read LoRA Adapter safetensor",
        R"(
            Immutable LoRA Adapter that carries the adaptation matrices and serves as unique adapter identifier.
            safetensor (ov.Tensor): Pre-read LoRA Adapter safetensor.
        )")
        .def(
            "__bool__",
            [](ov::genai::Adapter& self
            ) {
                return bool(self);
            });

    auto adapter_config = py::class_<ov::genai::AdapterConfig>(m, "AdapterConfig", "Adapter config that defines a combination of LoRA adapters with blending parameters.");
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
        py::arg_v("mode", ov::genai::AdapterConfig::Mode::MODE_AUTO, "AdapterConfig.Mode.MODE_AUTO"));

    adapter_config.def(py::init([](
        const ov::genai::Adapter& adapter,
        float alpha,
         ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(adapter, alpha, mode);
        }),
        py::arg("adapter"),
        py::arg("alpha"),
        py::arg_v("mode", ov::genai::AdapterConfig::Mode::MODE_AUTO, "AdapterConfig.Mode.MODE_AUTO"));

    adapter_config.def(py::init([](
        const ov::genai::Adapter& adapter,
        ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(adapter, mode);
        }),
        py::arg("adapter"),
        py::arg_v("mode", ov::genai::AdapterConfig::Mode::MODE_AUTO, "AdapterConfig.Mode.MODE_AUTO"));

    adapter_config.def(py::init([](
        const std::vector<ov::genai::Adapter>& adapters,
        ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(adapters, mode);
        }),
        py::arg("adapters"),
        py::arg_v("mode", ov::genai::AdapterConfig::Mode::MODE_AUTO, "AdapterConfig.Mode.MODE_AUTO"));

    adapter_config.def(py::init([](
        const std::vector<std::pair<ov::genai::Adapter, float>>& adapters,
        ov::genai::AdapterConfig::Mode mode) {
            return std::make_unique<ov::genai::AdapterConfig>(adapters, mode);
        }),
        py::arg("adapters"),
        py::arg_v("mode", ov::genai::AdapterConfig::Mode::MODE_AUTO, "AdapterConfig.Mode.MODE_AUTO"));
    adapter_config.def(
        "__bool__",
        [](ov::genai::AdapterConfig& self
        ) {
            return bool(self);
        });

    adapter_config.def("set_alpha", &ov::genai::AdapterConfig::set_alpha, py::arg("adapter"), py::arg("alpha"));
    adapter_config.def("get_alpha", &ov::genai::AdapterConfig::get_alpha, py::arg("adapter"));
    adapter_config.def("remove", &ov::genai::AdapterConfig::remove, py::arg("adapter"));
    adapter_config.def("get_adapters", &ov::genai::AdapterConfig::get_adapters);
    adapter_config.def("add", static_cast<ov::genai::AdapterConfig& (ov::genai::AdapterConfig::*)(const ov::genai::Adapter&, float)>(&ov::genai::AdapterConfig::add), py::arg("adapter"), py::arg("alpha"));
    adapter_config.def("add", static_cast<ov::genai::AdapterConfig& (ov::genai::AdapterConfig::*)(const ov::genai::Adapter&)>(&ov::genai::AdapterConfig::add), py::arg("adapter"));
    adapter_config.def("get_adapters_and_alphas", &ov::genai::AdapterConfig::get_adapters_and_alphas);
    adapter_config.def("set_adapters_and_alphas", &ov::genai::AdapterConfig::set_adapters_and_alphas, py::arg("adapters"));
}
