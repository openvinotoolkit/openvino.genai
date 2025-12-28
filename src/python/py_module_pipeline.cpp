// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "openvino/genai/module_genai/pipeline.hpp"
#include "py_utils.hpp"
#include "bindings_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;


auto module_generate_docstring = R"(
    Generates sequences or image, video for Modular LLM.

    :param prompt: input prompt
    :type prompt: str
    The prompt can contain <ov_genai_image_i> with i replaced with
    

    :param inputs: Any inputs.
    :type inputs: dict

    :return: return results
    :rtype: dict
)";

ov::AnyMap kwargs_to_intputs(const py::kwargs& kwargs) {
    ov::AnyMap params = {};

    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);

        if (pyutils::py_object_is_any_map(value)) {
            auto map = pyutils::py_object_to_any_map(value);
            params.insert(map.begin(), map.end());
        } else if (py::isinstance<ov::Tensor>(value)) {
            params[key] = value.cast<ov::Tensor>();
        } else if (py::isinstance<py::list>(value)) {
            auto list = value.cast<py::list>();
            if (list.size() > 0 && py::isinstance<ov::Tensor>(list[0])) {
                params[key] = value.cast<std::vector<ov::Tensor>>();
            } else if (list.size() > 0 && py::isinstance<py::str>(list[0])) {
                params[key] = value.cast<std::vector<std::string>>();
            } else {
                std::cout << "Error: Input unsupported data type in list with key: " << key << std::endl;
            }
        } else if (py::isinstance<py::str>(value)) {
            params[key] = value.cast<std::string>();
        } else {
            std::cout << "Error: Input unsupported data type with key: " << key << std::endl;
        }
    }
    return params;
}

py::object output_to_pyobject(const ov::Any& value) {
    if (value.is<int>()) {
        return py::cast(value.as<int>());
    }
    else if (value.is<float>()) {
        return py::cast(value.as<float>());
    }
    else if (value.is<double>()) {
        return py::cast(value.as<double>());
    }
    else if (value.is<bool>()) {
        return py::cast(value.as<bool>());
    }
    else if (value.is<std::vector<float>>()) {
        return py::cast(value.as<std::vector<float>>());
    }
    else if (value.is<std::vector<int>>()) {
        return py::cast(value.as<std::vector<int>>());
    }
    else if (value.is<std::string>()) {
        return py::cast(value.as<std::string>());
    }
    else if (value.is<ov::Tensor>()) {
        return py::cast(value.as<ov::Tensor>());
    }
    else if (value.is<std::vector<ov::Tensor>>()) {
        return py::cast(value.as<std::vector<ov::Tensor>>());
    }
    else if (value.is<ov::AnyMap>()) {
        ov::AnyMap any_map = value.as<ov::AnyMap>();
        py::dict py_dict;
        for (const auto& [k, v] : any_map) {
            py_dict[py::cast(k)] = output_to_pyobject(v);
        }
        return py_dict;
    }
    
    throw std::runtime_error("Unsupported type in ov::Any conversion");
}

void call_module_generate(
    ov::genai::module::ModulePipeline& pipe,
    const py::kwargs& kwargs
) {
    auto inputs = kwargs_to_intputs(kwargs);
    {
        py::gil_scoped_release rel;
        pipe.generate(inputs);
    }
}

void init_module_pipeline(py::module_& m) {
    py::class_<ov::genai::module::ModulePipeline>(m,
                                                  "ModulePipeline",
                                                  "This class is used for generation with ModulePipeline")
        .def(py::init([](const std::filesystem::path& config_yaml_path, const py::kwargs& kwargs) {
                 return std::make_unique<ov::genai::module::ModulePipeline>(config_yaml_path);
             }),
             py::arg("config_yaml_path"),
             "config file path",
             R"(
            Module Pipeline class constructor.
            config_yaml_path (os.PathLike): Path to the configuration YAML file.
        )")
        .def(py::init([](const std::string& config_yaml_content, const py::kwargs& kwargs) {
                 return std::make_unique<ov::genai::module::ModulePipeline>(config_yaml_content);
             }),
             py::arg("config_yaml_content"),
             "config content string",
             R"(
            Module Pipeline class constructor.
            config_yaml_content (str): YAML content as a string.
        )")

        .def("start_chat", &ov::genai::module::ModulePipeline::start_chat, py::arg("system_message") = "")
        .def("finish_chat", &ov::genai::module::ModulePipeline::finish_chat)
        .def(
            "generate",
            [](ov::genai::module::ModulePipeline& pipe, const py::kwargs& inputs) -> void {
                return call_module_generate(pipe, inputs);
            },
            (std::string("generate(**kwargs) -> None\n\n") + "Accepts arbitrary keyword arguments as AnyMap.\n\n" +
             module_generate_docstring)
                .c_str())
        .def(
            "get_output",
            [](ov::genai::module::ModulePipeline& pipe, const std::string& output_name) -> py::object {
                ov::Any result = pipe.get_output(output_name);
                return output_to_pyobject(result);
            },
            py::arg("output_name"),
            "Get output by name.\n\n:param output_name: Name of the output.\n:return: Output value.");
}