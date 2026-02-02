// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <filesystem>
#include <cstdint>
#include <climits>

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

ov::AnyMap kwargs_to_inputs(const py::kwargs& kwargs) {
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
        } else if (py::isinstance<py::int_>(value)) {
            // Check if the value fits in int32 or needs int64
            int64_t int64_val = value.cast<int64_t>();
            if (int64_val >= INT32_MIN && int64_val <= INT32_MAX) {
                params[key] = static_cast<int>(int64_val);
            } else {
                params[key] = int64_val;
            }
        } else if (py::isinstance<py::float_>(value)) {
            params[key] = value.cast<float>();
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
    else if (value.is<int64_t>()) {
        return py::cast(value.as<int64_t>());
    }
    else if (value.is<uint32_t>()) {
        return py::cast(value.as<uint32_t>());
    }
    else if (value.is<uint64_t>()) {
        return py::cast(value.as<uint64_t>());
    }
    else if (value.is<size_t>() && !std::is_same_v<size_t, uint64_t>) {
        return py::cast(value.as<size_t>());
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
    else if (value.is<std::vector<int64_t>>()) {
        return py::cast(value.as<std::vector<int64_t>>());
    }
    else if (value.is<std::vector<std::string>>()) {
        return py::cast(value.as<std::vector<std::string>>());
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
    auto inputs = kwargs_to_inputs(kwargs);
    {
        py::gil_scoped_release rel;
        pipe.generate(inputs);
    }
}

// Helper function to convert ov::AnyMap to Python dict
py::dict anymap_to_pydict(const ov::AnyMap& params) {
    py::dict result;
    for (const auto& [key, value] : params) {
        result[py::cast(key)] = output_to_pyobject(value);
    }
    return result;
}

void init_module_pipeline(py::module_& m) {
    // Bind ValidationResult structure
    py::class_<ov::genai::module::ModulePipeline::ValidationResult>(m, "ValidationResult",
        "Result of YAML configuration validation")
        .def(py::init<>())
        .def_readwrite("valid", &ov::genai::module::ModulePipeline::ValidationResult::valid,
            "Whether the configuration is valid")
        .def_readwrite("errors", &ov::genai::module::ModulePipeline::ValidationResult::errors,
            "List of error messages")
        .def_readwrite("warnings", &ov::genai::module::ModulePipeline::ValidationResult::warnings,
            "List of warning messages")
        .def("__repr__", [](const ov::genai::module::ModulePipeline::ValidationResult& self) {
            std::stringstream ss;
            ss << "ValidationResult(valid=" << (self.valid ? "True" : "False");
            ss << ", errors=[";
            for (size_t i = 0; i < self.errors.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << "'" << self.errors[i] << "'";
            }
            ss << "], warnings=[";
            for (size_t i = 0; i < self.warnings.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << "'" << self.warnings[i] << "'";
            }
            ss << "])";
            return ss.str();
        });

    // Helper function to convert Python dict to ConfigModelsMap
    // Python dict format: {"module_name": {"model_name": ov.Model, ...}, ...}
    auto py_dict_to_config_models_map = [](const py::object& obj) -> ov::genai::module::ConfigModelsMap {
        ov::genai::module::ConfigModelsMap result;
        if (obj.is_none()) {
            return result;
        }

        py::dict models_dict = obj.cast<py::dict>();
        for (const auto& module_item : models_dict) {
            std::string module_name = py::cast<std::string>(module_item.first);
            py::dict inner_dict = module_item.second.cast<py::dict>();

            std::map<std::string, std::shared_ptr<ov::Model>> inner_map;
            for (const auto& model_item : inner_dict) {
                std::string model_name = py::cast<std::string>(model_item.first);
                auto model = model_item.second.cast<std::shared_ptr<ov::Model>>();
                inner_map[model_name] = model;
            }
            result[module_name] = inner_map;
        }
        return result;
    };

    py::class_<ov::genai::module::ModulePipeline>(m,
                                                  "ModulePipeline",
                                                  "This class is used for generation with ModulePipeline")
        .def(py::init([py_dict_to_config_models_map](const std::filesystem::path& config_yaml_path,
                                                      const py::object& models_map,
                                                      const py::kwargs& kwargs) {
                 auto cpp_models_map = py_dict_to_config_models_map(models_map);
                 return std::make_unique<ov::genai::module::ModulePipeline>(config_yaml_path, cpp_models_map);
             }),
             py::arg("config_yaml_path"),
             py::arg("models_map") = py::none(),
             R"(
            Module Pipeline class constructor.

            :param config_yaml_path: Path to the configuration YAML file.
            :type config_yaml_path: os.PathLike
            :param models_map: Optional pre-loaded models map. Format: {"module_name": {"model_name": ov.Model, ...}, ...}
            :type models_map: dict[str, dict[str, openvino.Model]], optional
        )")
        .def(py::init([py_dict_to_config_models_map](const std::string& config_yaml_content,
                                                      const py::object& models_map,
                                                      const py::kwargs& kwargs) {
                 auto cpp_models_map = py_dict_to_config_models_map(models_map);
                 return std::make_unique<ov::genai::module::ModulePipeline>(config_yaml_content, cpp_models_map);
             }),
             py::arg("config_yaml_content"),
             py::arg("models_map") = py::none(),
             R"(
            Module Pipeline class constructor.

            :param config_yaml_content: YAML content as a string.
            :type config_yaml_content: str
            :param models_map: Optional pre-loaded models map. Format: {"module_name": {"model_name": ov.Model, ...}, ...}
            :type models_map: dict[str, dict[str, openvino.Model]], optional
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
            "generate_async",
            [](ov::genai::module::ModulePipeline& pipe, const py::kwargs& inputs) -> void {
                auto ov_inputs = kwargs_to_inputs(inputs);
                {
                    py::gil_scoped_release rel;
                    pipe.generate_async(ov_inputs);
                }
            },
            (std::string("generate_async(**kwargs) -> None\n\n") + "Accepts arbitrary keyword arguments as AnyMap.\n\n" +
             module_generate_docstring)
                .c_str())
        .def(
            "get_output",
            [](ov::genai::module::ModulePipeline& pipe, const std::string& output_name) -> py::object {
                ov::Any result = pipe.get_output(output_name);
                return output_to_pyobject(result);
            },
            py::arg("output_name"),
            "Get output by name.\n\n:param output_name: Name of the output.\n:return: Output value.")
        .def(
            "get_output",
            [](ov::genai::module::ModulePipeline& pipe) -> py::object {
                ov::Any result = pipe.get_output();
                return output_to_pyobject(result);
            },
            "Get the only output when there is a single output.\n\n:return: Output value.")
        // Static method: validate_config (file path)
        .def_static(
            "validate_config",
            [](const std::filesystem::path& config_yaml_path) {
                return ov::genai::module::ModulePipeline::validate_config(config_yaml_path);
            },
            py::arg("config_yaml_path"),
            R"(
            Validate YAML configuration file.

            :param config_yaml_path: Path to the YAML configuration file.
            :type config_yaml_path: os.PathLike
            :return: ValidationResult containing valid flag, errors and warnings.
            :rtype: ValidationResult
            )")
        // Static method: validate_config_string (YAML content string)
        .def_static(
            "validate_config_string",
            [](const std::string& config_yaml_content) {
                return ov::genai::module::ModulePipeline::validate_config_string(config_yaml_content);
            },
            py::arg("config_yaml_content"),
            R"(
            Validate YAML configuration content string.

            :param config_yaml_content: YAML configuration content as string.
            :type config_yaml_content: str
            :return: ValidationResult containing valid flag, errors and warnings.
            :rtype: ValidationResult
            )")
        // Static method: comfyui_json_to_yaml (file path)
        .def_static(
            "comfyui_json_to_yaml",
            [](const std::filesystem::path& comfyui_json_path, const py::kwargs& kwargs) -> py::tuple {
                ov::AnyMap pipeline_inputs = kwargs_to_inputs(kwargs);
                std::string yaml_content = ov::genai::module::ModulePipeline::comfyui_json_to_yaml(
                    comfyui_json_path, pipeline_inputs);
                return py::make_tuple(yaml_content, anymap_to_pydict(pipeline_inputs));
            },
            py::arg("comfyui_json_path"),
            R"(
            Convert ComfyUI JSON file to YAML configuration string.

            :param comfyui_json_path: Path to ComfyUI JSON file (workflow or API format).
            :type comfyui_json_path: os.PathLike
            :param kwargs: Optional pipeline parameters like model_path_base, default_device.
            :return: Tuple of (yaml_config_string, extracted_parameters_dict).
            :rtype: tuple[str, dict]
            )")
        // Static method: comfyui_json_string_to_yaml (JSON content string)
        .def_static(
            "comfyui_json_string_to_yaml",
            [](const std::string& comfyui_json_content, const py::kwargs& kwargs) -> py::tuple {
                ov::AnyMap pipeline_inputs = kwargs_to_inputs(kwargs);
                std::string yaml_content = ov::genai::module::ModulePipeline::comfyui_json_string_to_yaml(
                    comfyui_json_content, pipeline_inputs);
                return py::make_tuple(yaml_content, anymap_to_pydict(pipeline_inputs));
            },
            py::arg("comfyui_json_content"),
            R"(
            Convert ComfyUI JSON string to YAML configuration string.

            :param comfyui_json_content: ComfyUI JSON content string (workflow or API format).
            :type comfyui_json_content: str
            :param kwargs: Optional pipeline parameters like model_path_base, default_device.
            :return: Tuple of (yaml_config_string, extracted_parameters_dict).
            :rtype: tuple[str, dict]
            )");
}
