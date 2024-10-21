// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "py_utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <openvino/runtime/auto/properties.hpp>

#include "tokenizers_path.hpp"
#include "openvino/genai/llm_pipeline.hpp"

namespace py = pybind11;
namespace ov::genai::pybind::utils {

py::str handle_utf8(const std::string& text) {
    // pybind11 decodes strings similar to Pythons's
    // bytes.decode('utf-8'). It raises if the decoding fails.
    // generate() may return incomplete Unicode points if max_new_tokens
    // was reached. Replace such points with � instead of raising an exception
    PyObject* py_s = PyUnicode_DecodeUTF8(text.data(), text.length(), "replace");
    return py::reinterpret_steal<py::object>(py_s);
}

py::list handle_utf8(const std::vector<std::string>& decoded_res) {
    // pybind11 decodes strings similar to Pythons's
    // bytes.decode('utf-8'). It raises if the decoding fails.
    // generate() may return incomplete Unicode points if max_new_tokens
    // was reached. Replace such points with � instead of raising an exception
    py::list res;
    for (const auto s: decoded_res) {
        PyObject* py_s = PyUnicode_DecodeUTF8(s.data(), s.length(), "replace");
        res.append(py::reinterpret_steal<py::object>(py_s));
    }
    return res;
}

bool py_object_is_any_map(const py::object& py_obj) {
    if (!py::isinstance<py::dict>(py_obj)) {
        return false;
    }
    auto dict = py::cast<py::dict>(py_obj);
    return std::all_of(dict.begin(), dict.end(), [&](const std::pair<py::object::handle, py::object::handle>& elem) {
        return py::isinstance<py::str>(elem.first);
    });
}

ov::Any py_object_to_any(const py::object& py_obj);

ov::AnyMap py_object_to_any_map(const py::object& py_obj) {
    OPENVINO_ASSERT(py_object_is_any_map(py_obj), "Unsupported attribute type.");
    ov::AnyMap return_value = {};
    for (auto& item : py::cast<py::dict>(py_obj)) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);
        if (py_object_is_any_map(value)) {
            return_value[key] = py_object_to_any_map(value);
        } else {
            return_value[key] = py_object_to_any(value);
        }
    }
    return return_value;
}

ov::Any py_object_to_any(const py::object& py_obj) {
    // Python types
    py::object float_32_type = py::module_::import("numpy").attr("float32");
    
    if (py::isinstance<py::str>(py_obj)) {
        return py_obj.cast<std::string>();
    } else if (py::isinstance<py::bool_>(py_obj)) {
        return py_obj.cast<bool>();
    } else if (py::isinstance<py::bytes>(py_obj)) {
        return py_obj.cast<std::string>();
    } else if (py::isinstance<py::float_>(py_obj)) {
        return py_obj.cast<double>();
    } else if (py::isinstance(py_obj, float_32_type)) {
        return py_obj.cast<float>();
    } else if (py::isinstance<py::int_>(py_obj)) {
        return py_obj.cast<int64_t>();
    } else if (py::isinstance<py::none>(py_obj)) {
        return {};
    } else if (py::isinstance<py::list>(py_obj)) {
        auto _list = py_obj.cast<py::list>();
        enum class PY_TYPE : int { UNKNOWN = 0, STR, INT, FLOAT, BOOL, PARTIAL_SHAPE };
        PY_TYPE detected_type = PY_TYPE::UNKNOWN;
        for (const auto& it : _list) {
            auto check_type = [&](PY_TYPE type) {
                if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                    detected_type = type;
                    return;
                }
                OPENVINO_THROW("Incorrect attribute. Mixed types in the list are not allowed.");
            };
            if (py::isinstance<py::str>(it)) {
                check_type(PY_TYPE::STR);
            } else if (py::isinstance<py::int_>(it)) {
                check_type(PY_TYPE::INT);
            } else if (py::isinstance<py::float_>(it)) {
                check_type(PY_TYPE::FLOAT);
            } else if (py::isinstance<py::bool_>(it)) {
                check_type(PY_TYPE::BOOL);
            } else if (py::isinstance<ov::PartialShape>(it)) {
                check_type(PY_TYPE::PARTIAL_SHAPE);
            }
        }

        if (_list.empty())
            return ov::Any();

        switch (detected_type) {
        case PY_TYPE::STR:
            return _list.cast<std::vector<std::string>>();
        case PY_TYPE::FLOAT:
            return _list.cast<std::vector<double>>();
        case PY_TYPE::INT:
            return _list.cast<std::vector<int64_t>>();
        case PY_TYPE::BOOL:
            return _list.cast<std::vector<bool>>();
        case PY_TYPE::PARTIAL_SHAPE:
            return _list.cast<std::vector<ov::PartialShape>>();
        default:
            OPENVINO_ASSERT(false, "Unsupported attribute type.");
        }
    
    // OV types
    } else if (py_object_is_any_map(py_obj)) {
        return py_object_to_any_map(py_obj);
    } else if (py::isinstance<ov::Any>(py_obj)) {
        return py::cast<ov::Any>(py_obj);
    } else if (py::isinstance<ov::element::Type>(py_obj)) {
        return py::cast<ov::element::Type>(py_obj);
    } else if (py::isinstance<ov::PartialShape>(py_obj)) {
        return py::cast<ov::PartialShape>(py_obj);
    } else if (py::isinstance<ov::hint::Priority>(py_obj)) {
        return py::cast<ov::hint::Priority>(py_obj);
    } else if (py::isinstance<ov::hint::PerformanceMode>(py_obj)) {
        return py::cast<ov::hint::PerformanceMode>(py_obj);
    } else if (py::isinstance<ov::intel_auto::SchedulePolicy>(py_obj)) {
        return py::cast<ov::intel_auto::SchedulePolicy>(py_obj);
    } else if (py::isinstance<ov::hint::SchedulingCoreType>(py_obj)) {
        return py::cast<ov::hint::SchedulingCoreType>(py_obj);
    } else if (py::isinstance<std::set<ov::hint::ModelDistributionPolicy>>(py_obj)) {
        return py::cast<std::set<ov::hint::ModelDistributionPolicy>>(py_obj);
    } else if (py::isinstance<ov::hint::ExecutionMode>(py_obj)) {
        return py::cast<ov::hint::ExecutionMode>(py_obj);
    } else if (py::isinstance<ov::log::Level>(py_obj)) {
        return py::cast<ov::log::Level>(py_obj);
    } else if (py::isinstance<ov::device::Type>(py_obj)) {
        return py::cast<ov::device::Type>(py_obj);
    } else if (py::isinstance<ov::streams::Num>(py_obj)) {
        return py::cast<ov::streams::Num>(py_obj);
    } else if (py::isinstance<ov::Affinity>(py_obj)) {
        return py::cast<ov::Affinity>(py_obj);
    } else if (py::isinstance<ov::Tensor>(py_obj)) {
        return py::cast<ov::Tensor>(py_obj);
    } else if (py::isinstance<ov::Output<ov::Node>>(py_obj)) {
        return py::cast<ov::Output<ov::Node>>(py_obj);
    } else if (py::isinstance<ov::genai::SchedulerConfig>(py_obj)) {
        return py::cast<ov::genai::SchedulerConfig>(py_obj);
    } else if (py::isinstance<py::object>(py_obj)) {
        return py_obj;
    }
    OPENVINO_ASSERT(false, "Unsupported attribute type.");
}

std::map<std::string, ov::Any> properties_to_any_map(const std::map<std::string, py::object>& properties) {
    std::map<std::string, ov::Any> properties_to_cpp;
    for (const auto& property : properties) {
        properties_to_cpp[property.first] = py_object_to_any(property.second);
    }
    return properties_to_cpp;
}


ov::AnyMap kwargs_to_any_map(const py::kwargs& kwargs) {
    ov::AnyMap params = {};

    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);
        if (utils::py_object_is_any_map(value)) {
            auto map = utils::py_object_to_any_map(value);
            params.insert(map.begin(), map.end());
        } else {
            params[key] = utils::py_object_to_any(value);
        }

    }
    return params;
}

std::string ov_tokenizers_module_path() {
    // Try a path relative to build artifacts folder first.
    std::filesystem::path from_relative = tokenizers_relative_to_genai();
    if (std::filesystem::exists(from_relative)) {
        return from_relative.string();
    }
    return py::str(py::module_::import("openvino_tokenizers").attr("_ext_path"));
}

ov::genai::StreamerVariant pystreamer_to_streamer(const PyBindStreamerVariant& py_streamer) {
    ov::genai::StreamerVariant streamer = std::monostate();

    std::visit(overloaded {
    [&streamer](const std::function<bool(py::str)>& py_callback){
        // Wrap python streamer with manual utf-8 decoding. Do not rely
        // on pybind automatic decoding since it raises exceptions on incomplete strings.
        auto callback_wrapped = [py_callback](std::string subword) -> bool {
            auto py_str = PyUnicode_DecodeUTF8(subword.data(), subword.length(), "replace");
            return py_callback(py::reinterpret_borrow<py::str>(py_str));
        };
        streamer = callback_wrapped;
    },
    [&streamer](std::shared_ptr<StreamerBase> streamer_cls){
        streamer = streamer_cls;
    },
    [](std::monostate none){ /*streamer is already a monostate */ }
    }, py_streamer);
    return streamer;
}

ov::genai::OptionalGenerationConfig update_config_from_kwargs(const ov::genai::OptionalGenerationConfig& config, const py::kwargs& kwargs) {
    if(!config.has_value() && kwargs.empty())
        return std::nullopt;

    ov::genai::GenerationConfig res_config;
    if(config.has_value())
        res_config = *config;
 
    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);

        if (item.second.is_none()) {
            // Even if argument key name does not fit GenerationConfig name 
            // it's not an eror if it's not defined. 
            // Some HF configs can have parameters for methods currenly unsupported in ov_genai
            // but if their values are not set / None, then this should not block 
            // us from reading such configs, e.g. {"typical_p": None, 'top_p': 1.0,...}
            return res_config;
        }
        
        if (key == "max_new_tokens") {
            res_config.max_new_tokens = py::cast<int>(item.second);
        } else if (key == "max_length") {
            res_config.max_length = py::cast<int>(item.second);
        } else if (key == "ignore_eos") {
            res_config.ignore_eos = py::cast<bool>(item.second);
        } else if (key == "num_beam_groups") {
            res_config.num_beam_groups = py::cast<int>(item.second);
        } else if (key == "num_beams") {
            res_config.num_beams = py::cast<int>(item.second);
        } else if (key == "diversity_penalty") {
            res_config.diversity_penalty = py::cast<float>(item.second);
        } else if (key == "length_penalty") {
            res_config.length_penalty = py::cast<float>(item.second);
        } else if (key == "num_return_sequences") {
            res_config.num_return_sequences = py::cast<int>(item.second);
        } else if (key == "no_repeat_ngram_size") {
            res_config.no_repeat_ngram_size = py::cast<int>(item.second);
        } else if (key == "stop_criteria") {
            res_config.stop_criteria = py::cast<StopCriteria>(item.second);
        } else if (key == "temperature") {
            res_config.temperature = py::cast<float>(item.second);
        } else if (key == "top_p") {
            res_config.top_p = py::cast<float>(item.second);
        } else if (key == "top_k") {
            res_config.top_k = py::cast<int>(item.second);
        } else if (key == "do_sample") {
            res_config.do_sample = py::cast<bool>(item.second);
        } else if (key == "repetition_penalty") {
            res_config.repetition_penalty = py::cast<float>(item.second);
        } else if (key == "eos_token_id") {
            res_config.set_eos_token_id(py::cast<int>(item.second));
        } else {
            throw(std::invalid_argument("'" + key + "' is incorrect GenerationConfig parameter name. "
                                        "Use help(openvino_genai.GenerationConfig) to get list of acceptable parameters."));
        }
    }

    return res_config;
}

}  // namespace ov::genai::pybind::utils
