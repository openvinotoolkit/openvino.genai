// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "py_utils.hpp"

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include <openvino/runtime/auto/properties.hpp>

#include "tokenizer/tokenizers_path.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"

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
    py::list res;
    for (const auto s: decoded_res) {
        py::str r = handle_utf8(s);
        res.append(r);
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

ov::Any py_object_to_any(const py::object& py_obj, std::string property_name);

ov::AnyMap py_object_to_any_map(const py::object& py_obj) {
    OPENVINO_ASSERT(py_object_is_any_map(py_obj), "Unsupported attribute type.");
    ov::AnyMap return_value = {};
    for (auto& item : py::cast<py::dict>(py_obj)) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);
        if (py_object_is_any_map(value)) {
            return_value[key] = py_object_to_any_map(value);
        } else {
            return_value[key] = py_object_to_any(value, key);
        }
    }
    return return_value;
}

ov::Any py_object_to_any(const py::object& py_obj, std::string property_name) {
    // These properties should be casted to ov::AnyMap, instead of std::map. 
    std::set<std::string> any_map_properties = {
        "GENERATE_CONFIG",
        "PREFILL_CONFIG",
        "SHARED_HEAD_CONFIG",
        "NPUW_LLM_GENERATE_CONFIG",
        "NPUW_LLM_PREFILL_CONFIG",
        "NPUW_LLM_SHARED_HEAD_CONFIG",
        "++GENERATE_CONFIG",
        "++PREFILL_CONFIG",
        "++SHARED_HEAD_CONFIG",
        "++NPUW_LLM_GENERATE_CONFIG",
        "++NPUW_LLM_PREFILL_CONFIG",
        "++NPUW_LLM_SHARED_HEAD_CONFIG"
    };

    py::object float_32_type = py::module_::import("numpy").attr("float32");
    if (py::isinstance<py::str>(py_obj)) {
        if (property_name == "structural_tags_config") {
            std::variant<ov::genai::StructuralTagsConfig, ov::genai::StructuredOutputConfig::StructuralTag> variant_value =
                py_obj.cast<std::string>();
            return variant_value;
        }
        return py_obj.cast<std::string>();
    } else if (py::isinstance<py::bool_>(py_obj)) {
        return py_obj.cast<bool>();
    } else if (py::isinstance<py::bytes>(py_obj)) {
        return py_obj.cast<std::string>();
    } else if (py::isinstance<py::float_>(py_obj)) {
        return py_obj.cast<float>();
    } else if (py::isinstance(py_obj, float_32_type)) {
        return py_obj.cast<float>();
    } else if (py::isinstance<py::int_>(py_obj)) {
        return py_obj.cast<int64_t>();
    } else if (py::isinstance<py::none>(py_obj)) {
        return {};
    } else if (py::isinstance<py::list>(py_obj)) {
        if (property_name == ov::cache_encryption_callbacks.name()) {
            // this impl is based on OpenVINO python bindings impl.
            auto property_list = py_obj.cast<py::list>();
            // Wrapped to sp due-to we need to hold GIL upon destruction of python function
            auto py_encrypt = std::shared_ptr<py::function>(new py::function(std::move(property_list[0])),
                                                            [](py::function* py_encrypt) {
                                                                py::gil_scoped_acquire acquire;
                                                                delete py_encrypt;
                                                            });
            auto py_decrypt = std::shared_ptr<py::function>(new py::function(std::move(property_list[1])),
                                                            [](py::function* py_decrypt) {
                                                                py::gil_scoped_acquire acquire;
                                                                delete py_decrypt;
                                                            });

            std::function<std::string(const std::string&)> encrypt_func =
                [py_encrypt](const std::string& in_str) -> std::string {
                // Acquire GIL, execute Python function
                py::gil_scoped_acquire acquire;
                return (*py_encrypt)(py::bytes(in_str)).cast<std::string>();
            };

            std::function<std::string(const std::string&)> decrypt_func =
                [py_decrypt](const std::string& in_str) -> std::string {
                // Acquire GIL, execute Python function
                py::gil_scoped_acquire acquire;
                return (*py_decrypt)(py::bytes(in_str)).cast<std::string>();
            };
            ov::EncryptionCallbacks encryption_callbacks{encrypt_func, decrypt_func};
            return encryption_callbacks;
        } else if (property_name == "structural_tags") {
            // this impl is based on OpenVINO python bindings impl.
            auto property_list = py_obj.cast<py::list>();
            std::vector<ov::genai::StructuralTagItem> structural_tags;
            for (const auto& item : property_list) {
                if (py::isinstance<ov::genai::StructuralTagItem>(item)) {
                    structural_tags.push_back(item.cast<ov::genai::StructuralTagItem>());
                } else {
                    OPENVINO_THROW("Incorrect value in \"", property_name, "\". Expected StructuralTagItem.");
                }
            }
            return structural_tags;
        } else {
            auto _list = py_obj.cast<py::list>();
            enum class PY_TYPE : int { UNKNOWN = 0, STR, INT, FLOAT, BOOL, PARTIAL_SHAPE, TENSOR, DICT};
            PY_TYPE detected_type = PY_TYPE::UNKNOWN;
            for (const auto& it : _list) {
                auto check_type = [&](PY_TYPE type) {
                    if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                        detected_type = type;
                        return;
                    }
                    OPENVINO_THROW("Incorrect value in \"" + property_name + "\". Mixed types in the list are not allowed.");
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
                } else if (py::isinstance<ov::Tensor>(it)) {
                    check_type(PY_TYPE::TENSOR);
                } else if (py::isinstance<py::dict>(it)) {
                    check_type(PY_TYPE::DICT);
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
            case PY_TYPE::TENSOR:
                return _list.cast<std::vector<ov::Tensor>>();
            case PY_TYPE::DICT: {
                std::vector<ov::AnyMap> any_map_list(_list.size());
                for (const auto& item : _list) {
                    any_map_list.push_back(py_object_to_any_map(item.cast<py::dict>()));
                }
                return any_map_list;
            }
            default:
                OPENVINO_THROW("Property \"" + property_name + "\" got unsupported type.");
            }
        }

    } else if (py::isinstance<py::dict>(py_obj) && any_map_properties.find(property_name) == any_map_properties.end()) {
        auto _dict = py_obj.cast<py::dict>();
        enum class PY_TYPE : int { UNKNOWN = 0, STR, INT};
        PY_TYPE detected_key_type = PY_TYPE::UNKNOWN;
        PY_TYPE detected_value_type = PY_TYPE::UNKNOWN;
        for (const auto& it : _dict) {
            auto check_type = [&](PY_TYPE type, PY_TYPE& detected_type) {
                if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                    detected_type = type;
                    return;
                }
                OPENVINO_THROW("Incorrect value in \"" + property_name + "\". Mixed types in the dict are not allowed.");
            };
            // check key type
            if (py::isinstance<py::str>(it.first)) {
                check_type(PY_TYPE::STR, detected_key_type);
            }

            // check value type
            if (py::isinstance<py::int_>(it.second)) {
                check_type(PY_TYPE::INT, detected_value_type);
            }
        }
        if (_dict.empty()) {
            return ov::Any();
        }

        switch (detected_key_type) {
        case PY_TYPE::STR:
            switch (detected_value_type) {
            case PY_TYPE::INT:
                return _dict.cast<std::map<std::string, int64_t>>();
            default:
                OPENVINO_THROW("Property \"" + property_name + "\" got unsupported type.");
            }
        default:
            OPENVINO_THROW("Property \"" + property_name + "\" got unsupported type.");
        }
    } else if (py::isinstance<py::set>(py_obj)) {
        auto _set = py_obj.cast<py::set>();
        enum class PY_TYPE : int { UNKNOWN = 0, STR, INT, FLOAT, BOOL};
        PY_TYPE detected_type = PY_TYPE::UNKNOWN;
        for (const auto& it : _set) {
            auto check_type = [&](PY_TYPE type) {
                if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                    detected_type = type;
                    return;
                }
                OPENVINO_THROW("Incorrect value in \"" + property_name + "\". Mixed types in the set are not allowed.");
            };
            if (py::isinstance<py::str>(it)) {
                check_type(PY_TYPE::STR);
            } else if (py::isinstance<py::int_>(it)) {
                check_type(PY_TYPE::INT);
            } else if (py::isinstance<py::float_>(it)) {
                check_type(PY_TYPE::FLOAT);
            } else if (py::isinstance<py::bool_>(it)) {
                check_type(PY_TYPE::BOOL);
            }
        }

        if (_set.empty())
            return ov::Any();

        switch (detected_type) {
        case PY_TYPE::STR:
            return _set.cast<std::set<std::string>>();
        case PY_TYPE::FLOAT:
            return _set.cast<std::set<double>>();
        case PY_TYPE::INT:
            return _set.cast<std::set<int64_t>>();
        case PY_TYPE::BOOL:
            return _set.cast<std::set<bool>>();
        default:
            OPENVINO_THROW("Property \"" + property_name + "\" got unsupported type.");
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
    } else if (py::isinstance<ov::Tensor>(py_obj)) {
        return py::cast<ov::Tensor>(py_obj);
    } else if (py::isinstance<ov::Output<ov::Node>>(py_obj)) {
        return py::cast<ov::Output<ov::Node>>(py_obj);
    } else if (py::isinstance<ov::genai::SchedulerConfig>(py_obj)) {
        return py::cast<ov::genai::SchedulerConfig>(py_obj);
    } else if (py::isinstance<ov::genai::AdapterConfig>(py_obj)) {
        return py::cast<ov::genai::AdapterConfig>(py_obj);
    } else if (py::isinstance<ov::genai::StructuralTagItem>(py_obj)) {
        return py::cast<ov::genai::StructuralTagItem>(py_obj);
    } else if (py::isinstance<ov::genai::StructuralTagsConfig>(py_obj)) {
        // For structural_tags_config property, wrap in variant
        if (property_name == "structural_tags_config") {
            std::variant<ov::genai::StructuralTagsConfig, ov::genai::StructuredOutputConfig::StructuralTag> variant_value = 
                py::cast<ov::genai::StructuralTagsConfig>(py_obj);
            return variant_value;
        }
        return py::cast<ov::genai::StructuralTagsConfig>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::Regex>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::EBNF>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::JSONSchema>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::ConstString>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::AnyText>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::QwenXMLParametersFormat>(py_obj)
               // python does not use std::shared_ptr to obj
               || py::isinstance<ov::genai::StructuredOutputConfig::Union>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::Concat>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::Tag>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::TriggeredTags>(py_obj)
               || py::isinstance<ov::genai::StructuredOutputConfig::TagsWithSeparator>(py_obj)) {
        // For structural_tags_config property, wrap in variant
        if (property_name == "structural_tags_config") {
            std::variant<ov::genai::StructuralTagsConfig, ov::genai::StructuredOutputConfig::StructuralTag> variant_value = 
                py_obj_to_structural_tag(py_obj);
            return variant_value;
        }
        return py_obj_to_structural_tag(py_obj);
    } else if (py::isinstance<ov::genai::GenerationConfig>(py_obj)) {
        return py::cast<ov::genai::GenerationConfig>(py_obj);
    } else if (py::isinstance<ov::genai::ImageGenerationConfig>(py_obj)) {
        return py::cast<ov::genai::ImageGenerationConfig>(py_obj);
    } else if (py::isinstance<ov::genai::WhisperGenerationConfig>(py_obj)) {
        return py::cast<ov::genai::WhisperGenerationConfig>(py_obj);
    } else if (py::isinstance<ov::genai::TextEmbeddingPipeline::PoolingType>(py_obj)) {
        return py::cast<ov::genai::TextEmbeddingPipeline::PoolingType>(py_obj);
    } else if (py::isinstance<ov::genai::StopCriteria>(py_obj)) {
        return py::cast<ov::genai::StopCriteria>(py_obj);
    } else if (py::isinstance<ov::genai::Generator>(py_obj)) {
        return py::cast<std::shared_ptr<ov::genai::Generator>>(py_obj);
    } else if (py::isinstance<py::function>(py_obj) && property_name == "callback") {
        return py::cast<std::function<bool(size_t, size_t, ov::Tensor&)>>(py_obj);
    } else if ((py::isinstance<py::function>(py_obj) || py::isinstance<ov::genai::StreamerBase>(py_obj) || py::isinstance<std::monostate>(py_obj)) && property_name == "streamer") {
        auto streamer = py::cast<ov::genai::pybind::utils::PyBindStreamerVariant>(py_obj);
        return ov::genai::streamer(pystreamer_to_streamer(streamer)).second;
    } else if (py::isinstance(py_obj, py::module_::import("pathlib").attr("Path"))) {
        return py::cast<std::filesystem::path>(py_obj);
    }
    OPENVINO_THROW(py::str(py_obj.get_type()), " isn't supported for argument ", property_name);
}

void add_deprecation_warning_for_chunk_streamer(std::shared_ptr<StreamerBase> streamer) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (auto chunk_streamer = std::dynamic_pointer_cast<ov::genai::ChunkStreamerBase>(streamer)) {
        PyErr_WarnEx(PyExc_DeprecationWarning, "ChunkStreamerBase is deprecated and will be removed in 2026.0.0 release. Use StreamerBase instead.", 1);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
}

ov::AnyMap properties_to_any_map(const std::map<std::string, py::object>& properties) {
    ov::AnyMap properties_to_cpp;
    for (const auto& property : properties) {
        properties_to_cpp[property.first] = py_object_to_any(property.second, property.first);
    }
    return properties_to_cpp;
}

ov::AnyMap kwargs_to_any_map(const py::kwargs& kwargs) {
    ov::AnyMap params = {};

    for (const auto& item : kwargs) {
        std::string key = py::cast<std::string>(item.first);
        py::object value = py::cast<py::object>(item.second);
        // we need to unpack only dictionaries, which are passed with "config" name,
        // because there are dictionary properties that should not be unpacked
        if (utils::py_object_is_any_map(value) && key == "config") {
            auto map = utils::py_object_to_any_map(value);
            params.insert(map.begin(), map.end());
        } else if (py::isinstance<ov::genai::StructuredOutputConfig>(value)) {
            params[key] = py::cast<ov::genai::StructuredOutputConfig>(value);
        } else {
            if (py::isinstance<py::none>(value)) {
                OPENVINO_ASSERT(!py::isinstance<py::none>(value), "Property \"", key, "\" can't be None.");
            }
            params[key] = utils::py_object_to_any(value, key);
        }

    }
    return params;
}

std::filesystem::path ov_tokenizers_module_path() {
    // Try a path relative to build artifacts folder first.
    std::filesystem::path from_relative = tokenizers_relative_to_genai();
    if (std::filesystem::exists(from_relative)) {
        return from_relative;
    }
    return py::module_::import("openvino_tokenizers").attr("_ext_path").cast<std::filesystem::path>();
}

ov::genai::StreamerVariant pystreamer_to_streamer(const PyBindStreamerVariant& py_streamer) {
    ov::genai::StreamerVariant streamer = std::monostate();

    std::visit(overloaded {
        [&streamer](const std::function<std::optional<uint16_t>(py::str)>& py_callback){
            // Wrap python streamer with manual utf-8 decoding. Do not rely
            // on pybind automatic decoding since it raises exceptions on incomplete strings.
            auto callback_wrapped = [py_callback](std::string subword) -> ov::genai::StreamingStatus {
                py::gil_scoped_acquire acquire;
                auto py_str = PyUnicode_DecodeUTF8(subword.data(), subword.length(), "replace");
                std::optional<uint16_t> callback_output = py_callback(py::reinterpret_borrow<py::str>(py_str));
                if (callback_output.has_value()) {
                    if (*callback_output == (uint16_t)StreamingStatus::RUNNING)
                        return StreamingStatus::RUNNING;
                    else if (*callback_output == (uint16_t)StreamingStatus::CANCEL)
                        return StreamingStatus::CANCEL;
                    return StreamingStatus::STOP;
                } else {
                    return StreamingStatus::RUNNING;
                }
            };
            streamer = callback_wrapped;
        },
        [&streamer](std::shared_ptr<StreamerBase> streamer_cls){
            add_deprecation_warning_for_chunk_streamer(streamer_cls);
            streamer = streamer_cls;
        },
        [](std::monostate none){ /*streamer is already a monostate */ }
    }, py_streamer);
    return streamer;
}

StructuredOutputConfig::StructuralTag py_obj_to_structural_tag(const py::object& py_obj) {
    if (py::isinstance<ov::genai::StructuredOutputConfig::Regex>(py_obj)) {
        return py::cast<ov::genai::StructuredOutputConfig::Regex>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::JSONSchema>(py_obj)) {
        return py::cast<ov::genai::StructuredOutputConfig::JSONSchema>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::EBNF>(py_obj)) {
        return py::cast<ov::genai::StructuredOutputConfig::EBNF>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::ConstString>(py_obj)) {
        return py::cast<ov::genai::StructuredOutputConfig::ConstString>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::AnyText>(py_obj)) {
        return py::cast<ov::genai::StructuredOutputConfig::AnyText>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::QwenXMLParametersFormat>(py_obj)) {
        return py::cast<ov::genai::StructuredOutputConfig::QwenXMLParametersFormat>(py_obj);
    } else if (py::isinstance<py::str>(py_obj)) {
        return py::cast<std::string>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::Concat>(py_obj)) {
        return py::cast<std::shared_ptr<ov::genai::StructuredOutputConfig::Concat>>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::Union>(py_obj)) {
        return py::cast<std::shared_ptr<ov::genai::StructuredOutputConfig::Union>>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::Tag>(py_obj)) {
        return py::cast<std::shared_ptr<ov::genai::StructuredOutputConfig::Tag>>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::TriggeredTags>(py_obj)) {
        return py::cast<std::shared_ptr<ov::genai::StructuredOutputConfig::TriggeredTags>>(py_obj);
    } else if (py::isinstance<ov::genai::StructuredOutputConfig::TagsWithSeparator>(py_obj)) {
        return py::cast<std::shared_ptr<ov::genai::StructuredOutputConfig::TagsWithSeparator>>(py_obj);
    } else {
        OPENVINO_THROW(py_obj.get_type(), " type isn't supported for StructuralTag: ", py::str(py_obj));
    }
}

ov::genai::GenerationConfig update_config_from_kwargs(ov::genai::GenerationConfig config, const py::kwargs& kwargs) {
    config.update_generation_config(kwargs_to_any_map(kwargs));
    return config;
}

ov::genai::JsonContainer py_object_to_json_container(const py::object& obj) {
    if (obj.is_none()) {
        return JsonContainer();
    }
    // TODO Consider using direct native JsonContainer conversion instead of string serialization
    py::module_ json_module = py::module_::import("json");
    std::string json_string = py::cast<std::string>(json_module.attr("dumps")(obj));
    return JsonContainer::from_json_string(json_string);
}

py::object json_container_to_py_object(const ov::genai::JsonContainer& container) {
    // TODO Consider using direct native JsonContainer conversion instead of string serialization
    std::string json_string = container.to_json_string();
    py::module_ json_module = py::module_::import("json");
    return json_module.attr("loads")(json_string);
}

}  // namespace ov::genai::pybind::utils
