// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/helper.hpp"

#include "include/addon.hpp"
#include "include/chat_history.hpp"
#include "include/parser.hpp"
#include "include/perf_metrics.hpp"
#include "include/vlm_pipeline/perf_metrics.hpp"

namespace {
constexpr const char* JS_SCHEDULER_CONFIG_KEY = "schedulerConfig";
constexpr const char* CPP_SCHEDULER_CONFIG_KEY = "scheduler_config";
constexpr const char* POOLING_TYPE_KEY = "pooling_type";
constexpr const char* STRUCTURED_OUTPUT_CONFIG_KEY = "structured_output_config";
constexpr const char* PARSERS_KEY = "parsers";

}  // namespace

template <>
ov::Any js_to_cpp<ov::Any>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsString()) {
        return ov::Any(value.ToString().Utf8Value());
    } else if (value.IsBigInt()) {
        Napi::BigInt big_value = value.As<Napi::BigInt>();
        bool is_lossless;
        int64_t big_num = big_value.Int64Value(&is_lossless);

        if (!is_lossless) {
            OPENVINO_THROW("Result of BigInt conversion to int64_t results in a loss of precision");
        }

        return ov::Any(big_num);
    } else if (value.IsNumber()) {
        Napi::Number num = value.ToNumber();

        if (is_napi_value_int(env, value)) {
            return ov::Any(num.Int32Value());
        } else {
            return ov::Any(num.DoubleValue());
        }
    } else if (value.IsBoolean()) {
        return ov::Any(static_cast<bool>(value.ToBoolean()));
    } else if (value.IsArray()) {
        return ov::Any(js_to_cpp<std::vector<std::string>>(env, value));
    } else if (value.IsObject()) {
        if (value.ToString().Utf8Value() == "[object Set]") {
            try {
                // try to cast to set of strings
                auto object_value = value.As<Napi::Object>();
                auto values = object_value.Get("values").As<Napi::Function>();
                auto iterator = values.Call(object_value, {}).As<Napi::Object>();
                auto next = iterator.Get("next").As<Napi::Function>();
                auto size = object_value.Get("size").As<Napi::Number>().Int32Value();

                std::set<std::string> set;
                for (uint32_t i = 0; i < size; ++i) {
                    auto item = next.Call(iterator, {}).As<Napi::Object>();
                    set.insert(item.Get("value").As<Napi::String>().Utf8Value());
                }

                return ov::Any(set);
            } catch (std::exception& e) {
                std::cerr << "Cannot convert to set: " << e.what() << std::endl;
            }
        }
    }
    OPENVINO_THROW("Cannot convert to ov::Any");
}

template <>
ov::AnyMap js_to_cpp<ov::AnyMap>(const Napi::Env& env, const Napi::Value& value) {
    std::map<std::string, ov::Any> result_map;
    if(value.IsUndefined() || value.IsNull()) {
        return result_map;
    }
    if (!value.IsObject()) {
        OPENVINO_THROW("Passed Napi::Value must be an object.");
    }
    const auto& object = value.ToObject();
    const auto& keys = object.GetPropertyNames();

    for (uint32_t i = 0; i < keys.Length(); ++i) {
        const std::string& key_name = keys.Get(i).ToString();
        auto value_by_key = object.Get(key_name);
        if (value_by_key.IsUndefined() || value_by_key.IsNull()) {
            continue;
        }
        if (key_name == JS_SCHEDULER_CONFIG_KEY) {
            result_map[CPP_SCHEDULER_CONFIG_KEY] = js_to_cpp<ov::genai::SchedulerConfig>(env, value_by_key);
        } else if (key_name == POOLING_TYPE_KEY) {
            result_map[key_name] =
                ov::genai::TextEmbeddingPipeline::PoolingType(value_by_key.ToNumber().Int32Value());
        } else if (key_name == STRUCTURED_OUTPUT_CONFIG_KEY) {
            result_map[key_name] = js_to_cpp<ov::genai::StructuredOutputConfig>(env, value_by_key);
        } else if (key_name == PARSERS_KEY) {
            result_map[key_name] = js_to_cpp<std::vector<std::shared_ptr<ov::genai::Parser>>>(env, value_by_key);
        } else {
            result_map[key_name] = js_to_cpp<ov::Any>(env, value_by_key);
        }
    }

    return result_map;
}

template <>
std::string js_to_cpp<std::string>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsString(), "Passed argument must be of type String.");
    return value.As<Napi::String>().Utf8Value();
}

template <>
int64_t js_to_cpp<int64_t>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsNumber() || value.IsBigInt(), "Passed argument must be of type Number or BigInt.");
    if (value.IsNumber()) {
        return value.As<Napi::Number>().Int64Value();
    } 
    bool lossless;
    auto result = value.As<Napi::BigInt>().Int64Value(&lossless);
    OPENVINO_ASSERT(lossless, "BigInt value is too large to fit in int64_t without precision loss.");
    return result;
}

template <>
std::vector<std::string> js_to_cpp<std::vector<std::string>>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsArray()) {
        auto array = value.As<Napi::Array>();
        size_t arrayLength = array.Length();

        std::vector<std::string> nativeArray;
        for (uint32_t i = 0; i < arrayLength; ++i) {
            Napi::Value arrayItem = array[i];
            if (!arrayItem.IsString()) {
                OPENVINO_THROW(std::string("Passed array must contain only strings."));
            }
            nativeArray.push_back(arrayItem.As<Napi::String>().Utf8Value());
        }
        return nativeArray;

    } else {
        OPENVINO_THROW("Passed argument must be of type Array or TypedArray.");
    }
}

template <>
std::vector<int64_t> js_to_cpp<std::vector<int64_t>>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsArray(), "Passed argument must be of type Array.");
    auto array = value.As<Napi::Array>();
    size_t arrayLength = array.Length();

    std::vector<int64_t> vector;
    vector.reserve(arrayLength);
    for (uint32_t i = 0; i < arrayLength; ++i) {
        vector.push_back(js_to_cpp<int64_t>(env, array[i]));
    }
    return vector;
}

template <>
ov::genai::JsonContainer js_to_cpp<ov::genai::JsonContainer>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsObject() || value.IsArray(), "JsonContainer must be a JS object or an array but got " + std::string(value.ToString().Utf8Value()));
    // TODO Consider using direct native JsonContainer conversion instead of string serialization
    return ov::genai::JsonContainer::from_json_string(json_stringify(env, value));
}

template <>
ov::genai::SchedulerConfig js_to_cpp<ov::genai::SchedulerConfig>(const Napi::Env& env, const Napi::Value& value) {
    ov::genai::SchedulerConfig config;
    OPENVINO_ASSERT(value.IsObject(), "SchedulerConfig must be a JS object");
    auto obj = value.As<Napi::Object>();

    if (obj.Has("max_num_batched_tokens")) {
        config.max_num_batched_tokens = obj.Get("max_num_batched_tokens").ToNumber().Uint32Value();
    }
    if (obj.Has("num_kv_blocks")) {
        config.num_kv_blocks = obj.Get("num_kv_blocks").ToNumber().Uint32Value();
    }
    if (obj.Has("cache_size")) {
        config.cache_size = obj.Get("cache_size").ToNumber().Uint32Value();
    }
    if (obj.Has("dynamic_split_fuse")) {
        config.dynamic_split_fuse = obj.Get("dynamic_split_fuse").ToBoolean().Value();
    }

    return config;
}

template <>
ov::genai::StructuredOutputConfig::Tag js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsObject(), "Tag must be a JS object");
    auto obj = value.As<Napi::Object>();

    return ov::genai::StructuredOutputConfig::Tag(
        js_to_cpp<std::string>(env, obj.Get("begin")),
        js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, obj.Get("content")),
        js_to_cpp<std::string>(env, obj.Get("end"))
    );
}

template <>
ov::genai::StructuredOutputConfig::StructuralTag js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsString()) {
        return js_to_cpp<std::string>(env, value);
    }
    
    OPENVINO_ASSERT(value.IsObject(), "StructuralTag must be a JS object or string");
    auto obj = value.As<Napi::Object>();

    std::string tag_type = obj.Get("structuralTagType").ToString().Utf8Value();

    if (tag_type == "Regex") {
        return ov::genai::StructuredOutputConfig::Regex(
            js_to_cpp<std::string>(env, obj.Get("value"))
        );
    } else if (tag_type == "JSONSchema") {
        return ov::genai::StructuredOutputConfig::JSONSchema(
            js_to_cpp<std::string>(env, obj.Get("value"))
        );
    } else if (tag_type == "EBNF") {
        return ov::genai::StructuredOutputConfig::EBNF(
            js_to_cpp<std::string>(env, obj.Get("value"))
        );
    } else if (tag_type == "ConstString") {
        return ov::genai::StructuredOutputConfig::ConstString(
            js_to_cpp<std::string>(env, obj.Get("value"))
        );
    } else if (tag_type == "AnyText") {
        return ov::genai::StructuredOutputConfig::AnyText();
    } else if (tag_type == "QwenXMLParametersFormat") {
        return ov::genai::StructuredOutputConfig::QwenXMLParametersFormat(
            js_to_cpp<std::string>(env, obj.Get("jsonSchema"))
        );
    } else if (tag_type == "Concat") {
        std::vector<ov::genai::StructuredOutputConfig::StructuralTag> elements;
        auto js_elements = obj.Get("elements");
        OPENVINO_ASSERT(js_elements.IsArray(), "Concat StructuralTag 'elements' must be an array");
        auto js_array = js_elements.As<Napi::Array>();
        size_t arrayLength = js_array.Length();
        for (uint32_t i = 0; i < arrayLength; ++i) {
            elements.push_back(js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, js_array[i]));
        }
        return std::make_shared<ov::genai::StructuredOutputConfig::Concat>(elements);
    } else if (tag_type == "Union") {
        std::vector<ov::genai::StructuredOutputConfig::StructuralTag> elements;
        auto js_elements = obj.Get("elements");
        OPENVINO_ASSERT(js_elements.IsArray(), "Union StructuralTag 'elements' must be an array");
        auto js_array = js_elements.As<Napi::Array>();
        size_t arrayLength = js_array.Length();
        for (uint32_t i = 0; i < arrayLength; ++i) {
            elements.push_back(js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, js_array[i]));
        }
        return std::make_shared<ov::genai::StructuredOutputConfig::Union>(elements);
    } else if (tag_type == "Tag") {
        return  std::make_shared<ov::genai::StructuredOutputConfig::Tag>(js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(env, obj));
    } else if (tag_type == "TriggeredTags") {
        std::vector<ov::genai::StructuredOutputConfig::Tag> tags;
        auto js_tags = obj.Get("tags");
        auto triggers = js_to_cpp<std::vector<std::string>>(env, obj.Get("triggers"));
        auto at_least_one = obj.Get("atLeastOne");
        auto stop_after_first = obj.Get("stopAfterFirst");
        OPENVINO_ASSERT(
            at_least_one.IsBoolean() && stop_after_first.IsBoolean(),
            "TriggeredTags 'atLeastOne', and 'stopAfterFirst' must be booleans"
        );
        OPENVINO_ASSERT(js_tags.IsArray(), "TriggeredTags 'tags' must be an array");
        auto js_array = js_tags.As<Napi::Array>();
        size_t arrayLength = js_array.Length();
        for (uint32_t i = 0; i < arrayLength; ++i) {
            tags.push_back(js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(env, js_array[i]));
        }
        return std::make_shared<ov::genai::StructuredOutputConfig::TriggeredTags>(
            triggers,
            tags,
            at_least_one.As<Napi::Boolean>().Value(),
            stop_after_first.As<Napi::Boolean>().Value()
        );
    } else if (tag_type == "TagsWithSeparator") {
        std::vector<ov::genai::StructuredOutputConfig::Tag> tags;
        auto separator = js_to_cpp<std::string>(env, obj.Get("separator"));
        auto at_least_one = obj.Get("atLeastOne");
        auto stop_after_first = obj.Get("stopAfterFirst");
        OPENVINO_ASSERT(
            at_least_one.IsBoolean() && stop_after_first.IsBoolean(),
            "TagsWithSeparator 'atLeastOne', and 'stopAfterFirst' must be booleans"
        );

        auto js_tags = obj.Get("tags");
        OPENVINO_ASSERT(js_tags.IsArray(), "TagsWithSeparator 'tags' must be an array");
        auto js_array = js_tags.As<Napi::Array>();
        size_t arrayLength = js_array.Length();
        for (uint32_t i = 0; i < arrayLength; ++i) {
            tags.push_back(js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(env, js_array[i]));
        }

        return std::make_shared<ov::genai::StructuredOutputConfig::TagsWithSeparator>(
            tags,
            separator,
            at_least_one.As<Napi::Boolean>().Value(),
            stop_after_first.As<Napi::Boolean>().Value()
        );
    } else {
        OPENVINO_THROW("Unknown StructuralTag type: " + tag_type);
    }
}

template <>
ov::genai::StructuredOutputConfig js_to_cpp<ov::genai::StructuredOutputConfig>(const Napi::Env& env, const Napi::Value& value) {
    ov::genai::StructuredOutputConfig config;
    OPENVINO_ASSERT(value.IsObject(), "StructuredOutputConfig must be a JS object");
    auto obj = value.As<Napi::Object>();

    if (obj.Has("json_schema") && !obj.Get("json_schema").IsUndefined()) {
        config.json_schema = js_to_cpp<std::string>(env, obj.Get("json_schema"));
    }
    if (obj.Has("regex") && !obj.Get("regex").IsUndefined()) {
        config.regex = js_to_cpp<std::string>(env, obj.Get("regex"));
    }
    if (obj.Has("grammar") && !obj.Get("grammar").IsUndefined()) {
        config.grammar = js_to_cpp<std::string>(env, obj.Get("grammar"));
    }
    if (obj.Has("structural_tags_config") && !obj.Get("structural_tags_config").IsUndefined()) {
        config.structural_tags_config = js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, obj.Get("structural_tags_config"));
    }

    return config;
}

template <>
ov::Tensor js_to_cpp<ov::Tensor>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsObject(), "Passed argument must be an object.");

    auto tensor_wrap = value.As<Napi::Object>();
    auto tensor_prototype = get_prototype_from_ov_addon(env, "Tensor");
    OPENVINO_ASSERT(tensor_wrap.InstanceOf(tensor_prototype), "Passed argument is not of type Tensor");

    Napi::Value get_external_tensor_val = tensor_wrap.Get("__getExternalTensor");
    OPENVINO_ASSERT(get_external_tensor_val.IsFunction(), "Tensor object does not have a '__getExternalTensor' function. This may indicate an incompatible or outdated openvino-node version.");
    auto native_tensor_func = get_external_tensor_val.As<Napi::Function>();
    Napi::Value native_tensor_value = native_tensor_func.Call(tensor_wrap, {});
    OPENVINO_ASSERT(native_tensor_value.IsExternal(), "__getExternalTensor() did not return an External object.");

    auto external = native_tensor_value.As<Napi::External<ov::Tensor>>();
    auto tensor_ptr = external.Data();
    return *tensor_ptr;
}

template <>
std::shared_ptr<ov::genai::Parser> js_to_cpp<std::shared_ptr<ov::genai::Parser>>(const Napi::Env& env,
                                                                                 const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsObject(), "Parser must be a JS object with a 'parse' method");
    Napi::Object obj = value.As<Napi::Object>();
    
    // Check if it's a native parser instance
    auto native_parser = get_native_parser(env, obj);
    if (native_parser) {
        return native_parser;
    }

    // Treat as custom JS parser (including JS subclasses)
    OPENVINO_ASSERT(obj.Has("parse"), "Parser object must have a 'parse' method");
    Napi::Value parse_method = obj.Get("parse");
    OPENVINO_ASSERT(parse_method.IsFunction(), "'parse' property of Parser object must be a function");
    return std::make_shared<JSParser>(env, obj);
}

template <>
std::vector<std::shared_ptr<ov::genai::Parser>> js_to_cpp<std::vector<std::shared_ptr<ov::genai::Parser>>>(
    const Napi::Env& env,
    const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsArray(), "Passed argument must be of type Array.");
    Napi::Array arr = value.As<Napi::Array>();
    std::vector<std::shared_ptr<ov::genai::Parser>> parsers;
    parsers.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i) {
        parsers.push_back(js_to_cpp<std::shared_ptr<ov::genai::Parser>>(env, arr[i]));
    }
    return parsers;
}

template <>
std::vector<ov::Tensor> js_to_cpp<std::vector<ov::Tensor>>(const Napi::Env& env, const Napi::Value& value) {
    std::vector<ov::Tensor> tensors;
    if (value.IsUndefined() || value.IsNull()) {
        return tensors;
    }
    if (value.IsArray()) {
        auto array = value.As<Napi::Array>();
        size_t length = array.Length();
        tensors.reserve(length);
        for (uint32_t i = 0; i < length; ++i) {
            tensors.push_back(js_to_cpp<ov::Tensor>(env, array[i]));
        }
    } else {
        OPENVINO_THROW("Passed argument must be an array of Tensors.");
    }
    return tensors;
}

template <>
ov::genai::PerfMetrics& unwrap<ov::genai::PerfMetrics>(const Napi::Env& env, const Napi::Value& value) {
    const auto obj = value.As<Napi::Object>();
    const auto& prototype = env.GetInstanceData<AddonData>()->perf_metrics;

    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    OPENVINO_ASSERT(obj.InstanceOf(prototype.Value().As<Napi::Function>()),
                    "Passed argument is not of type PerfMetrics");

    const auto js_metrics = Napi::ObjectWrap<PerfMetricsWrapper>::Unwrap(obj);
    return js_metrics->get_value();
}

template <>
ov::genai::VLMPerfMetrics& unwrap<ov::genai::VLMPerfMetrics>(const Napi::Env& env, const Napi::Value& value) {
    const auto obj = value.As<Napi::Object>();
    const auto& prototype = env.GetInstanceData<AddonData>()->vlm_perf_metrics;
    OPENVINO_ASSERT(prototype, "Invalid pointer to prototype.");
    OPENVINO_ASSERT(obj.InstanceOf(prototype.Value().As<Napi::Function>()),
                    "Passed argument is not of type VLMPerfMetrics");
    const auto js_metrics = Napi::ObjectWrap<VLMPerfMetricsWrapper>::Unwrap(obj);
    return js_metrics->get_value();
}

template <>
ov::genai::ChatHistory& unwrap<ov::genai::ChatHistory>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsObject(), "Passed argument must be an object.");
    const auto obj = value.As<Napi::Object>();
    OPENVINO_ASSERT(is_chat_history(env, value), "Passed argument is not of type ChatHistory");

    const auto chat_history = Napi::ObjectWrap<ChatHistoryWrap>::Unwrap(obj);
    return chat_history->get_value();
}

template <>
GenerateInputs js_to_cpp<GenerateInputs>(const Napi::Env& env, const Napi::Value& value) {
    try {
        if (value.IsString()) {
            return value.As<Napi::String>().Utf8Value();
        } else if (value.IsArray()) {
            return js_to_cpp<std::vector<std::string>>(env, value);
        } else if (is_chat_history(env, value)) {
            return unwrap<ov::genai::ChatHistory>(env, value);
        }
        OPENVINO_THROW("Passed argument must be a string, ChatHistory or an array of strings.");
    } catch (const ov::Exception& e) {
        OPENVINO_THROW("An incorrect input value has been passed. ", e.what());
    }
}

template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(const Napi::Env& env,
                                                               const ov::genai::EmbeddingResult& embedding_result) {
    return std::visit(overloaded{[env](std::vector<float> embed_vector) -> Napi::Value {
                                     auto vector_size = embed_vector.size();
                                     auto buffer = Napi::ArrayBuffer::New(env, vector_size * sizeof(float));
                                     std::memcpy(buffer.Data(), embed_vector.data(), vector_size * sizeof(float));
                                     Napi::Value typed_array = Napi::Float32Array::New(env, vector_size, buffer, 0);
                                     return typed_array;
                                 },
                                 [env](std::vector<int8_t> embed_vector) -> Napi::Value {
                                     auto buffer_size = embed_vector.size();
                                     auto buffer = Napi::ArrayBuffer::New(env, buffer_size * sizeof(int8_t));
                                     std::memcpy(buffer.Data(), embed_vector.data(), buffer_size * sizeof(int8_t));
                                     Napi::Value typed_array = Napi::Int8Array::New(env, buffer_size, buffer, 0);
                                     return typed_array;
                                 },
                                 [env](std::vector<uint8_t> embed_vector) -> Napi::Value {
                                     auto buffer_size = embed_vector.size();
                                     auto buffer = Napi::ArrayBuffer::New(env, buffer_size * sizeof(uint8_t));
                                     std::memcpy(buffer.Data(), embed_vector.data(), buffer_size * sizeof(uint8_t));
                                     Napi::Value typed_array = Napi::Uint8Array::New(env, buffer_size, buffer, 0);
                                     return typed_array;
                                 },
                                 [env](auto& args) -> Napi::Value {
                                     OPENVINO_THROW("Unsupported type for EmbeddingResult.");
                                 }},
                      embedding_result);
}

template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(const Napi::Env& env,
                                                                const ov::genai::EmbeddingResults& embedding_result) {
    return std::visit(
        [env](auto& embed_vector) {
            auto js_result = Napi::Array::New(env, embed_vector.size());
            for (auto i = 0; i < embed_vector.size(); i++) {
                js_result[i] = cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(env, embed_vector[i]);
            }
            return js_result;
        },
        embedding_result);
}

template <>
Napi::Value cpp_to_js<std::vector<std::string>, Napi::Value>(const Napi::Env& env,
                                                             const std::vector<std::string>& value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::String::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<float>, Napi::Value>(const Napi::Env& env, const std::vector<float>& value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<double>, Napi::Value>(const Napi::Env& env, const std::vector<double>& value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<size_t>, Napi::Value>(const Napi::Env& env, const std::vector<size_t>& value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (auto i = 0; i < value.size(); i++) {
        js_array[i] = Napi::Number::New(env, value[i]);
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<std::vector<std::pair<size_t, float>>, Napi::Value>(
    const Napi::Env& env,
    const std::vector<std::pair<size_t, float>>& rerank_results) {
    auto js_array = Napi::Array::New(env, rerank_results.size());
    for (size_t i = 0; i < rerank_results.size(); ++i) {
        const auto& [index, score] = rerank_results[i];
        auto tuple = Napi::Array::New(env, 2);
        tuple.Set((uint32_t)0, Napi::Number::New(env, index));
        tuple.Set((uint32_t)1, Napi::Number::New(env, score));
        js_array[i] = tuple;
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<ov::genai::JsonContainer, Napi::Value>(const Napi::Env& env, const ov::genai::JsonContainer& json_container) {
    return json_parse(env, json_container.to_json_string());
}

template <>
Napi::Value cpp_to_js<std::vector<ov::genai::JsonContainer>, Napi::Value>(
    const Napi::Env& env,
    const std::vector<ov::genai::JsonContainer>& value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (size_t i = 0; i < value.size(); i++) {
        js_array[i] = json_parse(env, value[i].to_json_string());
    }
    return js_array;
}

template <>
Napi::Value cpp_to_js<ov::Tensor, Napi::Value>(const Napi::Env& env, const ov::Tensor& tensor) {
    try {
        auto prototype = get_prototype_from_ov_addon(env, "Tensor");

        auto external = Napi::External<ov::Tensor>::New(env, new ov::Tensor(tensor),
                [](Napi::Env /*env*/, ov::Tensor* external_tensor) {
                    delete external_tensor;
                });
        auto tensor_wrap = prototype.New({ external });

        return tensor_wrap;
    } catch (const ov::Exception& e) {
        Napi::Error::New(env, std::string("Cannot create Tensor wrapper: ") + e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

template <>
Napi::Value cpp_to_js<ov::genai::TokenizedInputs, Napi::Value>(const Napi::Env& env, const ov::genai::TokenizedInputs& tokenized_inputs) {
    auto js_object = Napi::Object::New(env);

    js_object.Set("input_ids", cpp_to_js<ov::Tensor, Napi::Value>(env, tokenized_inputs.input_ids));
    js_object.Set("attention_mask", cpp_to_js<ov::Tensor, Napi::Value>(env, tokenized_inputs.attention_mask));
    // token_type_ids is optional and present only for paired inputs
    if (tokenized_inputs.token_type_ids.has_value()) {
        js_object.Set("token_type_ids", cpp_to_js<ov::Tensor, Napi::Value>(env, tokenized_inputs.token_type_ids.value()));
    }

    return js_object;
}

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num) {
    return env.Global().Get("Number").ToObject().Get("isInteger").As<Napi::Function>().Call({num}).ToBoolean().Value();
}

bool is_chat_history(const Napi::Env& env, const Napi::Value& value) {
    const auto obj = value.As<Napi::Object>();
    const auto& prototype = env.GetInstanceData<AddonData>()->chat_history;

    OPENVINO_ASSERT(prototype, "Invalid pointer to ChatHistory prototype.");

    return obj.InstanceOf(prototype.Value().As<Napi::Function>());
}

// Get native parser or return nullptr
std::shared_ptr<ov::genai::Parser> get_native_parser(const Napi::Env& env, const Napi::Object& object) {
    const auto addon_data = env.GetInstanceData<AddonData>();

    // Check ReasoningParser
    const auto& reasoning_parser_prototype = addon_data->reasoning_parser;
    OPENVINO_ASSERT(reasoning_parser_prototype, "Invalid pointer to ReasoningParser prototype.");
    if (object.Get("constructor").StrictEquals(reasoning_parser_prototype.Value())) {
        auto parser_wrapper = Napi::ObjectWrap<ReasoningParserWrapper>::Unwrap(object);
        return parser_wrapper->get_parser();
    }

    // Check DeepSeekR1ReasoningParser
    const auto& deepseek_parser_prototype = addon_data->deepseek_r1_reasoning_parser;
    OPENVINO_ASSERT(deepseek_parser_prototype, "Invalid pointer to DeepSeekR1ReasoningParser prototype.");
    if (object.Get("constructor").StrictEquals(deepseek_parser_prototype.Value())) {
        auto parser_wrapper = Napi::ObjectWrap<DeepSeekR1ReasoningParserWrapper>::Unwrap(object);
        return parser_wrapper->get_parser();
    }

    // Check Phi4ReasoningParser
    const auto& phi4_parser_prototype = addon_data->phi4_reasoning_parser;
    OPENVINO_ASSERT(phi4_parser_prototype, "Invalid pointer to Phi4ReasoningParser prototype.");
    if (object.Get("constructor").StrictEquals(phi4_parser_prototype.Value())) {
        auto parser_wrapper = Napi::ObjectWrap<Phi4ReasoningParserWrapper>::Unwrap(object);
        return parser_wrapper->get_parser();
    }

    // Check Llama3PythonicToolParser
    const auto& llama3_pythonic_parser_prototype = addon_data->llama3_pythonic_tool_parser;
    OPENVINO_ASSERT(llama3_pythonic_parser_prototype, "Invalid pointer to Llama3PythonicToolParser prototype.");
    if (object.Get("constructor").StrictEquals(llama3_pythonic_parser_prototype.Value())) {
        auto parser_wrapper = Napi::ObjectWrap<Llama3PythonicToolParserWrapper>::Unwrap(object);
        return parser_wrapper->get_parser();
    }

    // Check Llama3JsonToolParser
    const auto& llama3_json_parser_prototype = addon_data->llama3_json_tool_parser;
    OPENVINO_ASSERT(llama3_json_parser_prototype, "Invalid pointer to Llama3JsonToolParser prototype.");
    if (object.Get("constructor").StrictEquals(llama3_json_parser_prototype.Value())) {
        auto parser_wrapper = Napi::ObjectWrap<Llama3JsonToolParserWrapper>::Unwrap(object);
        return parser_wrapper->get_parser();
    }

    return nullptr;
}

std::string json_stringify(const Napi::Env& env, const Napi::Value& value) {
    return env.Global()
        .Get("JSON")
        .ToObject()
        .Get("stringify")
        .As<Napi::Function>()
        .Call({ value })
        .ToString()
        .Utf8Value();
}

Napi::Value json_parse(const Napi::Env& env, const std::string& value) {
    return env.Global()
        .Get("JSON")
        .ToObject()
        .Get("parse")
        .As<Napi::Function>()
        .Call({ Napi::String::New(env, value) });
}

Napi::Function get_prototype_from_ov_addon(const Napi::Env& env, const std::string& ctor_name) {
    auto addon_data = env.GetInstanceData<AddonData>();
    OPENVINO_ASSERT(!addon_data->openvino_addon.IsEmpty(), "Addon data is not initialized");
    Napi::Value ov_addon = addon_data->openvino_addon.Value();
    OPENVINO_ASSERT(!ov_addon.IsUndefined() && !ov_addon.IsNull() && ov_addon.IsObject(), "OV addon value is not an object");
    Napi::Object addon_obj = ov_addon.As<Napi::Object>();
    OPENVINO_ASSERT(addon_obj.Has(ctor_name), std::string("OV addon does not export '") + ctor_name + "' class");
    Napi::Value ctor_val = addon_obj.Get(ctor_name);
    OPENVINO_ASSERT(ctor_val.IsFunction(), ctor_name + std::string(" is not a prototype"));

    return ctor_val.As<Napi::Function>();
}

Napi::Object to_decoded_result(const Napi::Env& env, const ov::genai::DecodedResults& results) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("texts", cpp_to_js<std::vector<std::string>, Napi::Value>(env, results.texts));
    obj.Set("scores", cpp_to_js<std::vector<float>, Napi::Value>(env, results.scores));
    obj.Set("perfMetrics", PerfMetricsWrapper::wrap(env, results.perf_metrics));
    obj.Set("parsed", cpp_to_js<std::vector<ov::genai::JsonContainer>, Napi::Value>(env, results.parsed));
    obj.Set("subword", Napi::String::New(env, results));
    return obj;
}

Napi::Object to_vlm_decoded_result(const Napi::Env& env, const ov::genai::VLMDecodedResults& results) {
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("texts", cpp_to_js<std::vector<std::string>, Napi::Value>(env, results.texts));
    obj.Set("scores", cpp_to_js<std::vector<float>, Napi::Value>(env, results.scores));
    obj.Set("perfMetrics", VLMPerfMetricsWrapper::wrap(env, results.perf_metrics));
    obj.Set("parsed", cpp_to_js<std::vector<ov::genai::JsonContainer>, Napi::Value>(env, results.parsed));
    return obj;
}
