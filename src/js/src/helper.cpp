// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/helper.hpp"

#include <cmath>
#include <typeindex>

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
constexpr const char* STOP_CRITERIA_KEY = "stop_criteria";

// Safe integer range for JS Number: -(2^53 - 1) .. (2^53 - 1).
constexpr int64_t NAPI_NUMBER_MIN_INTEGER = -(1LL << 53) + 1;
constexpr int64_t NAPI_NUMBER_MAX_INTEGER = (1LL << 53) - 1;

/** True if the JS Number has no fractional part. */
bool is_js_integer(const Napi::Env& env, const Napi::Number& value) {
    return env.Global()
        .Get("Number")
        .ToObject()
        .Get("isInteger")
        .As<Napi::Function>()
        .Call({value})
        .ToBoolean()
        .Value();
}

/** True if the value is a JS Set entity. */
bool is_js_set(const Napi::Value& value) {
    return value.IsObject() && value.ToString().Utf8Value() == "[object Set]";
}

/** Get the first element of a JS Set, or Undefined() if empty. */
Napi::Value get_first_set_value(const Napi::Env& env, const Napi::Value& value) {
    const auto obj = value.As<Napi::Object>();
    const auto values_fn = obj.Get("values").As<Napi::Function>();
    const auto iterator = values_fn.Call(obj, {}).As<Napi::Object>();
    const auto next_fn = iterator.Get("next").As<Napi::Function>();
    const auto item = next_fn.Call(iterator, {}).As<Napi::Object>();
    return item.Get("value");
}

/** Determine C++ scalar type for the first collection element. */
std::type_index get_cpp_type(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsUndefined()) {
        return typeid(void);
    }
    if (value.IsBoolean()) {
        return typeid(bool);
    }
    if (value.IsString()) {
        return typeid(std::string);
    }
    if (value.IsNumber()) {
        if (is_js_integer(env, value.As<Napi::Number>())) {
            return typeid(int64_t);
        }
        return typeid(double);
    }
    if (value.IsBigInt()) {
        bool lossless = false;
        value.As<Napi::BigInt>().Int64Value(&lossless);
        if (lossless) {
            return typeid(int64_t);
        }
        const uint64_t uval = value.As<Napi::BigInt>().Uint64Value(&lossless);
        if (!lossless || uval > static_cast<uint64_t>(SIZE_MAX)) {
            OPENVINO_THROW("BigInt value is too large to fit in int64_t or size_t.");
        }
        return typeid(size_t);
    }
    if (value.IsArray()) {
        return typeid(std::vector<ov::Any>);
    }
    return typeid(std::nullptr_t);
}

/** Collects elements of a JS Set into a vector of Napi::Value. Value must be a JS Set. */
std::vector<Napi::Value> js_set_to_values(const Napi::Env& env, const Napi::Value& value) {
    const auto object_value = value.As<Napi::Object>();
    const auto values_fn = object_value.Get("values").As<Napi::Function>();
    const auto iterator = values_fn.Call(object_value, {}).As<Napi::Object>();
    const auto next_fn = iterator.Get("next").As<Napi::Function>();
    const auto size = object_value.Get("size").As<Napi::Number>().Int32Value();
    std::vector<Napi::Value> result;
    result.reserve(static_cast<size_t>(size));
    for (int32_t i = 0; i < size; ++i) {
        const auto item = next_fn.Call(iterator, {}).As<Napi::Object>();
        result.push_back(item.Get("value"));
    }
    return result;
}

/**
 * Convert JS Set to ov::Any. Element type is determined and all elements are validated;
 * conversion is set<string>, set<float>, set<size_t>, or set<int64_t> as for arrays.
 */
ov::Any js_set_to_any(const Napi::Env& env, const Napi::Value& value) {
    const auto first_element_type = get_cpp_type(env, get_first_set_value(env, value));
    const std::vector<Napi::Value> values = js_set_to_values(env, value);
    if (first_element_type == typeid(void)) {
        return ov::Any();
    }
    if (first_element_type == typeid(std::string)) {
        std::set<std::string> result;
        for (const Napi::Value& v : values) {
            result.insert(js_to_cpp<std::string>(env, v));
        }
        return ov::Any(std::move(result));
    }
    if (first_element_type == typeid(double)) {
        std::set<double> result;
        for (const Napi::Value& v : values) {
            result.insert(js_to_cpp<double>(env, v));
        }
        return ov::Any(std::move(result));
    }
    if (first_element_type == typeid(int64_t)) {
        std::set<int64_t> result;
        for (const Napi::Value& v : values) {
            result.insert(js_to_cpp<int64_t>(env, v));
        }
        return ov::Any(std::move(result));
    }
    if (first_element_type == typeid(size_t)) {
        std::set<size_t> result;
        for (const Napi::Value& v : values) {
            result.insert(js_to_cpp<size_t>(env, v));
        }
        return ov::Any(std::move(result));
    }
    OPENVINO_THROW("Cannot convert Set to ov::Any. " + std::string(first_element_type.name()) +
                   " is not a supported element type.");
}

/**
 * Convert a JS array to ov::Any. Handles strings, numbers (int/float), BigInt, and nested arrays (recursive).
 */
ov::Any js_array_to_any(const Napi::Env& env, const Napi::Array& array) {
    const size_t arrayLength = array.Length();
    if (arrayLength == 0) {
        return ov::Any(std::vector<ov::Any>());
    }
    const auto first_element_type = get_cpp_type(env, array[0u]);
    if (first_element_type == typeid(std::string)) {
        return ov::Any(js_to_cpp<std::vector<std::string>>(env, array));
    }
    if (first_element_type == typeid(double)) {
        return ov::Any(js_to_cpp<std::vector<double>>(env, array));
    }
    if (first_element_type == typeid(int64_t)) {
        return ov::Any(js_to_cpp<std::vector<int64_t>>(env, array));
    }
    if (first_element_type == typeid(size_t)) {
        return ov::Any(js_to_cpp<std::vector<size_t>>(env, array));
    }
    if (first_element_type == typeid(std::vector<ov::Any>)) {
        std::vector<ov::Any> inner_anys;
        for (uint32_t i = 0; i < arrayLength; ++i) {
            inner_anys.push_back(js_array_to_any(env, array[i].As<Napi::Array>()));
        }
        return ov::Any(inner_anys);
    }
    OPENVINO_THROW("Cannot convert Array to ov::Any. " + std::string(first_element_type.name()) +
                   " is not a supported element type.");
}

}  // namespace

template <>
ov::Any js_to_cpp<ov::Any>(const Napi::Env& env, const Napi::Value& value) {
    const auto cpp_type = get_cpp_type(env, value);
    if (cpp_type == typeid(void)) {
        return ov::Any();
    }
    if (cpp_type == typeid(std::string)) {
        return ov::Any(js_to_cpp<std::string>(env, value));
    }
    if (cpp_type == typeid(double)) {
        return ov::Any(js_to_cpp<double>(env, value));
    }
    if (cpp_type == typeid(int64_t)) {
        return ov::Any(js_to_cpp<int64_t>(env, value));
    }
    if (cpp_type == typeid(size_t)) {
        return ov::Any(js_to_cpp<size_t>(env, value));
    }
    if (cpp_type == typeid(bool)) {
        return ov::Any(value.ToBoolean().Value());
    }
    if (cpp_type == typeid(std::vector<ov::Any>)) {
        return js_array_to_any(env, value.As<Napi::Array>());
    }
    if (value.IsTypedArray()) {
        const napi_typedarray_type type = value.As<Napi::TypedArray>().TypedArrayType();
        switch (type) {
        case napi_float32_array:
        case napi_float64_array:
            return ov::Any(js_to_cpp<std::vector<double>>(env, value));
        case napi_int8_array:
        case napi_uint8_array:
        case napi_uint8_clamped_array:
        case napi_int16_array:
        case napi_uint16_array:
        case napi_int32_array:
        case napi_uint32_array:
        case napi_bigint64_array:
            return ov::Any(js_to_cpp<std::vector<int64_t>>(env, value));
        case napi_biguint64_array:
            return ov::Any(js_to_cpp<std::vector<size_t>>(env, value));
        default:
            OPENVINO_THROW("Cannot convert TypedArray to ov::Any: unsupported type.");
        }
    }
    if (value.IsObject()) {
        if (is_js_set(value)) {
            return js_set_to_any(env, value);
        } else {
            return ov::Any(js_to_cpp<ov::AnyMap>(env, value));
        }
    }
    OPENVINO_THROW("Cannot convert " + value.ToString().Utf8Value() + " to ov::Any");
}

template <>
ov::AnyMap js_to_cpp<ov::AnyMap>(const Napi::Env& env, const Napi::Value& value) {
    std::map<std::string, ov::Any> result_map;
    if (value.IsUndefined() || value.IsNull()) {
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
            result_map[key_name] = ov::genai::TextEmbeddingPipeline::PoolingType(value_by_key.ToNumber().Int32Value());
        } else if (key_name == STRUCTURED_OUTPUT_CONFIG_KEY) {
            result_map[key_name] = js_to_cpp<ov::genai::StructuredOutputConfig>(env, value_by_key);
        } else if (key_name == PARSERS_KEY) {
            result_map[key_name] = js_to_cpp<std::vector<std::shared_ptr<ov::genai::Parser>>>(env, value_by_key);
        } else if (key_name == STOP_CRITERIA_KEY) {
            result_map[key_name] = ov::Any(js_to_cpp<ov::genai::StopCriteria>(env, value_by_key));
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
        const Napi::Number number_value = value.As<Napi::Number>();
        OPENVINO_ASSERT(is_js_integer(env, number_value), "Passed argument must be an integer.");
        const auto value = number_value.Int64Value();
        return value;
    }
    bool lossless;
    auto result = value.As<Napi::BigInt>().Int64Value(&lossless);
    OPENVINO_ASSERT(lossless, "BigInt value is too large to fit in int64_t without precision loss.");
    return result;
}

template <>
double js_to_cpp<double>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsNumber(), "Passed argument must be of type Number.");
    auto result = value.As<Napi::Number>().DoubleValue();
    return result;
}

template <>
size_t js_to_cpp<size_t>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsNumber() || value.IsBigInt(), "Passed argument must be of type Number or BigInt.");
    if (value.IsNumber()) {
        const Napi::Number number_value = value.As<Napi::Number>();
        OPENVINO_ASSERT(is_js_integer(env, number_value), "Passed argument must be an integer.");
        const auto int_value = number_value.Int64Value();
        OPENVINO_ASSERT(int_value >= 0, "Passed argument must be non-negative for size_t.");
        OPENVINO_ASSERT(int_value <= static_cast<int64_t>(SIZE_MAX), "Number value is too large for size_t.");
        return static_cast<size_t>(int_value);
    }
    bool lossless = false;
    const uint64_t uval = value.As<Napi::BigInt>().Uint64Value(&lossless);
    if (!lossless || uval > static_cast<uint64_t>(SIZE_MAX))
        OPENVINO_THROW("BigInt value is too large for size_t.");
    auto result = static_cast<size_t>(uval);
    return result;
}

template <>
ov::genai::StopCriteria js_to_cpp<ov::genai::StopCriteria>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsNumber(), "stop_criteria must be a number (0=EARLY, 1=HEURISTIC, 2=NEVER).");
    int num = value.As<Napi::Number>().Int32Value();
    switch (num) {
    case static_cast<int>(ov::genai::StopCriteria::EARLY):
        return ov::genai::StopCriteria::EARLY;
        break;
    case static_cast<int>(ov::genai::StopCriteria::HEURISTIC):
        return ov::genai::StopCriteria::HEURISTIC;
        break;
    case static_cast<int>(ov::genai::StopCriteria::NEVER):
        return ov::genai::StopCriteria::NEVER;
        break;
    default:
        OPENVINO_THROW("Invalid stop criteria: " + std::to_string(num) +
                       ". Expected 0 (EARLY), 1 (HEURISTIC), or 2 (NEVER).");
    }
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
    OPENVINO_ASSERT(value.IsArray() || value.IsTypedArray(), "Passed argument must be of type Array or TypedArray.");

    uint32_t len = 0;
    if (value.IsTypedArray()) {
        const napi_typedarray_type type = value.As<Napi::TypedArray>().TypedArrayType();
        if (type == napi_float32_array || type == napi_float64_array || type == napi_biguint64_array)
            OPENVINO_THROW("Cannot convert TypedArray type to std::vector<int64_t>.");
        len = static_cast<uint32_t>(value.As<Napi::TypedArray>().ElementLength());
    } else {
        len = value.As<Napi::Array>().Length();
    }

    std::vector<int64_t> vec(len);
    const Napi::Object obj = value.As<Napi::Object>();
    for (auto i = 0; i < len; ++i)
        vec[i] = js_to_cpp<int64_t>(env, obj[i]);
    return vec;
}

template <>
std::vector<double> js_to_cpp<std::vector<double>>(const Napi::Env& env, const Napi::Value& value) {
    if (value.IsTypedArray()) {
        const Napi::TypedArray ta = value.As<Napi::TypedArray>();
        const size_t length = ta.ElementLength();
        std::vector<double> vector(length);
        if (ta.TypedArrayType() == napi_float32_array) {
            auto typed = value.As<Napi::TypedArrayOf<float>>();
            for (size_t i = 0; i < length; ++i)
                vector[i] = static_cast<double>(typed[i]);
        } else if (ta.TypedArrayType() == napi_float64_array) {
            auto typed = value.As<Napi::TypedArrayOf<double>>();
            for (size_t i = 0; i < length; ++i)
                vector[i] = typed[i];
        } else {
            OPENVINO_THROW("TypedArray must be Float32Array or Float64Array for std::vector<double>.");
        }
        return vector;
    }
    if (value.IsArray()) {
        auto array = value.As<Napi::Array>();
        size_t arrayLength = array.Length();
        std::vector<double> vector;
        vector.reserve(arrayLength);
        for (uint32_t i = 0; i < arrayLength; ++i) {
            Napi::Value elem = array.Get(i);
            vector.push_back(elem.ToNumber().DoubleValue());
        }
        return vector;
    }
    OPENVINO_THROW("Passed argument must be of type Array or Float32Array or Float64Array (e.g. raw speech).");
}

template <>
std::vector<size_t> js_to_cpp<std::vector<size_t>>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsTypedArray(), "Passed argument must be of type TypedArray (BigUint64Array).");
    const napi_typedarray_type type = value.As<Napi::TypedArray>().TypedArrayType();
    if (type != napi_biguint64_array)
        OPENVINO_THROW("TypedArray must be BigUint64Array for std::vector<size_t>.");
    auto typed = value.As<Napi::BigUint64Array>();
    const size_t len = typed.ElementLength();
    std::vector<size_t> vec(len);
    for (size_t i = 0; i < len; ++i) {
        const uint64_t u = typed[i];
        if (u > static_cast<uint64_t>(SIZE_MAX))
            OPENVINO_THROW("BigUint64Array element is too large for size_t.");
        vec[i] = static_cast<size_t>(u);
    }
    return vec;
}

template <>
ov::genai::JsonContainer js_to_cpp<ov::genai::JsonContainer>(const Napi::Env& env, const Napi::Value& value) {
    OPENVINO_ASSERT(
        value.IsObject() || value.IsArray(),
        "JsonContainer must be a JS object or an array but got " + std::string(value.ToString().Utf8Value()));
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
ov::genai::StructuredOutputConfig::Tag js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(const Napi::Env& env,
                                                                                         const Napi::Value& value) {
    OPENVINO_ASSERT(value.IsObject(), "Tag must be a JS object");
    auto obj = value.As<Napi::Object>();

    return ov::genai::StructuredOutputConfig::Tag(
        js_to_cpp<std::string>(env, obj.Get("begin")),
        js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, obj.Get("content")),
        js_to_cpp<std::string>(env, obj.Get("end")));
}

template <>
ov::genai::StructuredOutputConfig::StructuralTag js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(
    const Napi::Env& env,
    const Napi::Value& value) {
    if (value.IsString()) {
        return js_to_cpp<std::string>(env, value);
    }

    OPENVINO_ASSERT(value.IsObject(), "StructuralTag must be a JS object or string");
    auto obj = value.As<Napi::Object>();

    std::string tag_type = obj.Get("structuralTagType").ToString().Utf8Value();

    if (tag_type == "Regex") {
        return ov::genai::StructuredOutputConfig::Regex(js_to_cpp<std::string>(env, obj.Get("value")));
    } else if (tag_type == "JSONSchema") {
        return ov::genai::StructuredOutputConfig::JSONSchema(js_to_cpp<std::string>(env, obj.Get("value")));
    } else if (tag_type == "EBNF") {
        return ov::genai::StructuredOutputConfig::EBNF(js_to_cpp<std::string>(env, obj.Get("value")));
    } else if (tag_type == "ConstString") {
        return ov::genai::StructuredOutputConfig::ConstString(js_to_cpp<std::string>(env, obj.Get("value")));
    } else if (tag_type == "AnyText") {
        return ov::genai::StructuredOutputConfig::AnyText();
    } else if (tag_type == "QwenXMLParametersFormat") {
        return ov::genai::StructuredOutputConfig::QwenXMLParametersFormat(
            js_to_cpp<std::string>(env, obj.Get("jsonSchema")));
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
        return std::make_shared<ov::genai::StructuredOutputConfig::Tag>(
            js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(env, obj));
    } else if (tag_type == "TriggeredTags") {
        std::vector<ov::genai::StructuredOutputConfig::Tag> tags;
        auto js_tags = obj.Get("tags");
        auto triggers = js_to_cpp<std::vector<std::string>>(env, obj.Get("triggers"));
        auto at_least_one = obj.Get("atLeastOne");
        auto stop_after_first = obj.Get("stopAfterFirst");
        OPENVINO_ASSERT(at_least_one.IsBoolean() && stop_after_first.IsBoolean(),
                        "TriggeredTags 'atLeastOne', and 'stopAfterFirst' must be booleans");
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
            stop_after_first.As<Napi::Boolean>().Value());
    } else if (tag_type == "TagsWithSeparator") {
        std::vector<ov::genai::StructuredOutputConfig::Tag> tags;
        auto separator = js_to_cpp<std::string>(env, obj.Get("separator"));
        auto at_least_one = obj.Get("atLeastOne");
        auto stop_after_first = obj.Get("stopAfterFirst");
        OPENVINO_ASSERT(at_least_one.IsBoolean() && stop_after_first.IsBoolean(),
                        "TagsWithSeparator 'atLeastOne', and 'stopAfterFirst' must be booleans");

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
            stop_after_first.As<Napi::Boolean>().Value());
    } else {
        OPENVINO_THROW("Unknown StructuralTag type: " + tag_type);
    }
}

template <>
ov::genai::StructuredOutputConfig js_to_cpp<ov::genai::StructuredOutputConfig>(const Napi::Env& env,
                                                                               const Napi::Value& value) {
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
        config.structural_tags_config =
            js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(env, obj.Get("structural_tags_config"));
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
    OPENVINO_ASSERT(get_external_tensor_val.IsFunction(),
                    "Tensor object does not have a '__getExternalTensor' function. This may indicate an incompatible "
                    "or outdated openvino-node version.");
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
ov::genai::GenerationConfig js_to_cpp<ov::genai::GenerationConfig>(const Napi::Env& env, const Napi::Value& value) {
    ov::genai::GenerationConfig config;
    if (value.IsUndefined() || value.IsNull()) {
        return config;
    }
    ov::AnyMap config_map = js_to_cpp<ov::AnyMap>(env, value);
    config.update_generation_config(config_map);
    return config;
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
Napi::Value cpp_to_js<int64_t, Napi::Value>(const Napi::Env& env, const int64_t& value) {
    if (value >= NAPI_NUMBER_MIN_INTEGER && value <= NAPI_NUMBER_MAX_INTEGER) {
        return Napi::Number::New(env, value);
    }
    return Napi::BigInt::New(env, value);
}

template <>
Napi::Value cpp_to_js<size_t, Napi::Value>(const Napi::Env& env, const size_t& value) {
    if (value <= NAPI_NUMBER_MAX_INTEGER) {
        return Napi::Number::New(env, value);
    }
    return Napi::BigInt::New(env, static_cast<uint64_t>(value));
}

template <>
Napi::Value cpp_to_js<float, Napi::Value>(const Napi::Env& env, const float& value) {
    return Napi::Number::New(env, std::round(static_cast<double>(value) * 1e7) / 1e7);
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
Napi::Value cpp_to_js<std::set<std::string>, Napi::Value>(const Napi::Env& env, const std::set<std::string>& value) {
    auto set_ctor = env.Global().Get("Set").As<Napi::Function>();
    auto js_set = set_ctor.New({});
    auto add_fn = js_set.Get("add").As<Napi::Function>();
    for (const auto& item : value) {
        add_fn.Call(js_set, {Napi::String::New(env, item)});
    }
    return js_set;
}

template <>
Napi::Value cpp_to_js<std::set<int64_t>, Napi::Value>(const Napi::Env& env, const std::set<int64_t>& value) {
    auto set_ctor = env.Global().Get("Set").As<Napi::Function>();
    auto js_set = set_ctor.New({});
    auto add_fn = js_set.Get("add").As<Napi::Function>();
    for (const auto& item : value) {
        add_fn.Call(js_set, {cpp_to_js<int64_t, Napi::Value>(env, item)});
    }
    return js_set;
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
Napi::Value cpp_to_js<ov::genai::StopCriteria, Napi::Value>(const Napi::Env& env,
                                                            const ov::genai::StopCriteria& value) {
    // Return numeric value to match JS enum StopCriteria: EARLY=0, HEURISTIC=1, NEVER=2
    switch (value) {
    case ov::genai::StopCriteria::EARLY:
        return Napi::Number::New(env, 0);
    case ov::genai::StopCriteria::HEURISTIC:
        return Napi::Number::New(env, 1);
    case ov::genai::StopCriteria::NEVER:
        return Napi::Number::New(env, 2);
    }
    OPENVINO_THROW("Unknown StopCriteria value");
}

template <>
Napi::Value cpp_to_js<ov::genai::JsonContainer, Napi::Value>(const Napi::Env& env,
                                                             const ov::genai::JsonContainer& json_container) {
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

        auto external = Napi::External<ov::Tensor>::New(env,
                                                        new ov::Tensor(tensor),
                                                        [](Napi::Env /*env*/, ov::Tensor* external_tensor) {
                                                            delete external_tensor;
                                                        });
        auto tensor_wrap = prototype.New({external});

        return tensor_wrap;
    } catch (const ov::Exception& e) {
        Napi::Error::New(env, std::string("Cannot create Tensor wrapper: ") + e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

template <>
Napi::Value cpp_to_js<ov::genai::TokenizedInputs, Napi::Value>(const Napi::Env& env,
                                                               const ov::genai::TokenizedInputs& tokenized_inputs) {
    auto js_object = Napi::Object::New(env);

    js_object.Set("input_ids", cpp_to_js<ov::Tensor, Napi::Value>(env, tokenized_inputs.input_ids));
    js_object.Set("attention_mask", cpp_to_js<ov::Tensor, Napi::Value>(env, tokenized_inputs.attention_mask));
    // token_type_ids is optional and present only for paired inputs
    if (tokenized_inputs.token_type_ids.has_value()) {
        js_object.Set("token_type_ids",
                      cpp_to_js<ov::Tensor, Napi::Value>(env, tokenized_inputs.token_type_ids.value()));
    }

    return js_object;
}

template <>
Napi::Value cpp_to_js<ov::genai::StructuredOutputConfig, Napi::Value>(const Napi::Env& env,
                                                                      const ov::genai::StructuredOutputConfig& config) {
    Napi::Object obj = Napi::Object::New(env);
    if (config.json_schema.has_value()) {
        obj.Set("json_schema", Napi::String::New(env, *config.json_schema));
    }
    if (config.regex.has_value()) {
        obj.Set("regex", Napi::String::New(env, *config.regex));
    }
    if (config.grammar.has_value()) {
        obj.Set("grammar", Napi::String::New(env, *config.grammar));
    }
    if (config.backend.has_value()) {
        obj.Set("backend", Napi::String::New(env, *config.backend));
    }
    if (config.structural_tags_config.has_value()) {
        std::visit(
            [&env, &obj](const auto& v) {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, ov::genai::StructuredOutputConfig::StructuralTag>) {
                    obj.Set("structural_tags_config",
                            cpp_to_js<ov::genai::StructuredOutputConfig::StructuralTag, Napi::Value>(env, v));
                } else {
                    Napi::Error::New(
                        env,
                        "structural_tags_config with structural_tags array is not supported in JavaScript API")
                        .ThrowAsJavaScriptException();
                }
            },
            *config.structural_tags_config);
    }
    if (config.compound_grammar.has_value()) {
        Napi::Error::New(env, "compound_grammar is not supported in JavaScript API").ThrowAsJavaScriptException();
    }
    return obj;
}

template <>
Napi::Value cpp_to_js<ov::genai::StructuredOutputConfig::Tag, Napi::Value>(
    const Napi::Env& env,
    const ov::genai::StructuredOutputConfig::Tag& tag) {
    using SOC = ov::genai::StructuredOutputConfig;
    Napi::Object obj = Napi::Object::New(env);
    obj.Set("structuralTagType", Napi::String::New(env, "Tag"));
    obj.Set("begin", Napi::String::New(env, tag.begin));
    obj.Set("content", cpp_to_js<SOC::StructuralTag, Napi::Value>(env, tag.content));
    obj.Set("end", Napi::String::New(env, tag.end));
    return obj;
}

template <>
Napi::Value cpp_to_js<ov::genai::StructuredOutputConfig::StructuralTag, Napi::Value>(
    const Napi::Env& env,
    const ov::genai::StructuredOutputConfig::StructuralTag& tag) {
    using SOC = ov::genai::StructuredOutputConfig;
    return std::visit(
        overloaded{[&env](const std::string& s) -> Napi::Value {
                       return Napi::String::New(env, s);
                   },
                   [&env](const SOC::Regex& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "Regex"));
                       obj.Set("value", Napi::String::New(env, g.value));
                       return obj;
                   },
                   [&env](const SOC::JSONSchema& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "JSONSchema"));
                       obj.Set("value", Napi::String::New(env, g.value));
                       return obj;
                   },
                   [&env](const SOC::EBNF& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "EBNF"));
                       obj.Set("value", Napi::String::New(env, g.value));
                       return obj;
                   },
                   [&env](const SOC::ConstString& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "ConstString"));
                       obj.Set("value", Napi::String::New(env, g.value));
                       return obj;
                   },
                   [&env](const SOC::AnyText&) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "AnyText"));
                       return obj;
                   },
                   [&env](const SOC::QwenXMLParametersFormat& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "QwenXMLParametersFormat"));
                       obj.Set("jsonSchema", Napi::String::New(env, g.json_schema));
                       return obj;
                   },
                   [&env](const std::shared_ptr<SOC::Concat>& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "Concat"));
                       Napi::Array arr = Napi::Array::New(env, g->elements.size());
                       for (size_t i = 0; i < g->elements.size(); ++i) {
                           arr[i] = cpp_to_js<SOC::StructuralTag, Napi::Value>(env, g->elements[i]);
                       }
                       obj.Set("elements", arr);
                       return obj;
                   },
                   [&env](const std::shared_ptr<SOC::Union>& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "Union"));
                       Napi::Array arr = Napi::Array::New(env, g->elements.size());
                       for (size_t i = 0; i < g->elements.size(); ++i) {
                           arr[i] = cpp_to_js<SOC::StructuralTag, Napi::Value>(env, g->elements[i]);
                       }
                       obj.Set("elements", arr);
                       return obj;
                   },
                   [&env](const std::shared_ptr<SOC::Tag>& g) -> Napi::Value {
                       return cpp_to_js<SOC::Tag, Napi::Value>(env, *g);
                   },
                   [&env](const std::shared_ptr<SOC::TriggeredTags>& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "TriggeredTags"));
                       obj.Set("triggers", cpp_to_js<std::vector<std::string>, Napi::Value>(env, g->triggers));
                       Napi::Array tags_arr = Napi::Array::New(env, g->tags.size());
                       for (size_t i = 0; i < g->tags.size(); ++i) {
                           tags_arr[i] =
                               cpp_to_js<ov::genai::StructuredOutputConfig::Tag, Napi::Value>(env, g->tags[i]);
                       }
                       obj.Set("tags", tags_arr);
                       obj.Set("atLeastOne", Napi::Boolean::New(env, g->at_least_one));
                       obj.Set("stopAfterFirst", Napi::Boolean::New(env, g->stop_after_first));
                       return obj;
                   },
                   [&env](const std::shared_ptr<SOC::TagsWithSeparator>& g) -> Napi::Value {
                       Napi::Object obj = Napi::Object::New(env);
                       obj.Set("structuralTagType", Napi::String::New(env, "TagsWithSeparator"));
                       Napi::Array tags_arr = Napi::Array::New(env, g->tags.size());
                       for (size_t i = 0; i < g->tags.size(); ++i) {
                           tags_arr[i] =
                               cpp_to_js<ov::genai::StructuredOutputConfig::Tag, Napi::Value>(env, g->tags[i]);
                       }
                       obj.Set("tags", tags_arr);
                       obj.Set("separator", Napi::String::New(env, g->separator));
                       obj.Set("atLeastOne", Napi::Boolean::New(env, g->at_least_one));
                       obj.Set("stopAfterFirst", Napi::Boolean::New(env, g->stop_after_first));
                       return obj;
                   },
                   [&env](const auto&) -> Napi::Value {
                       OPENVINO_THROW("Unknown StructuralTag type");
                   }},
        tag);
}

template <>
Napi::Value cpp_to_js<std::vector<std::shared_ptr<ov::genai::Parser>>, Napi::Value>(
    const Napi::Env& env,
    const std::vector<std::shared_ptr<ov::genai::Parser>>& parsers) {
    Napi::Array arr = Napi::Array::New(env, parsers.size());
    for (size_t i = 0; i < parsers.size(); ++i) {
        const auto& p = parsers[i];
        if (!p) {
            Napi::Error::New(env, "null parser in parsers array").ThrowAsJavaScriptException();
            return env.Undefined();
        }
        auto* jsp = dynamic_cast<JSParser*>(p.get());
        if (jsp) {
            arr[i] = jsp->get_js_object(env);
            continue;
        }
        if (auto dp = std::dynamic_pointer_cast<ov::genai::DeepSeekR1ReasoningParser>(p)) {
            arr[i] = DeepSeekR1ReasoningParserWrapper::wrap(env, dp);
        } else if (auto pp = std::dynamic_pointer_cast<ov::genai::Phi4ReasoningParser>(p)) {
            arr[i] = Phi4ReasoningParserWrapper::wrap(env, pp);
        } else if (auto rp = std::dynamic_pointer_cast<ov::genai::ReasoningParser>(p)) {
            arr[i] = ReasoningParserWrapper::wrap(env, rp);
        } else if (auto lp = std::dynamic_pointer_cast<ov::genai::Llama3PythonicToolParser>(p)) {
            arr[i] = Llama3PythonicToolParserWrapper::wrap(env, lp);
        } else if (auto lj = std::dynamic_pointer_cast<ov::genai::Llama3JsonToolParser>(p)) {
            arr[i] = Llama3JsonToolParserWrapper::wrap(env, lj);
        } else {
            Napi::Error::New(env, "unsupported parser type in parsers array").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }
    return arr;
}

template <>
Napi::Value cpp_to_js<ov::genai::GenerationConfig, Napi::Value>(const Napi::Env& env,
                                                                const ov::genai::GenerationConfig& config) {
    Napi::Object obj = Napi::Object::New(env);

    // Generic
    obj.Set("max_new_tokens", cpp_to_js<size_t, Napi::Value>(env, config.max_new_tokens));
    obj.Set("max_length", cpp_to_js<size_t, Napi::Value>(env, config.max_length));
    obj.Set("ignore_eos", Napi::Boolean::New(env, config.ignore_eos));
    obj.Set("min_new_tokens", cpp_to_js<size_t, Napi::Value>(env, config.min_new_tokens));
    obj.Set("echo", Napi::Boolean::New(env, config.echo));
    obj.Set("logprobs", cpp_to_js<size_t, Napi::Value>(env, config.logprobs));

    // EOS special token
    obj.Set("eos_token_id", cpp_to_js<int64_t, Napi::Value>(env, config.eos_token_id));
    if (!config.stop_strings.empty()) {
        obj.Set("stop_strings", cpp_to_js<std::set<std::string>, Napi::Value>(env, config.stop_strings));
    } else {
        obj.Set("stop_strings", env.Undefined());
    }
    obj.Set("include_stop_str_in_output", Napi::Boolean::New(env, config.include_stop_str_in_output));
    if (!config.stop_token_ids.empty()) {
        obj.Set("stop_token_ids", cpp_to_js<std::set<int64_t>, Napi::Value>(env, config.stop_token_ids));
    } else {
        obj.Set("stop_token_ids", env.Undefined());
    }

    // penalties (not used in beam search)
    obj.Set("repetition_penalty", cpp_to_js<float, Napi::Value>(env, config.repetition_penalty));
    obj.Set("presence_penalty", cpp_to_js<float, Napi::Value>(env, config.presence_penalty));
    obj.Set("frequency_penalty", cpp_to_js<float, Napi::Value>(env, config.frequency_penalty));

    // Beam search specific
    obj.Set("num_beam_groups", cpp_to_js<size_t, Napi::Value>(env, config.num_beam_groups));
    obj.Set("num_beams", cpp_to_js<size_t, Napi::Value>(env, config.num_beams));
    obj.Set("diversity_penalty", cpp_to_js<float, Napi::Value>(env, config.diversity_penalty));
    obj.Set("length_penalty", cpp_to_js<float, Napi::Value>(env, config.length_penalty));
    obj.Set("num_return_sequences", cpp_to_js<size_t, Napi::Value>(env, config.num_return_sequences));
    obj.Set("no_repeat_ngram_size", cpp_to_js<size_t, Napi::Value>(env, config.no_repeat_ngram_size));
    obj.Set(STOP_CRITERIA_KEY, cpp_to_js<ov::genai::StopCriteria, Napi::Value>(env, config.stop_criteria));

    // Multinomial
    obj.Set("temperature", cpp_to_js<float, Napi::Value>(env, config.temperature));
    obj.Set("top_p", cpp_to_js<float, Napi::Value>(env, config.top_p));
    obj.Set("top_k", cpp_to_js<size_t, Napi::Value>(env, config.top_k));
    obj.Set("do_sample", Napi::Boolean::New(env, config.do_sample));
    obj.Set("rng_seed", cpp_to_js<size_t, Napi::Value>(env, config.rng_seed));

    // CDPruner config
    obj.Set("pruning_ratio", cpp_to_js<size_t, Napi::Value>(env, config.pruning_ratio));
    obj.Set("relevance_weight", cpp_to_js<float, Napi::Value>(env, config.relevance_weight));

    // Assisting generation parameters
    obj.Set("assistant_confidence_threshold",
            cpp_to_js<float, Napi::Value>(env, config.assistant_confidence_threshold));
    obj.Set("num_assistant_tokens", cpp_to_js<size_t, Napi::Value>(env, config.num_assistant_tokens));
    obj.Set("max_ngram_size", cpp_to_js<size_t, Napi::Value>(env, config.max_ngram_size));

    // Structured output parameters
    if (config.structured_output_config.has_value()) {
        obj.Set(STRUCTURED_OUTPUT_CONFIG_KEY,
                cpp_to_js<ov::genai::StructuredOutputConfig, Napi::Value>(env, *config.structured_output_config));
    } else {
        obj.Set(STRUCTURED_OUTPUT_CONFIG_KEY, env.Undefined());
    }
    if (!config.parsers.empty()) {
        obj.Set(PARSERS_KEY,
                cpp_to_js<std::vector<std::shared_ptr<ov::genai::Parser>>, Napi::Value>(env, config.parsers));
    } else {
        obj.Set(PARSERS_KEY, env.Undefined());
    }

    // set to true if chat template should be applied for non-chat scenarios, set to false otherwise
    obj.Set("apply_chat_template", Napi::Boolean::New(env, config.apply_chat_template));

    return obj;
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
        .Call({value})
        .ToString()
        .Utf8Value();
}

Napi::Value json_parse(const Napi::Env& env, const std::string& value) {
    return env.Global().Get("JSON").ToObject().Get("parse").As<Napi::Function>().Call({Napi::String::New(env, value)});
}

Napi::Function get_prototype_from_ov_addon(const Napi::Env& env, const std::string& ctor_name) {
    auto addon_data = env.GetInstanceData<AddonData>();
    OPENVINO_ASSERT(!addon_data->openvino_addon.IsEmpty(), "Addon data is not initialized");
    Napi::Value ov_addon = addon_data->openvino_addon.Value();
    OPENVINO_ASSERT(!ov_addon.IsUndefined() && !ov_addon.IsNull() && ov_addon.IsObject(),
                    "OV addon value is not an object");
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
