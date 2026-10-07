// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/any_converter.hpp"

#include <string>

#include "include/napi_number.hpp"
#include "openvino/core/except.hpp"

template <>
Napi::Value cpp_to_js<ov::Any, Napi::Value>(const Napi::Env& env, const ov::Any& any) {
    if (any.empty()) {
        return env.Null();
    }
    if (any.is<std::string>()) {
        return Napi::String::New(env, any.as<std::string>());
    }
    if (any.is<bool>()) {
        return Napi::Boolean::New(env, any.as<bool>());
    }
    if (any.is<int>()) {
        return Napi::Number::New(env, any.as<int>());
    }
    if (any.is<int64_t>()) {
        return ov::js::number_or_bigint(env, any.as<int64_t>());
    }
    if (any.is<size_t>()) {
        return ov::js::number_or_bigint(env, any.as<size_t>());
    }
    if (any.is<float>()) {
        return ov::js::rounded_number(env, any.as<float>());
    }
    if (any.is<double>()) {
        return Napi::Number::New(env, any.as<double>());
    }
    if (any.is<ov::AnyMap>()) {
        return cpp_to_js<ov::AnyMap, Napi::Value>(env, any.as<ov::AnyMap>());
    }
    if (any.is<std::vector<std::string>>()) {
        const auto& vec = any.as<std::vector<std::string>>();
        auto js_array = Napi::Array::New(env, vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            js_array[i] = Napi::String::New(env, vec[i]);
        }
        return js_array;
    }
    if (any.is<std::vector<int64_t>>()) {
        const auto& vec = any.as<std::vector<int64_t>>();
        auto js_array = Napi::Array::New(env, vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            js_array[i] = ov::js::number_or_bigint(env, vec[i]);
        }
        return js_array;
    }
    if (any.is<std::vector<double>>()) {
        const auto& vec = any.as<std::vector<double>>();
        auto js_array = Napi::Array::New(env, vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            js_array[i] = Napi::Number::New(env, vec[i]);
        }
        return js_array;
    }
    if (any.is<std::vector<float>>()) {
        const auto& vec = any.as<std::vector<float>>();
        auto js_array = Napi::Array::New(env, vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            js_array[i] = Napi::Number::New(env, vec[i]);
        }
        return js_array;
    }
    if (any.is<std::vector<ov::Any>>()) {
        const auto& vec = any.as<std::vector<ov::Any>>();
        auto js_array = Napi::Array::New(env, vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            js_array[i] = cpp_to_js<ov::Any, Napi::Value>(env, vec[i]);
        }
        return js_array;
    }
    OPENVINO_THROW("Cannot convert ov::Any to a JS value: unsupported stored type '",
                   any.type_info().name(),
                   "'");
}

template <>
Napi::Value cpp_to_js<ov::AnyMap, Napi::Value>(const Napi::Env& env, const ov::AnyMap& any_map) {
    auto js_object = Napi::Object::New(env);
    for (const auto& [key, value] : any_map) {
        js_object.Set(key, cpp_to_js<ov::Any, Napi::Value>(env, value));
    }
    return js_object;
}

template <>
Napi::Value cpp_to_js<std::vector<ov::AnyMap>, Napi::Value>(const Napi::Env& env,
                                                            const std::vector<ov::AnyMap>& value) {
    auto js_array = Napi::Array::New(env, value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        js_array[i] = cpp_to_js<ov::AnyMap, Napi::Value>(env, value[i]);
    }
    return js_array;
}
