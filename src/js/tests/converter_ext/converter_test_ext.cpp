// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <napi.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "include/any_converter.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"

namespace {

struct UnsupportedType {};

// Verify the stored type before exercising the converter.
template <typename T>
Napi::Value convert_checked(const Napi::Env& env, const ov::Any& any) {
    OPENVINO_ASSERT(any.is<T>(), "test input does not store the intended C++ type");
    return cpp_to_js<ov::Any, Napi::Value>(env, any);
}

Napi::Value ConvString(const Napi::CallbackInfo& info) {
    return convert_checked<std::string>(info.Env(), ov::Any(std::string("HAPPY")));
}

Napi::Value CstrStoredAsString(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), ov::Any("Speech").is<std::string>());
}

Napi::Value ConvBool(const Napi::CallbackInfo& info) {
    return convert_checked<bool>(info.Env(), ov::Any(true));
}

Napi::Value ConvInt(const Napi::CallbackInfo& info) {
    return convert_checked<int>(info.Env(), ov::Any(static_cast<int>(-42)));
}

Napi::Value ConvInt64Safe(const Napi::CallbackInfo& info) {
    return convert_checked<int64_t>(info.Env(), ov::Any(static_cast<int64_t>(42)));
}

Napi::Value ConvInt64Big(const Napi::CallbackInfo& info) {
    return convert_checked<int64_t>(info.Env(), ov::Any(static_cast<int64_t>(9007199254740993LL)));
}

Napi::Value ConvSizeT(const Napi::CallbackInfo& info) {
    return convert_checked<size_t>(info.Env(), ov::Any(static_cast<size_t>(7)));
}

Napi::Value ConvSizeTBig(const Napi::CallbackInfo& info) {
    return convert_checked<size_t>(info.Env(), ov::Any(static_cast<size_t>(9007199254740993ULL)));
}

Napi::Value ConvInt64MaxSafe(const Napi::CallbackInfo& info) {
    return convert_checked<int64_t>(info.Env(), ov::Any(static_cast<int64_t>(9007199254740991LL)));
}

Napi::Value ConvInt64JustAbove(const Napi::CallbackInfo& info) {
    return convert_checked<int64_t>(info.Env(), ov::Any(static_cast<int64_t>(9007199254740992LL)));
}

Napi::Value ConvInt64MinSafe(const Napi::CallbackInfo& info) {
    return convert_checked<int64_t>(info.Env(), ov::Any(static_cast<int64_t>(-9007199254740991LL)));
}

Napi::Value ConvInt64JustBelow(const Napi::CallbackInfo& info) {
    return convert_checked<int64_t>(info.Env(), ov::Any(static_cast<int64_t>(-9007199254740992LL)));
}

Napi::Value ConvSizeTMaxSafe(const Napi::CallbackInfo& info) {
    return convert_checked<size_t>(info.Env(), ov::Any(static_cast<size_t>(9007199254740991ULL)));
}

Napi::Value ConvSizeTJustAbove(const Napi::CallbackInfo& info) {
    return convert_checked<size_t>(info.Env(), ov::Any(static_cast<size_t>(9007199254740992ULL)));
}

Napi::Value ConvFloat(const Napi::CallbackInfo& info) {
    return convert_checked<float>(info.Env(), ov::Any(1.5f));
}

Napi::Value ConvDouble(const Napi::CallbackInfo& info) {
    return convert_checked<double>(info.Env(), ov::Any(2.5));
}

Napi::Value ConvNestedMap(const Napi::CallbackInfo& info) {
    ov::AnyMap inner{{"emotion", std::string("HAPPY")}};
    ov::AnyMap outer{{"nested", inner}, {"flag", true}};
    return cpp_to_js<ov::AnyMap, Napi::Value>(info.Env(), outer);
}

Napi::Value ConvVecString(const Napi::CallbackInfo& info) {
    return convert_checked<std::vector<std::string>>(info.Env(), ov::Any(std::vector<std::string>{"a", "b"}));
}

Napi::Value ConvVecInt64(const Napi::CallbackInfo& info) {
    return convert_checked<std::vector<int64_t>>(info.Env(), ov::Any(std::vector<int64_t>{1, 2, 3}));
}

Napi::Value ConvVecDouble(const Napi::CallbackInfo& info) {
    return convert_checked<std::vector<double>>(info.Env(), ov::Any(std::vector<double>{1.5, 2.5}));
}

Napi::Value ConvVecFloat(const Napi::CallbackInfo& info) {
    return convert_checked<std::vector<float>>(info.Env(), ov::Any(std::vector<float>{1.5f, 2.5f}));
}

Napi::Value ConvVecAny(const Napi::CallbackInfo& info) {
    std::vector<ov::Any> values;
    values.emplace_back(std::string("x"));
    values.emplace_back(static_cast<int64_t>(5));
    values.emplace_back(true);
    return convert_checked<std::vector<ov::Any>>(info.Env(), ov::Any(values));
}

Napi::Value ConvEmptyAny(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::Any, Napi::Value>(info.Env(), ov::Any());
}

Napi::Value ConvEmptyMap(const Napi::CallbackInfo& info) {
    return cpp_to_js<ov::AnyMap, Napi::Value>(info.Env(), ov::AnyMap{});
}

Napi::Value ConvFeaturesExample(const Napi::CallbackInfo& info) {
    std::vector<ov::AnyMap> features{
        ov::AnyMap{{"emotion", std::string("HAPPY")}, {"event", std::string("Speech")}}};
    return cpp_to_js<std::vector<ov::AnyMap>, Napi::Value>(info.Env(), features);
}

Napi::Value ConvUnsupported(const Napi::CallbackInfo& info) {
    try {
        return cpp_to_js<ov::Any, Napi::Value>(info.Env(), ov::Any(UnsupportedType{}));
    } catch (const std::exception& e) {
        Napi::Error::New(info.Env(), e.what()).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("convString", Napi::Function::New(env, ConvString));
    exports.Set("cstrStoredAsString", Napi::Function::New(env, CstrStoredAsString));
    exports.Set("convBool", Napi::Function::New(env, ConvBool));
    exports.Set("convInt", Napi::Function::New(env, ConvInt));
    exports.Set("convInt64Safe", Napi::Function::New(env, ConvInt64Safe));
    exports.Set("convInt64Big", Napi::Function::New(env, ConvInt64Big));
    exports.Set("convSizeT", Napi::Function::New(env, ConvSizeT));
    exports.Set("convSizeTBig", Napi::Function::New(env, ConvSizeTBig));
    exports.Set("convInt64MaxSafe", Napi::Function::New(env, ConvInt64MaxSafe));
    exports.Set("convInt64JustAbove", Napi::Function::New(env, ConvInt64JustAbove));
    exports.Set("convInt64MinSafe", Napi::Function::New(env, ConvInt64MinSafe));
    exports.Set("convInt64JustBelow", Napi::Function::New(env, ConvInt64JustBelow));
    exports.Set("convSizeTMaxSafe", Napi::Function::New(env, ConvSizeTMaxSafe));
    exports.Set("convSizeTJustAbove", Napi::Function::New(env, ConvSizeTJustAbove));
    exports.Set("convFloat", Napi::Function::New(env, ConvFloat));
    exports.Set("convDouble", Napi::Function::New(env, ConvDouble));
    exports.Set("convNestedMap", Napi::Function::New(env, ConvNestedMap));
    exports.Set("convVecString", Napi::Function::New(env, ConvVecString));
    exports.Set("convVecInt64", Napi::Function::New(env, ConvVecInt64));
    exports.Set("convVecDouble", Napi::Function::New(env, ConvVecDouble));
    exports.Set("convVecFloat", Napi::Function::New(env, ConvVecFloat));
    exports.Set("convVecAny", Napi::Function::New(env, ConvVecAny));
    exports.Set("convEmptyAny", Napi::Function::New(env, ConvEmptyAny));
    exports.Set("convEmptyMap", Napi::Function::New(env, ConvEmptyMap));
    exports.Set("convFeaturesExample", Napi::Function::New(env, ConvFeaturesExample));
    exports.Set("convUnsupported", Napi::Function::New(env, ConvUnsupported));
    return exports;
}

}  // namespace

NODE_API_MODULE(converter_test_ext, Init)
