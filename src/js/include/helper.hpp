#pragma once
#include <napi.h>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/openvino.hpp"

template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

#define VALIDATE_ARGS_COUNT(info, expected_count, method_name)                                 \
    if (info.Length() != expected_count) {                                                     \
        Napi::TypeError::New(info.Env(), method_name " expects " #expected_count " arguments") \
            .ThrowAsJavaScriptException();                                                     \
        return info.Env().Undefined();                                                         \
    }

/**
 * @brief  Template function to convert Javascript data types into C++ data types
 * @tparam TargetType destinated C++ data type
 * @param info Napi::CallbackInfo contains all arguments passed to a function or method
 * @param idx specifies index of a argument inside info.
 * @return specified argument converted to a TargetType.
 */
template <typename TargetType>
TargetType js_to_cpp(const Napi::Env& env, const Napi::Value& value);

/** @brief  A template specialization for TargetType ov::Any */
template <>
ov::Any js_to_cpp<ov::Any>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::AnyMap */
template <>
ov::AnyMap js_to_cpp<ov::AnyMap>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType std::string */
template <>
std::string js_to_cpp<std::string>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType std::vector<std::string> */
template <>
std::vector<std::string> js_to_cpp<std::vector<std::string>>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StringInputs */
template <>
ov::genai::StringInputs js_to_cpp<ov::genai::StringInputs>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::ChatHistory */
template <>
ov::genai::ChatHistory js_to_cpp<ov::genai::ChatHistory>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::SchedulerConfig */
template <>
ov::genai::SchedulerConfig js_to_cpp<ov::genai::SchedulerConfig>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StructuredOutputConfig */
template <>
ov::genai::StructuredOutputConfig js_to_cpp<ov::genai::StructuredOutputConfig>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StructuredOutputConfig::Tag */
template <>
ov::genai::StructuredOutputConfig::Tag js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StructuredOutputConfig::StructuralTag */
template <>
ov::genai::StructuredOutputConfig::StructuralTag js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(const Napi::Env& env, const Napi::Value& value);
/**
 * @brief  Unwraps a C++ object from a JavaScript wrapper.
 * @tparam TargetType The C++ class type to extract.
 * @return Reference to the unwrapped C++ object.
 */
template <typename TargetType>
TargetType& unwrap(const Napi::Env& env, const Napi::Value& value);

template <>
ov::genai::PerfMetrics& unwrap<ov::genai::PerfMetrics>(const Napi::Env& env, const Napi::Value& value);

/**
 * @brief  Template function to convert C++ data types into Javascript data types
 * @tparam TargetType Destinated Javascript data type.
 * @tparam SourceType C++ data type.
 * @param info Contains the environment in which to construct a JavaScript object.
 * @return SourceType converted to a TargetType.
 */
template <typename SourceType, typename TargetType>
TargetType cpp_to_js(const Napi::Env& env, SourceType);

/** @brief  A template specialization for TargetType Napi::Value and SourceType ov::genai::EmbeddingResult */
template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(
    const Napi::Env& env,
    const ov::genai::EmbeddingResult embedding_result
);

/** @brief  A template specialization for TargetType Napi::Value and SourceType ov::genai::EmbeddingResults */
template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(
    const Napi::Env& env,
    const ov::genai::EmbeddingResults embedding_result
);

/** @brief  A template specialization for TargetType Napi::Value and SourceType std::vector<std::string> */
template <>
Napi::Value cpp_to_js<std::vector<std::string>, Napi::Value>(const Napi::Env& env,
                                                             const std::vector<std::string> value);

/** @brief  A template specialization for TargetType Napi::Value and SourceType std::vector<float> */
template <>
Napi::Value cpp_to_js<std::vector<float>, Napi::Value>(const Napi::Env& env, const std::vector<float> value);

template <>
Napi::Value cpp_to_js<std::vector<double>, Napi::Value>(const Napi::Env& env, const std::vector<double> value);

template <>
Napi::Value cpp_to_js<std::vector<size_t>, Napi::Value>(const Napi::Env& env, const std::vector<size_t> value);

/**
 * @brief  Template function to convert C++ map into Javascript Object. Map key must be std::string.
 * @tparam MapElementType C++ data type of map elements.
 */
template <typename MapElementType>
Napi::Object cpp_map_to_js_object(const Napi::Env& env, const std::map<std::string, MapElementType>& map) {
    Napi::Object obj = Napi::Object::New(env);
    for (const auto& [k, v] : map) {
        obj.Set(k, v);
    }
    return obj;
}

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num);

std::string json_stringify(const Napi::Env& env, const Napi::Value& value);
