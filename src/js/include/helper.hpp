#pragma once
#include <napi.h>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/openvino.hpp"

template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

ov::AnyMap to_anyMap(const Napi::Env&, const Napi::Value&);

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
/** @brief  A template specialization for TargetType std::vector<std::string> */
template <>
std::vector<std::string> js_to_cpp<std::vector<std::string>>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StringInputs */
template <>
ov::genai::StringInputs js_to_cpp<ov::genai::StringInputs>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::ChatHistory */
template <>
ov::genai::ChatHistory js_to_cpp<ov::genai::ChatHistory>(const Napi::Env& env, const Napi::Value& value);

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

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num);
