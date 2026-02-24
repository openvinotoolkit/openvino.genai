// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <napi.h>

#include <set>

#include "openvino/core/type/element_type.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/genai/rag/text_rerank_pipeline.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/openvino.hpp"

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using GenerateInputs = std::variant<ov::genai::StringInputs, ov::genai::ChatHistory>;

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
template <>
int64_t js_to_cpp<int64_t>(const Napi::Env& env, const Napi::Value& value);
template <>
size_t js_to_cpp<size_t>(const Napi::Env& env, const Napi::Value& value);
template <>
double js_to_cpp<double>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType std::vector<std::string> */
template <>
std::vector<std::string> js_to_cpp<std::vector<std::string>>(const Napi::Env& env, const Napi::Value& value);
template <>
std::vector<int64_t> js_to_cpp<std::vector<int64_t>>(const Napi::Env& env, const Napi::Value& value);
template <>
std::vector<size_t> js_to_cpp<std::vector<size_t>>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StopCriteria (accepts number 0, 1, 2) */
template <>
ov::genai::StopCriteria js_to_cpp<ov::genai::StopCriteria>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType std::vector<double> (e.g. raw speech) */
template <>
std::vector<double> js_to_cpp<std::vector<double>>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType GenerateInputs */
template <>
GenerateInputs js_to_cpp<GenerateInputs>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::JsonContainer */
template <>
ov::genai::JsonContainer js_to_cpp<ov::genai::JsonContainer>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::SchedulerConfig */
template <>
ov::genai::SchedulerConfig js_to_cpp<ov::genai::SchedulerConfig>(const Napi::Env& env, const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StructuredOutputConfig */
template <>
ov::genai::StructuredOutputConfig js_to_cpp<ov::genai::StructuredOutputConfig>(const Napi::Env& env,
                                                                               const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StructuredOutputConfig::Tag */
template <>
ov::genai::StructuredOutputConfig::Tag js_to_cpp<ov::genai::StructuredOutputConfig::Tag>(const Napi::Env& env,
                                                                                         const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::StructuredOutputConfig::StructuralTag */
template <>
ov::genai::StructuredOutputConfig::StructuralTag js_to_cpp<ov::genai::StructuredOutputConfig::StructuralTag>(
    const Napi::Env& env,
    const Napi::Value& value);
template <>
ov::Tensor js_to_cpp<ov::Tensor>(const Napi::Env& env, const Napi::Value& value);
template <>
std::shared_ptr<ov::genai::Parser> js_to_cpp<std::shared_ptr<ov::genai::Parser>>(const Napi::Env& env,
                                                                                 const Napi::Value& value);
template <>
std::vector<std::shared_ptr<ov::genai::Parser>> js_to_cpp<std::vector<std::shared_ptr<ov::genai::Parser>>>(
    const Napi::Env& env,
    const Napi::Value& value);
/** @brief  A template specialization for TargetType ov::genai::GenerationConfig */
template <>
ov::genai::GenerationConfig js_to_cpp<ov::genai::GenerationConfig>(const Napi::Env& env, const Napi::Value& value);
template <>
std::vector<ov::Tensor> js_to_cpp<std::vector<ov::Tensor>>(const Napi::Env& env, const Napi::Value& value);
/**
 * @brief  Unwraps a C++ object from a JavaScript wrapper.
 * @tparam TargetType The C++ class type to extract.
 * @return Reference to the unwrapped C++ object.
 */
template <typename TargetType>
TargetType& unwrap(const Napi::Env& env, const Napi::Value& value);

template <>
ov::genai::PerfMetrics& unwrap<ov::genai::PerfMetrics>(const Napi::Env& env, const Napi::Value& value);

template <>
ov::genai::VLMPerfMetrics& unwrap<ov::genai::VLMPerfMetrics>(const Napi::Env& env, const Napi::Value& value);

/**
 * @brief  Template function to convert C++ data types into Javascript data types
 * @tparam TargetType Destinated Javascript data type.
 * @tparam SourceType C++ data type.
 * @param info Contains the environment in which to construct a JavaScript object.
 * @return SourceType converted to a TargetType.
 */
template <typename SourceType, typename TargetType>
TargetType cpp_to_js(const Napi::Env& env, const SourceType& value);

/** @brief  A template specialization for TargetType Napi::Value and SourceType int64_t */
template <>
Napi::Value cpp_to_js<int64_t, Napi::Value>(const Napi::Env& env, const int64_t& value);

/** @brief  A template specialization for TargetType Napi::Value and SourceType size_t */
template <>
Napi::Value cpp_to_js<size_t, Napi::Value>(const Napi::Env& env, const size_t& value);

/** @brief  A template specialization for TargetType Napi::Value and SourceType float */
template <>
Napi::Value cpp_to_js<float, Napi::Value>(const Napi::Env& env, const float& value);

/** @brief  A template specialization for TargetType Napi::Value and SourceType ov::genai::EmbeddingResult */
template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResult, Napi::Value>(const Napi::Env& env,
                                                               const ov::genai::EmbeddingResult& embedding_result);

/** @brief  A template specialization for TargetType Napi::Value and SourceType ov::genai::EmbeddingResults */
template <>
Napi::Value cpp_to_js<ov::genai::EmbeddingResults, Napi::Value>(const Napi::Env& env,
                                                                const ov::genai::EmbeddingResults& embedding_result);

/** @brief  A template specialization for TargetType Napi::Value and SourceType std::vector<std::string> */
template <>
Napi::Value cpp_to_js<std::vector<std::string>, Napi::Value>(const Napi::Env& env,
                                                             const std::vector<std::string>& value);
/** @brief  A template specialization for TargetType Napi::Value and SourceType std::set<std::string> (JS Set) */
template <>
Napi::Value cpp_to_js<std::set<std::string>, Napi::Value>(const Napi::Env& env, const std::set<std::string>& value);
/** @brief  A template specialization for TargetType Napi::Value and SourceType std::set<int64_t> (JS Set) */
template <>
Napi::Value cpp_to_js<std::set<int64_t>, Napi::Value>(const Napi::Env& env, const std::set<int64_t>& value);

/** @brief  A template specialization for TargetType Napi::Value and SourceType std::vector<float> */
template <>
Napi::Value cpp_to_js<std::vector<float>, Napi::Value>(const Napi::Env& env, const std::vector<float>& value);

template <>
Napi::Value cpp_to_js<std::vector<double>, Napi::Value>(const Napi::Env& env, const std::vector<double>& value);

template <>
Napi::Value cpp_to_js<std::vector<size_t>, Napi::Value>(const Napi::Env& env, const std::vector<size_t>& value);

template <>
Napi::Value cpp_to_js<ov::genai::JsonContainer, Napi::Value>(const Napi::Env& env,
                                                             const ov::genai::JsonContainer& json_container);

template <>
Napi::Value cpp_to_js<std::vector<ov::genai::JsonContainer>, Napi::Value>(
    const Napi::Env& env,
    const std::vector<ov::genai::JsonContainer>& value);

template <>
Napi::Value cpp_to_js<std::vector<std::pair<size_t, float>>, Napi::Value>(
    const Napi::Env& env,
    const std::vector<std::pair<size_t, float>>& rerank_results);

template <>
Napi::Value cpp_to_js<ov::Tensor, Napi::Value>(const Napi::Env& env, const ov::Tensor& tensor);

template <>
Napi::Value cpp_to_js<ov::genai::TokenizedInputs, Napi::Value>(const Napi::Env& env,
                                                               const ov::genai::TokenizedInputs& tokenized_inputs);

/** @brief  A template specialization for TargetType Napi::Value and SourceType
 * ov::genai::StructuredOutputConfig::StructuralTag */
template <>
Napi::Value cpp_to_js<ov::genai::StructuredOutputConfig::StructuralTag, Napi::Value>(
    const Napi::Env& env,
    const ov::genai::StructuredOutputConfig::StructuralTag& value);

/** @brief  A template specialization for TargetType Napi::Value and SourceType ov::genai::StructuredOutputConfig */
template <>
Napi::Value cpp_to_js<ov::genai::StructuredOutputConfig, Napi::Value>(const Napi::Env& env,
                                                                      const ov::genai::StructuredOutputConfig& config);

/** @brief  A template specialization for TargetType Napi::Value and SourceType
 * std::vector<std::shared_ptr<ov::genai::Parser>> */
template <>
Napi::Value cpp_to_js<std::vector<std::shared_ptr<ov::genai::Parser>>, Napi::Value>(
    const Napi::Env& env,
    const std::vector<std::shared_ptr<ov::genai::Parser>>& parsers);

/**
 * @brief  Template function to convert C++ map into Javascript Object. Map key must be std::string.
 * @tparam MapElementType C++ data type of map elements.
 */

/** @brief  A template specialization for TargetType Napi::Value and SourceType ov::genai::GenerationConfig */
template <>
Napi::Value cpp_to_js<ov::genai::GenerationConfig, Napi::Value>(const Napi::Env& env,
                                                                const ov::genai::GenerationConfig& config);

/** @brief  A template specialization for TargetType Napi::Value and SourceType ov::genai::StopCriteria */
template <>
Napi::Value cpp_to_js<ov::genai::StopCriteria, Napi::Value>(const Napi::Env& env, const ov::genai::StopCriteria& value);

template <typename MapElementType>
Napi::Object cpp_map_to_js_object(const Napi::Env& env, const std::map<std::string, MapElementType>& map) {
    Napi::Object obj = Napi::Object::New(env);
    for (const auto& [k, v] : map) {
        obj.Set(k, v);
    }
    return obj;
}

bool is_napi_value_int(const Napi::Env& env, const Napi::Value& num);

bool is_chat_history(const Napi::Env& env, const Napi::Value& value);

std::shared_ptr<ov::genai::Parser> get_native_parser(const Napi::Env& env, const Napi::Object& object);

std::string json_stringify(const Napi::Env& env, const Napi::Value& value);

Napi::Value json_parse(const Napi::Env& env, const std::string& value);

Napi::Function get_prototype_from_ov_addon(const Napi::Env& env, const std::string& ctor_name);

Napi::Object to_decoded_result(const Napi::Env& env, const ov::genai::DecodedResults& results);

Napi::Object to_vlm_decoded_result(const Napi::Env& env, const ov::genai::VLMDecodedResults& results);
