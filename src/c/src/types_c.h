// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdarg>
#include <functional>
#include <memory>
#include <string>

#include "openvino/c/ov_common.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/json_container.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"

inline ov_element_type_e cpp_element_type_to_c(const ov::element::Type& t) {
    if (t == ov::element::f32)
        return static_cast<ov_element_type_e>(ov::element::Type_t::f32);
    if (t == ov::element::f16)
        return static_cast<ov_element_type_e>(ov::element::Type_t::f16);
    if (t == ov::element::i32)
        return static_cast<ov_element_type_e>(ov::element::Type_t::i32);
    if (t == ov::element::i64)
        return static_cast<ov_element_type_e>(ov::element::Type_t::i64);
    if (t == ov::element::u8)
        return static_cast<ov_element_type_e>(ov::element::Type_t::u8);
    return static_cast<ov_element_type_e>(0);
}

#define GET_PROPERTY_FROM_ARGS_LIST                                                                            \
    std::string property_key = va_arg(args_ptr, char*);                                                        \
    if (property_key == ov::cache_encryption_callbacks.name()) {                                               \
        ov_encryption_callbacks* _value = va_arg(args_ptr, ov_encryption_callbacks*);                          \
        auto encrypt_func = _value->encrypt_func;                                                              \
        auto decrypt_func = _value->decrypt_func;                                                              \
        std::function<std::string(const std::string&)> encrypt_value = [encrypt_func](const std::string& in) { \
            size_t out_size = 0;                                                                               \
            std::string out_str;                                                                               \
            encrypt_func(in.c_str(), in.length(), nullptr, &out_size);                                         \
            if (out_size > 0) {                                                                                \
                std::unique_ptr<char[]> output_ptr = std::make_unique<char[]>(out_size);                       \
                if (output_ptr) {                                                                              \
                    char* output = output_ptr.get();                                                           \
                    encrypt_func(in.c_str(), in.length(), output, &out_size);                                  \
                    out_str.assign(output, out_size);                                                          \
                }                                                                                              \
            }                                                                                                  \
            return out_str;                                                                                    \
        };                                                                                                     \
        std::function<std::string(const std::string&)> decrypt_value = [decrypt_func](const std::string& in) { \
            size_t out_size = 0;                                                                               \
            std::string out_str;                                                                               \
            decrypt_func(in.c_str(), in.length(), nullptr, &out_size);                                         \
            if (out_size > 0) {                                                                                \
                std::unique_ptr<char[]> output_ptr = std::make_unique<char[]>(out_size);                       \
                if (output_ptr) {                                                                              \
                    char* output = output_ptr.get();                                                           \
                    decrypt_func(in.c_str(), in.length(), output, &out_size);                                  \
                    out_str.assign(output, out_size);                                                          \
                }                                                                                              \
            }                                                                                                  \
            return out_str;                                                                                    \
        };                                                                                                     \
        ov::EncryptionCallbacks encryption_callbacks{std::move(encrypt_value), std::move(decrypt_value)};      \
        property[property_key] = encryption_callbacks;                                                         \
    } else {                                                                                                   \
        std::string _value = va_arg(args_ptr, char*);                                                          \
        ov::Any value = _value;                                                                                \
        property[property_key] = value;                                                                        \
    }

/**
 * @struct ov_genai_generation_config_opaque
 * @brief This is an interface of ov::genai::GenerationConfig
 */
struct ov_genai_generation_config_opaque {
    std::shared_ptr<ov::genai::GenerationConfig> object;
};
typedef struct ov_genai_generation_config_opaque ov_genai_generation_config;

/**
 * @struct ov_genai_llm_pipeline_opaque
 * @brief This is an interface of ov::genai::LLMPipeline
 */
struct ov_genai_llm_pipeline_opaque {
    std::shared_ptr<ov::genai::LLMPipeline> object;
};
typedef struct ov_genai_llm_pipeline_opaque ov_genai_llm_pipeline;

/**
 * @struct ov_genai_perf_metrics_opaque
 * @brief This is an interface of ov::genai::PerfMetrics
 */
struct ov_genai_perf_metrics_opaque {
    std::shared_ptr<ov::genai::PerfMetrics> object;
};
typedef struct ov_genai_perf_metrics_opaque ov_genai_perf_metrics;

/**
 * @struct ov_genai_decoded_results_opaque
 * @brief This is an interface of ov::genai::DecodedResults
 */
struct ov_genai_decoded_results_opaque {
    std::shared_ptr<ov::genai::DecodedResults> object;
};
typedef struct ov_genai_decoded_results_opaque ov_genai_decoded_results;

/**
 * @struct ov_genai_whisper_decoded_result_chunk_opaque
 * @brief This is an interface of ov::genai::WhisperDecodedResultChunk
 */
struct ov_genai_whisper_decoded_result_chunk_opaque {
    std::shared_ptr<ov::genai::WhisperDecodedResultChunk> object;
};
typedef struct ov_genai_whisper_decoded_result_chunk_opaque ov_genai_whisper_decoded_result_chunk;

/**
 * @struct ov_genai_whisper_decoded_results_opaque
 * @brief This is an interface of ov::genai::WhisperDecodedResults
 */
struct ov_genai_whisper_decoded_results_opaque {
    std::shared_ptr<ov::genai::WhisperDecodedResults> object;
};
typedef struct ov_genai_whisper_decoded_results_opaque ov_genai_whisper_decoded_results;

/**
 * @struct ov_genai_whisper_generation_config_opaque
 * @brief This is an interface of ov::genai::WhisperGenerationConfig
 */
struct ov_genai_whisper_generation_config_opaque {
    std::shared_ptr<ov::genai::WhisperGenerationConfig> object;
};
typedef struct ov_genai_whisper_generation_config_opaque ov_genai_whisper_generation_config;

/**
 * @struct ov_genai_whisper_pipeline_opaque
 * @brief This is an interface of ov::genai::WhisperPipeline
 */
struct ov_genai_whisper_pipeline_opaque {
    std::shared_ptr<ov::genai::WhisperPipeline> object;
};
typedef struct ov_genai_whisper_pipeline_opaque ov_genai_whisper_pipeline;

/**
 * @struct ov_genai_vlm_decoded_results_opaque
 * @brief This is an interface of ov::genai::VLMDecodedResults
 */
struct ov_genai_vlm_decoded_results_opaque {
    std::shared_ptr<ov::genai::VLMDecodedResults> object;
};
typedef struct ov_genai_vlm_decoded_results_opaque ov_genai_vlm_decoded_results;

/**
 * @struct ov_genai_vlm_pipeline_opaque
 * @brief This is an interface of ov::genai::VLMPipeline
 */
struct ov_genai_vlm_pipeline_opaque {
    std::shared_ptr<ov::genai::VLMPipeline> object;
};
typedef struct ov_genai_vlm_pipeline_opaque ov_genai_vlm_pipeline;

/**
 * @struct ov_genai_chat_history_opaque
 * @brief This is an interface of ov::genai::ChatHistory
 */
struct ov_genai_chat_history_opaque {
    std::shared_ptr<ov::genai::ChatHistory> object;
};
typedef struct ov_genai_chat_history_opaque ov_genai_chat_history;

/**
 * @struct ov_genai_json_container_opaque
 * @brief This is an interface of ov::genai::JsonContainer
 */
struct ov_genai_json_container_opaque {
    std::shared_ptr<ov::genai::JsonContainer> object;
};
typedef struct ov_genai_json_container_opaque ov_genai_json_container;

/**
 * @struct ov_genai_speech_generation_config_opaque
 * @brief This is an interface of ov::genai::SpeechGenerationConfig
 */
struct ov_genai_speech_generation_config_opaque {
    std::shared_ptr<ov::genai::SpeechGenerationConfig> object;
};
typedef struct ov_genai_speech_generation_config_opaque ov_genai_speech_generation_config;

/**
 * @struct ov_genai_text2speech_decoded_results_opaque
 * @brief This is an interface of ov::genai::Text2SpeechDecodedResults
 */
struct ov_genai_text2speech_decoded_results_opaque {
    std::shared_ptr<ov::genai::Text2SpeechDecodedResults> object;
};
typedef struct ov_genai_text2speech_decoded_results_opaque ov_genai_text2speech_decoded_results;

/**
 * @struct ov_genai_text2speech_pipeline_opaque
 * @brief This is an interface of ov::genai::Text2SpeechPipeline
 */
struct ov_genai_text2speech_pipeline_opaque {
    std::shared_ptr<ov::genai::Text2SpeechPipeline> object;
};
typedef struct ov_genai_text2speech_pipeline_opaque ov_genai_text2speech_pipeline;
