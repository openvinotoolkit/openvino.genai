// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text_embedding_pipeline.h"

#include <stdarg.h>

#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "types_c.h"

namespace {

bool parse_bool(const std::string& s, bool& out) {
    if (s == "true" || s == "True" || s == "TRUE" || s == "1") {
        out = true;
        return true;
    }
    if (s == "false" || s == "False" || s == "FALSE" || s == "0") {
        out = false;
        return true;
    }
    return false;
}

bool parse_pooling(const std::string& s, ov::genai::TextEmbeddingPipeline::PoolingType& out) {
    if (s == "CLS") {
        out = ov::genai::TextEmbeddingPipeline::PoolingType::CLS;
        return true;
    }
    if (s == "MEAN") {
        out = ov::genai::TextEmbeddingPipeline::PoolingType::MEAN;
        return true;
    }
    if (s == "LAST_TOKEN") {
        out = ov::genai::TextEmbeddingPipeline::PoolingType::LAST_TOKEN;
        return true;
    }
    return false;
}

// Splits the variadic property pack into embedding-specific Config keys (typed) and pass-through plugin properties
// (kept as strings).
ov_status_e split_embedding_properties(va_list args_ptr,
                                       size_t property_size,
                                       ov::AnyMap& config_properties,
                                       ov::AnyMap& plugin_properties) {
    for (size_t i = 0; i < property_size; ++i) {
        const char* key_cstr = va_arg(args_ptr, const char*);
        const char* value_cstr = va_arg(args_ptr, const char*);
        if (!key_cstr || !value_cstr) {
            return ov_status_e::INVALID_C_PARAM;
        }
        std::string key(key_cstr);
        std::string value(value_cstr);
        if (key == "max_length" || key == "batch_size") {
            config_properties[key] = static_cast<size_t>(std::stoull(value));
        } else if (key == "pad_to_max_length" || key == "normalize") {
            bool b = false;
            if (!parse_bool(value, b)) {
                return ov_status_e::INVALID_C_PARAM;
            }
            config_properties[key] = b;
        } else if (key == "padding_side" || key == "query_instruction" || key == "embed_instruction") {
            config_properties[key] = value;
        } else if (key == "pooling_type") {
            ov::genai::TextEmbeddingPipeline::PoolingType pt;
            if (!parse_pooling(value, pt)) {
                return ov_status_e::INVALID_C_PARAM;
            }
            config_properties[key] = pt;
        } else {
            plugin_properties[key] = value;
        }
    }
    return ov_status_e::OK;
}

std::vector<std::string> to_string_vector(const char** texts, size_t count) {
    std::vector<std::string> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        out.emplace_back(texts[i] ? texts[i] : "");
    }
    return out;
}

ov_status_e wrap_result(ov::genai::EmbeddingResult&& value, ov_genai_embedding_result** out) {
    std::unique_ptr<ov_genai_embedding_result> r = std::make_unique<ov_genai_embedding_result>();
    r->value = std::move(value);
    *out = r.release();
    return ov_status_e::OK;
}

ov_status_e wrap_results(ov::genai::EmbeddingResults&& value, ov_genai_embedding_results** out) {
    std::unique_ptr<ov_genai_embedding_results> r = std::make_unique<ov_genai_embedding_results>();
    r->value = std::move(value);
    *out = r.release();
    return ov_status_e::OK;
}

ov_genai_embedding_dtype_e dtype_of(const ov::genai::EmbeddingResult& v) {
    if (std::holds_alternative<std::vector<float>>(v)) {
        return OV_GENAI_EMBEDDING_DTYPE_F32;
    }
    if (std::holds_alternative<std::vector<int8_t>>(v)) {
        return OV_GENAI_EMBEDDING_DTYPE_I8;
    }
    return OV_GENAI_EMBEDDING_DTYPE_U8;
}

ov_genai_embedding_dtype_e dtype_of(const ov::genai::EmbeddingResults& v) {
    if (std::holds_alternative<std::vector<std::vector<float>>>(v)) {
        return OV_GENAI_EMBEDDING_DTYPE_F32;
    }
    if (std::holds_alternative<std::vector<std::vector<int8_t>>>(v)) {
        return OV_GENAI_EMBEDDING_DTYPE_I8;
    }
    return OV_GENAI_EMBEDDING_DTYPE_U8;
}

template <typename T>
ov_status_e get_data_single(const ov_genai_embedding_result* result, const T** data, size_t* size) {
    if (!result || !data || !size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    const auto* vec = std::get_if<std::vector<T>>(&result->value);
    if (!vec) {
        return ov_status_e::INVALID_C_PARAM;
    }
    *data = vec->data();
    *size = vec->size();
    return ov_status_e::OK;
}

template <typename T>
ov_status_e get_data_at(const ov_genai_embedding_results* results, size_t i, const T** data, size_t* size) {
    if (!results || !data || !size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    const auto* outer = std::get_if<std::vector<std::vector<T>>>(&results->value);
    if (!outer) {
        return ov_status_e::INVALID_C_PARAM;
    }
    if (i >= outer->size()) {
        return ov_status_e::OUT_OF_BOUNDS;
    }
    *data = (*outer)[i].data();
    *size = (*outer)[i].size();
    return ov_status_e::OK;
}

}  // namespace

// ---- single embedding ----

void ov_genai_embedding_result_free(ov_genai_embedding_result* result) {
    if (result) {
        delete result;
    }
}

ov_status_e ov_genai_embedding_result_get_dtype(const ov_genai_embedding_result* result,
                                                ov_genai_embedding_dtype_e* dtype) {
    if (!result || !dtype) {
        return ov_status_e::INVALID_C_PARAM;
    }
    *dtype = dtype_of(result->value);
    return ov_status_e::OK;
}

ov_status_e ov_genai_embedding_result_get_size(const ov_genai_embedding_result* result, size_t* size) {
    if (!result || !size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    *size = std::visit([](const auto& v) { return v.size(); }, result->value);
    return ov_status_e::OK;
}

ov_status_e ov_genai_embedding_result_get_data_f32(const ov_genai_embedding_result* result,
                                                   const float** data,
                                                   size_t* size) {
    return get_data_single<float>(result, data, size);
}

ov_status_e ov_genai_embedding_result_get_data_i8(const ov_genai_embedding_result* result,
                                                  const int8_t** data,
                                                  size_t* size) {
    return get_data_single<int8_t>(result, data, size);
}

ov_status_e ov_genai_embedding_result_get_data_u8(const ov_genai_embedding_result* result,
                                                  const uint8_t** data,
                                                  size_t* size) {
    return get_data_single<uint8_t>(result, data, size);
}

// ---- batch embeddings ----

void ov_genai_embedding_results_free(ov_genai_embedding_results* results) {
    if (results) {
        delete results;
    }
}

ov_status_e ov_genai_embedding_results_get_dtype(const ov_genai_embedding_results* results,
                                                 ov_genai_embedding_dtype_e* dtype) {
    if (!results || !dtype) {
        return ov_status_e::INVALID_C_PARAM;
    }
    *dtype = dtype_of(results->value);
    return ov_status_e::OK;
}

ov_status_e ov_genai_embedding_results_get_count(const ov_genai_embedding_results* results, size_t* count) {
    if (!results || !count) {
        return ov_status_e::INVALID_C_PARAM;
    }
    *count = std::visit([](const auto& v) { return v.size(); }, results->value);
    return ov_status_e::OK;
}

ov_status_e ov_genai_embedding_results_get_size_at(const ov_genai_embedding_results* results, size_t i, size_t* size) {
    if (!results || !size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    ov_status_e status = ov_status_e::OK;
    std::visit(
        [&](const auto& v) {
            if (i >= v.size()) {
                status = ov_status_e::OUT_OF_BOUNDS;
                return;
            }
            *size = v[i].size();
        },
        results->value);
    return status;
}

ov_status_e ov_genai_embedding_results_get_data_f32_at(const ov_genai_embedding_results* results,
                                                       size_t i,
                                                       const float** data,
                                                       size_t* size) {
    return get_data_at<float>(results, i, data, size);
}

ov_status_e ov_genai_embedding_results_get_data_i8_at(const ov_genai_embedding_results* results,
                                                      size_t i,
                                                      const int8_t** data,
                                                      size_t* size) {
    return get_data_at<int8_t>(results, i, data, size);
}

ov_status_e ov_genai_embedding_results_get_data_u8_at(const ov_genai_embedding_results* results,
                                                      size_t i,
                                                      const uint8_t** data,
                                                      size_t* size) {
    return get_data_at<uint8_t>(results, i, data, size);
}

// ---- pipeline ----

ov_status_e ov_genai_text_embedding_pipeline_create(const char* models_path,
                                                    const char* device,
                                                    const size_t property_args_size,
                                                    ov_genai_text_embedding_pipeline** pipe,
                                                    ...) {
    if (!models_path || !device || !pipe || property_args_size % 2 != 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        ov::AnyMap config_properties;
        ov::AnyMap plugin_properties;
        va_list args_ptr;
        va_start(args_ptr, pipe);
        ov_status_e split_status =
            split_embedding_properties(args_ptr, property_args_size / 2, config_properties, plugin_properties);
        va_end(args_ptr);
        if (split_status != ov_status_e::OK) {
            return split_status;
        }
        ov::genai::TextEmbeddingPipeline::Config config(config_properties);
        std::unique_ptr<ov_genai_text_embedding_pipeline> _pipe =
            std::make_unique<ov_genai_text_embedding_pipeline>();
        _pipe->object = std::make_shared<ov::genai::TextEmbeddingPipeline>(std::filesystem::path(models_path),
                                                                           std::string(device),
                                                                           config,
                                                                           plugin_properties);
        *pipe = _pipe.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_text_embedding_pipeline_free(ov_genai_text_embedding_pipeline* pipe) {
    if (pipe) {
        delete pipe;
    }
}

ov_status_e ov_genai_text_embedding_pipeline_embed_documents(ov_genai_text_embedding_pipeline* pipe,
                                                             const char** texts,
                                                             size_t texts_count,
                                                             ov_genai_embedding_results** results) {
    if (!pipe || !(pipe->object) || !results || (texts_count > 0 && !texts)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<std::string> docs = to_string_vector(texts, texts_count);
        auto value = pipe->object->embed_documents(docs);
        return wrap_results(std::move(value), results);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text_embedding_pipeline_start_embed_documents_async(ov_genai_text_embedding_pipeline* pipe,
                                                                         const char** texts,
                                                                         size_t texts_count) {
    if (!pipe || !(pipe->object) || (texts_count > 0 && !texts)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<std::string> docs = to_string_vector(texts, texts_count);
        pipe->object->start_embed_documents_async(docs);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text_embedding_pipeline_wait_embed_documents(ov_genai_text_embedding_pipeline* pipe,
                                                                  ov_genai_embedding_results** results) {
    if (!pipe || !(pipe->object) || !results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto value = pipe->object->wait_embed_documents();
        return wrap_results(std::move(value), results);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text_embedding_pipeline_embed_query(ov_genai_text_embedding_pipeline* pipe,
                                                         const char* text,
                                                         ov_genai_embedding_result** result) {
    if (!pipe || !(pipe->object) || !text || !result) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto value = pipe->object->embed_query(std::string(text));
        return wrap_result(std::move(value), result);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text_embedding_pipeline_start_embed_query_async(ov_genai_text_embedding_pipeline* pipe,
                                                                     const char* text) {
    if (!pipe || !(pipe->object) || !text) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipe->object->start_embed_query_async(std::string(text));
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text_embedding_pipeline_wait_embed_query(ov_genai_text_embedding_pipeline* pipe,
                                                              ov_genai_embedding_result** result) {
    if (!pipe || !(pipe->object) || !result) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto value = pipe->object->wait_embed_query();
        return wrap_result(std::move(value), result);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}
