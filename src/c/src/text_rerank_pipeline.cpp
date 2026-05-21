// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text_rerank_pipeline.h"

#include <stdarg.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/genai/rag/text_rerank_pipeline.hpp"
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

// Splits the variadic property pack into rerank-specific Config keys (typed) and pass-through plugin properties
// (kept as strings).
ov_status_e split_rerank_properties(va_list args_ptr,
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
        if (key == "top_n") {
            config_properties[key] = static_cast<size_t>(std::stoull(value));
        } else if (key == "max_length") {
            config_properties[key] = static_cast<size_t>(std::stoull(value));
        } else if (key == "pad_to_max_length") {
            bool b = false;
            if (!parse_bool(value, b)) {
                return ov_status_e::INVALID_C_PARAM;
            }
            config_properties[key] = b;
        } else if (key == "padding_side") {
            config_properties[key] = value;
        } else {
            plugin_properties[key] = value;
        }
    }
    return ov_status_e::OK;
}

ov_status_e make_result(const std::vector<std::pair<size_t, float>>& items,
                        ov_genai_text_rerank_result** out) {
    std::unique_ptr<ov_genai_text_rerank_result> r = std::make_unique<ov_genai_text_rerank_result>();
    r->items = items;
    *out = r.release();
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

}  // namespace

void ov_genai_text_rerank_result_free(ov_genai_text_rerank_result* result) {
    if (result) {
        delete result;
    }
}

ov_status_e ov_genai_text_rerank_result_get_size(const ov_genai_text_rerank_result* result, size_t* size) {
    if (!result || !size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    *size = result->items.size();
    return ov_status_e::OK;
}

ov_status_e ov_genai_text_rerank_result_get_item(const ov_genai_text_rerank_result* result,
                                                 size_t i,
                                                 size_t* index,
                                                 float* score) {
    if (!result || !index || !score) {
        return ov_status_e::INVALID_C_PARAM;
    }
    if (i >= result->items.size()) {
        return ov_status_e::OUT_OF_BOUNDS;
    }
    *index = result->items[i].first;
    *score = result->items[i].second;
    return ov_status_e::OK;
}

ov_status_e ov_genai_text_rerank_pipeline_create(const char* models_path,
                                                 const char* device,
                                                 const size_t property_args_size,
                                                 ov_genai_text_rerank_pipeline** pipe,
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
            split_rerank_properties(args_ptr, property_args_size / 2, config_properties, plugin_properties);
        va_end(args_ptr);
        if (split_status != ov_status_e::OK) {
            return split_status;
        }
        ov::genai::TextRerankPipeline::Config config(config_properties);
        std::unique_ptr<ov_genai_text_rerank_pipeline> _pipe = std::make_unique<ov_genai_text_rerank_pipeline>();
        _pipe->object = std::make_shared<ov::genai::TextRerankPipeline>(std::filesystem::path(models_path),
                                                                       std::string(device),
                                                                       config,
                                                                       plugin_properties);
        *pipe = _pipe.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_text_rerank_pipeline_free(ov_genai_text_rerank_pipeline* pipe) {
    if (pipe) {
        delete pipe;
    }
}

ov_status_e ov_genai_text_rerank_pipeline_rerank(ov_genai_text_rerank_pipeline* pipe,
                                                 const char* query,
                                                 const char** texts,
                                                 size_t texts_count,
                                                 ov_genai_text_rerank_result** result) {
    if (!pipe || !(pipe->object) || !query || !result || (texts_count > 0 && !texts)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<std::string> docs = to_string_vector(texts, texts_count);
        auto items = pipe->object->rerank(std::string(query), docs);
        return make_result(items, result);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text_rerank_pipeline_start_rerank_async(ov_genai_text_rerank_pipeline* pipe,
                                                             const char* query,
                                                             const char** texts,
                                                             size_t texts_count) {
    if (!pipe || !(pipe->object) || !query || (texts_count > 0 && !texts)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<std::string> docs = to_string_vector(texts, texts_count);
        pipe->object->start_rerank_async(std::string(query), docs);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text_rerank_pipeline_wait_rerank(ov_genai_text_rerank_pipeline* pipe,
                                                      ov_genai_text_rerank_result** result) {
    if (!pipe || !(pipe->object) || !result) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto items = pipe->object->wait_rerank();
        return make_result(items, result);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}
