// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/genai/c/text2speech_pipeline.h"

#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "openvino/c/ov_tensor.h"
#include "openvino/genai/c/perf_metrics.h"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "types_c.h"


ov_status_e ov_genai_text2speech_decoded_results_create(ov_genai_text2speech_decoded_results** results) {
    if (!results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_text2speech_decoded_results> _results =
            std::make_unique<ov_genai_text2speech_decoded_results>();
        _results->object = std::make_shared<ov::genai::Text2SpeechDecodedResults>();
        *results = _results.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_text2speech_decoded_results_free(ov_genai_text2speech_decoded_results* results) {
    if (results) {
        delete results;
    }
}

ov_status_e ov_genai_text2speech_decoded_results_get_perf_metrics(const ov_genai_text2speech_decoded_results* results,
                                                                  ov_genai_perf_metrics** metrics) {
    if (!results || !(results->object) || !metrics) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_perf_metrics> _metrics = std::make_unique<ov_genai_perf_metrics>();
        _metrics->object = std::make_shared<ov::genai::SpeechGenerationPerfMetrics>(results->object->perf_metrics);
        *metrics = _metrics.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2speech_decoded_results_get_speeches_count(const ov_genai_text2speech_decoded_results* results,
                                                                    size_t* count) {
    if (!results || !(results->object) || !count) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *count = results->object->speeches.size();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2speech_decoded_results_get_speech_at(const ov_genai_text2speech_decoded_results* results,
                                                               size_t index,
                                                               ov_tensor_t** speech) {
    if (!results || !(results->object) || !speech) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (index >= results->object->speeches.size()) {
            return ov_status_e::OUT_OF_BOUNDS;
        }
        const ov::Tensor& cpp_tensor = results->object->speeches.at(index);
        ov_element_type_e et = (ov_element_type_e)cpp_tensor.get_element_type();

        ov::Shape cpp_shape = cpp_tensor.get_shape();
        std::vector<size_t> dims(cpp_shape.begin(), cpp_shape.end());
        std::vector<int64_t> dims_i64(dims.begin(), dims.end());
        ov_shape_t shape;
        ov_shape_create(dims_i64.size(), dims_i64.data(), &shape);

        // Explicitly create a new tensor and copy data to resolve life time issues
        ov_status_e status = ov_tensor_create(et, shape, speech);
        if (status == ov_status_e::OK) {
            void* data_ptr = nullptr;
            ov_tensor_data(*speech, &data_ptr);
            std::memcpy(data_ptr, cpp_tensor.data(), cpp_tensor.get_byte_size());
        }
        ov_shape_free(&shape);
        return status;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

ov_status_e ov_genai_text2speech_pipeline_create(const char* models_path,
                                                 const char* device,
                                                 const size_t property_args_size,
                                                 ov_genai_text2speech_pipeline** pipe,
                                                 ...) {
    if (!models_path || !device || !pipe || property_args_size % 2 != 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        ov::AnyMap property = {};
        va_list args_ptr;
        va_start(args_ptr, pipe);
        size_t property_size = property_args_size / 2;
        for (size_t i = 0; i < property_size; i++) {
            GET_PROPERTY_FROM_ARGS_LIST;
        }
        va_end(args_ptr);

        std::unique_ptr<ov_genai_text2speech_pipeline> _pipe = std::make_unique<ov_genai_text2speech_pipeline>();
        _pipe->object = std::make_shared<ov::genai::Text2SpeechPipeline>(std::filesystem::path(models_path),
                                                                         std::string(device),
                                                                         property);
        *pipe = _pipe.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_text2speech_pipeline_free(ov_genai_text2speech_pipeline* pipe) {
    if (pipe) {
        delete pipe;
    }
}

ov_status_e ov_genai_text2speech_pipeline_generate(ov_genai_text2speech_pipeline* pipe,
                                                   const char** texts,
                                                   size_t texts_size,
                                                   const ov_tensor_t* speaker_embedding,
                                                   const size_t property_args_size,
                                                   ov_genai_text2speech_decoded_results** results,
                                                   ...) {
    if (!pipe || !(pipe->object) || !texts || !results || property_args_size % 2 != 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::vector<std::string> texts_cpp;
        texts_cpp.reserve(texts_size);
        for (size_t i = 0; i < texts_size; ++i) {
            if (!texts[i]) {
                return ov_status_e::INVALID_C_PARAM;
            }
            texts_cpp.emplace_back(texts[i]);
        }

        ov::Tensor speaker_embedding_cpp;
        if (speaker_embedding) {
            ov_shape_t shape_c{};
            ov_tensor_get_shape(speaker_embedding, &shape_c);
            std::vector<size_t> dims(shape_c.rank);
            for (size_t d = 0; d < shape_c.rank; ++d)
                dims[d] = shape_c.dims[d];
            ov_shape_free(&shape_c);

            void* data_ptr = nullptr;
            ov_tensor_data(const_cast<ov_tensor_t*>(speaker_embedding), &data_ptr);

            ov_element_type_e et;
            ov_tensor_get_element_type(speaker_embedding, &et);

            speaker_embedding_cpp = ov::Tensor(ov::element::Type((ov::element::Type_t)et), ov::Shape(dims), data_ptr);
        }

        ov::AnyMap property = {};
        va_list args_ptr;
        va_start(args_ptr, results);
        size_t property_size = property_args_size / 2;
        for (size_t i = 0; i < property_size; i++) {
            GET_PROPERTY_FROM_ARGS_LIST;
        }
        va_end(args_ptr);

        std::unique_ptr<ov_genai_text2speech_decoded_results> _results =
            std::make_unique<ov_genai_text2speech_decoded_results>();
        _results->object = std::make_shared<ov::genai::Text2SpeechDecodedResults>();

        *(_results->object) = pipe->object->generate(texts_cpp, speaker_embedding_cpp, property);

        *results = _results.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2speech_pipeline_get_generation_config(const ov_genai_text2speech_pipeline* pipe,
                                                                ov_genai_speech_generation_config** config) {
    if (!pipe || !(pipe->object) || !config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_speech_generation_config> _config =
            std::make_unique<ov_genai_speech_generation_config>();
        _config->object = std::make_shared<ov::genai::SpeechGenerationConfig>(pipe->object->get_generation_config());
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2speech_pipeline_set_generation_config(ov_genai_text2speech_pipeline* pipe,
                                                                const ov_genai_speech_generation_config* config) {
    if (!pipe || !(pipe->object) || !config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipe->object->set_generation_config(*(config->object));
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
