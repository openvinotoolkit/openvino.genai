// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/whisper_pipeline.h"

#include <stdarg.h>

#include <filesystem>

#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "types_c.h"

// WhisperDecodedResultChunk implementation
ov_status_e ov_genai_whisper_decoded_result_chunk_create(ov_genai_whisper_decoded_result_chunk** chunk) {
    if (!chunk) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_whisper_decoded_result_chunk> _chunk =
            std::make_unique<ov_genai_whisper_decoded_result_chunk>();
        _chunk->object = std::make_shared<ov::genai::WhisperDecodedResultChunk>();
        *chunk = _chunk.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_whisper_decoded_result_chunk_free(ov_genai_whisper_decoded_result_chunk* chunk) {
    if (chunk) {
        delete chunk;
    }
}

ov_status_e ov_genai_whisper_decoded_result_chunk_get_start_ts(const ov_genai_whisper_decoded_result_chunk* chunk,
                                                               float* start_ts) {
    if (!chunk || !(chunk->object) || !start_ts) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *start_ts = chunk->object->start_ts;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_result_chunk_get_end_ts(const ov_genai_whisper_decoded_result_chunk* chunk,
                                                             float* end_ts) {
    if (!chunk || !(chunk->object) || !end_ts) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *end_ts = chunk->object->end_ts;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_result_chunk_get_text(const ov_genai_whisper_decoded_result_chunk* chunk,
                                                           char* text,
                                                           size_t* text_size) {
    if (!chunk || !(chunk->object) || !text_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        const std::string& str = chunk->object->text;
        if (!text) {
            *text_size = str.length() + 1;
        } else {
            if (*text_size < str.length() + 1) {
                return ov_status_e::OUT_OF_BOUNDS;
            }
            strncpy(text, str.c_str(), str.length() + 1);
            text[str.length()] = '\0';
            *text_size = str.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

// WhisperDecodedResults implementation
ov_status_e ov_genai_whisper_decoded_results_create(ov_genai_whisper_decoded_results** results) {
    if (!results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_whisper_decoded_results> _results =
            std::make_unique<ov_genai_whisper_decoded_results>();
        _results->object = std::make_shared<ov::genai::WhisperDecodedResults>();
        *results = _results.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_whisper_decoded_results_free(ov_genai_whisper_decoded_results* results) {
    if (results) {
        delete results;
    }
}

ov_status_e ov_genai_whisper_decoded_results_get_perf_metrics(const ov_genai_whisper_decoded_results* results,
                                                              ov_genai_perf_metrics** metrics) {
    if (!results || !(results->object) || !metrics) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_perf_metrics> _metrics = std::make_unique<ov_genai_perf_metrics>();
        _metrics->object = std::make_shared<ov::genai::PerfMetrics>(results->object->perf_metrics);
        *metrics = _metrics.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_results_get_texts_count(const ov_genai_whisper_decoded_results* results,
                                                             size_t* count) {
    if (!results || !(results->object) || !count) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *count = results->object->texts.size();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_results_get_text_at(const ov_genai_whisper_decoded_results* results,
                                                         size_t index,
                                                         char* text,
                                                         size_t* text_size) {
    if (!results || !(results->object) || !text_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (index >= results->object->texts.size()) {
            return ov_status_e::OUT_OF_BOUNDS;
        }
        const std::string& str = results->object->texts[index];
        if (!text) {
            *text_size = str.length() + 1;
        } else {
            if (*text_size < str.length() + 1) {
                return ov_status_e::OUT_OF_BOUNDS;
            }
            strncpy(text, str.c_str(), str.length() + 1);
            text[str.length()] = '\0';
            *text_size = str.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_results_get_score_at(const ov_genai_whisper_decoded_results* results,
                                                          size_t index,
                                                          float* score) {
    if (!results || !(results->object) || !score) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (index >= results->object->scores.size()) {
            return ov_status_e::OUT_OF_BOUNDS;
        }
        *score = results->object->scores[index];
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_results_has_chunks(const ov_genai_whisper_decoded_results* results,
                                                        bool* has_chunks) {
    if (!results || !(results->object) || !has_chunks) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *has_chunks = results->object->chunks.has_value();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_results_get_chunks_count(const ov_genai_whisper_decoded_results* results,
                                                              size_t* count) {
    if (!results || !(results->object) || !count) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (results->object->chunks.has_value()) {
            *count = results->object->chunks->size();
        } else {
            *count = 0;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_results_get_chunk_at(const ov_genai_whisper_decoded_results* results,
                                                          size_t index,
                                                          ov_genai_whisper_decoded_result_chunk** chunk) {
    if (!results || !(results->object) || !chunk) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        if (!results->object->chunks.has_value()) {
            return ov_status_e::NOT_FOUND;
        }
        if (index >= results->object->chunks->size()) {
            return ov_status_e::OUT_OF_BOUNDS;
        }

        std::unique_ptr<ov_genai_whisper_decoded_result_chunk> _chunk =
            std::make_unique<ov_genai_whisper_decoded_result_chunk>();
        _chunk->object = std::make_shared<ov::genai::WhisperDecodedResultChunk>(results->object->chunks->at(index));
        *chunk = _chunk.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_decoded_results_get_string(const ov_genai_whisper_decoded_results* results,
                                                        char* output,
                                                        size_t* output_size) {
    if (!results || !(results->object) || !output_size) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string str = *(results->object);
        if (!output) {
            *output_size = str.length() + 1;
        } else {
            if (*output_size < str.length() + 1) {
                return ov_status_e::OUT_OF_BOUNDS;
            }
            strncpy(output, str.c_str(), str.length() + 1);
            output[str.length()] = '\0';
            *output_size = str.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

// WhisperPipeline implementation
ov_status_e ov_genai_whisper_pipeline_create(const char* models_path,
                                             const char* device,
                                             const size_t property_args_size,
                                             ov_genai_whisper_pipeline** pipeline,
                                             ...) {
    if (!models_path || !device || !pipeline || property_args_size % 2 != 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        ov::AnyMap property = {};
        va_list args_ptr;
        va_start(args_ptr, pipeline);
        size_t property_size = property_args_size / 2;
        for (size_t i = 0; i < property_size; i++) {
            GET_PROPERTY_FROM_ARGS_LIST;
        }
        va_end(args_ptr);

        std::unique_ptr<ov_genai_whisper_pipeline> _pipeline = std::make_unique<ov_genai_whisper_pipeline>();
        _pipeline->object = std::make_shared<ov::genai::WhisperPipeline>(std::filesystem::path(models_path),
                                                                         std::string(device),
                                                                         property);
        *pipeline = _pipeline.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_whisper_pipeline_free(ov_genai_whisper_pipeline* pipeline) {
    if (pipeline) {
        delete pipeline;
    }
}

ov_status_e ov_genai_whisper_pipeline_generate(ov_genai_whisper_pipeline* pipeline,
                                               const float* raw_speech,
                                               size_t raw_speech_size,
                                               const ov_genai_whisper_generation_config* config,
                                               ov_genai_whisper_decoded_results** results) {
    if (!pipeline || !(pipeline->object) || !raw_speech || !results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_whisper_decoded_results> _results =
            std::make_unique<ov_genai_whisper_decoded_results>();
        _results->object = std::make_shared<ov::genai::WhisperDecodedResults>();

        // Convert raw speech input to vector
        ov::genai::RawSpeechInput speech_input(raw_speech, raw_speech + raw_speech_size);

        // Generate results
        if (config && config->object) {
            *(_results->object) = pipeline->object->generate(speech_input, *(config->object));
        } else {
            *(_results->object) = pipeline->object->generate(speech_input);
        }

        *results = _results.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_pipeline_get_generation_config(const ov_genai_whisper_pipeline* pipeline,
                                                            ov_genai_whisper_generation_config** config) {
    if (!pipeline || !(pipeline->object) || !config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_whisper_generation_config> _config =
            std::make_unique<ov_genai_whisper_generation_config>();
        _config->object =
            std::make_shared<ov::genai::WhisperGenerationConfig>(pipeline->object->get_generation_config());
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_whisper_pipeline_set_generation_config(ov_genai_whisper_pipeline* pipeline,
                                                            ov_genai_whisper_generation_config* config) {
    if (!pipeline || !(pipeline->object) || !config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipeline->object->set_generation_config(*(config->object));
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
