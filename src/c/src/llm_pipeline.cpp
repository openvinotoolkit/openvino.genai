// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/llm_pipeline.h"

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "types_c.h"

ov_status_e ov_genai_decoded_results_create(ov_genai_decoded_results** results) {
    if (!results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_decoded_results> _results = std::make_unique<ov_genai_decoded_results>();
        _results->object = std::make_shared<ov::genai::DecodedResults>();
        *results = _results.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
void ov_genai_decoded_results_free(ov_genai_decoded_results* results) {
    if (results) {
        delete results;
    }
}
ov_status_e ov_genai_decoded_results_get_perf_metrics(const ov_genai_decoded_results* results,
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
void ov_genai_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics) {
    if (metrics) {
        delete metrics;
    }
}
ov_status_e ov_genai_decoded_results_get_string(const ov_genai_decoded_results* results,
                                                char** output,
                                                size_t* output_size) {
    if (!results || !(results->object) || !(output && output_size)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string str = *(results->object);
        if (*output) {
            delete[] *output;  // Free the previous memory if exits
        }
        try {
            *output = new char[str.length() + 1];
        } catch (const std::bad_alloc& e) {
            *output = nullptr;  // Ensure that output is NULL if allocation fails
            return ov_status_e::NOT_ALLOCATED;
        }
        strncpy(*output, str.c_str(), str.length() + 1);
        (*output)[str.length()] = '\0';
        *output_size = str.length() + 1;

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_llm_pipeline_create(const char* models_path, const char* device, ov_genai_llm_pipeline** pipe) {
    if (!models_path || !device || !pipe) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_llm_pipeline> _pipe = std::make_unique<ov_genai_llm_pipeline>();
        _pipe->object =
            std::make_shared<ov::genai::LLMPipeline>(std::filesystem::path(models_path), std::string(device));
        *pipe = _pipe.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_llm_pipeline_free(ov_genai_llm_pipeline* pipe) {
    if (pipe) {
        delete pipe;
    }
}
ov_status_e ov_genai_llm_pipeline_generate(ov_genai_llm_pipeline* pipe,
                                           const char* inputs,
                                           const ov_genai_generation_config* config,
                                           const stream_callback* streamer,
                                           char** output,
                                           size_t* output_size) {
    if (!pipe || !(pipe->object) || !inputs || !(streamer || (output && output_size))) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string input_str(inputs);
        ov::genai::StringInputs input = {input_str};
        std::string results;
        if (streamer) {
            auto callback = [streamer](std::string word) -> ov::genai::StreamingStatus {
                return static_cast<ov::genai::StreamingStatus>((*streamer)(word.c_str()));
            };
            results = (config && config->object) ? pipe->object->generate(input, *(config->object), callback)
                                                 : pipe->object->generate(input, {}, callback);
        } else {
            results = (config && config->object) ? pipe->object->generate(input, *(config->object))
                                                 : pipe->object->generate(input);
        }
        if (output) {
            if (*output) {
                delete[] *output;  // Free the previous memory if exits
            }
            try {
                *output = new char[results.length() + 1];
            } catch (const std::bad_alloc& e) {
                *output = nullptr;  // Ensure that output is NULL if allocation fails
                return ov_status_e::NOT_ALLOCATED;
            }
            strncpy(*output, results.c_str(), results.length() + 1);
            (*output)[results.length()] = '\0';
            *output_size = results.length() + 1;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_llm_pipeline_generate_decoded_results(ov_genai_llm_pipeline* pipe,
                                                           const char* inputs,
                                                           const ov_genai_generation_config* config,
                                                           const stream_callback* streamer,
                                                           ov_genai_decoded_results** results) {
    if (!pipe || !(pipe->object) || !inputs || !results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_decoded_results> _results = std::make_unique<ov_genai_decoded_results>();
        _results->object = std::make_shared<ov::genai::DecodedResults>();
        std::string input_str(inputs);
        ov::genai::StringInputs input = {input_str};
        if (streamer) {
            auto callback = [streamer](std::string word) -> ov::genai::StreamingStatus {
                return static_cast<ov::genai::StreamingStatus>((*streamer)(word.c_str()));
            };
            *(_results->object) = (config && config->object)
                                      ? pipe->object->generate(input, *(config->object), callback)
                                      : pipe->object->generate(input, {}, callback);
        } else {
            *(_results->object) = (config && config->object) ? pipe->object->generate(input, *(config->object))
                                                             : pipe->object->generate(input);
        }
        *results = _results.release();

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_llm_pipeline_start_chat(ov_genai_llm_pipeline* pipe) {
    if (!pipe || !(pipe->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipe->object->start_chat();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_llm_pipeline_finish_chat(ov_genai_llm_pipeline* pipe) {
    if (!pipe || !(pipe->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipe->object->finish_chat();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_llm_pipeline_get_generation_config(const ov_genai_llm_pipeline* pipe,
                                                        ov_genai_generation_config** config) {
    if (!pipe || !(pipe->object) || !config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_generation_config> _config = std::make_unique<ov_genai_generation_config>();
        _config->object = std::make_shared<ov::genai::GenerationConfig>(pipe->object->get_generation_config());
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_llm_pipeline_set_generation_config(ov_genai_llm_pipeline* pipe,
                                                        ov_genai_generation_config* config) {
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
