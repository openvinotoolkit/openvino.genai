// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/llm_pipeline_c.h"

#include "openvino/genai/c/generation_config_c.h"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "types_c.h"

ov_status_e ov_genai_decoded_results_create(ov_genai_decoded_results** results) {
    if (!results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_decoded_results> _results(new ov_genai_decoded_results);
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
ov_status_e ov_genai_decoded_results_get_perf_metrics(ov_genai_decoded_results* results,
                                                      ov_genai_perf_metrics** metrics) {
    if (!results || !(results->object) || !metrics) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_perf_metrics> _metrics(new ov_genai_perf_metrics);
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
ov_status_e ov_genai_decoded_results_get_string(ov_genai_decoded_results* results, char* output, int max_size) {
    if (!results || !(results->object) || !output) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string str = *(results->object);
        strncpy(output, str.c_str(), max_size - 1);
        output[max_size - 1] = '\0';
        if (str.length() + 1 > max_size) {
            return ov_status_e::OUT_OF_BOUNDS;
        }
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
        std::unique_ptr<ov_genai_llm_pipeline> _pipe(new ov_genai_llm_pipeline);
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
                                           char* output,
                                           int max_size,
                                           ov_genai_generation_config* config) {
    if (!pipe || !(pipe->object) || !inputs || !output) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string input_str(inputs);
        ov::genai::StringInputs input = {input_str};
        std::string results;
        results = (config && config->object) ? pipe->object->generate(input, *(config->object))
                                             : pipe->object->generate(input);
        strncpy(output, results.c_str(), max_size - 1);
        output[max_size - 1] = '\0';
        if (results.length() + 1 > max_size) {
            return ov_status_e::OUT_OF_BOUNDS;
        }

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_llm_pipeline_generate_stream(ov_genai_llm_pipeline* pipe,
                                                  const char* inputs,
                                                  char* output,
                                                  int max_size,
                                                  ov_genai_generation_config* config,
                                                  char* buffer,
                                                  const int buffer_size,
                                                  int* buffer_pos) {
    if (!pipe || !(pipe->object) || !inputs || !output || !buffer || !buffer_pos) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string input_str(inputs);
        ov::genai::StringInputs input = {input_str};
        auto stream = [&](const std::string& word) -> bool {
            if ((*buffer_pos) + word.size() + 1 < buffer_size) {
                std::strcpy(buffer + (*buffer_pos), word.c_str());
                (*buffer_pos) += word.size();
            }
            return false;
        };
        std::string results = (config && config->object) ? pipe->object->generate(input, *(config->object), stream)
                                                         : pipe->object->generate(input, {}, stream);
        strncpy(output, results.c_str(), max_size - 1);
        output[max_size - 1] = '\0';
        if (results.length() + 1 > max_size) {
            return ov_status_e::OUT_OF_BOUNDS;
        }
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_llm_pipeline_generate_decode_results(ov_genai_llm_pipeline* pipe,
                                                          const char* inputs,
                                                          ov_genai_generation_config* config,
                                                          ov_genai_decoded_results** results) {
    if (!pipe || !(pipe->object) || !inputs || !results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_decoded_results> _results(new ov_genai_decoded_results);
        _results->object = std::make_shared<ov::genai::DecodedResults>();
        std::string input_str(inputs);
        ov::genai::StringInputs input = {input_str};
        *(_results->object) =
            (config->object) ? pipe->object->generate(input, *(config->object)) : pipe->object->generate(input);
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
ov_status_e ov_genai_llm_pipeline_get_generation_config(ov_genai_llm_pipeline* pipe,
                                                        ov_genai_generation_config** config) {
    if (!pipe || !(pipe->object) || !config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_generation_config> _config(new ov_genai_generation_config);
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
