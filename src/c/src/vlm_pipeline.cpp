// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/vlm_pipeline.h"

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/c/ov_tensor.h"
#include "types_c.h"
#include <stdarg.h>

ov_status_e ov_genai_vlm_decoded_results_create(ov_genai_vlm_decoded_results** results) {
    if (!results) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_vlm_decoded_results> _results = std::make_unique<ov_genai_vlm_decoded_results>();
        _results->object = std::make_shared<ov::genai::VLMDecodedResults>();
        *results = _results.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_vlm_decoded_results_free(ov_genai_vlm_decoded_results* results) {
    if (results) {
        delete results;
    }
}

ov_status_e ov_genai_vlm_decoded_results_get_perf_metrics(const ov_genai_vlm_decoded_results* results,
                                                          ov_genai_perf_metrics** metrics) {
    if (!results || !(results->object) || !metrics) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_perf_metrics> _metrics = std::make_unique<ov_genai_perf_metrics>();
        _metrics->object = std::make_shared<ov::genai::VLMPerfMetrics>(results->object->perf_metrics);
        *metrics = _metrics.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_vlm_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics) {
    if (metrics) {
        delete metrics;
    }
}

ov_status_e ov_genai_vlm_decoded_results_get_string(const ov_genai_vlm_decoded_results* results,
                                                    char* output,
                                                    size_t* output_size) {
    if (!results || !(results->object) || !(output_size)) {
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

ov_status_e ov_genai_vlm_pipeline_create(const char* models_path, const char* device, const size_t property_args_size, ov_genai_vlm_pipeline** pipe, ...) {
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
        // Check Property MAX_PROMPT_LEN and MIN_RESPONSE_LEN for NPU
        // These two special properties, which only affect the genai level, should be manually converted to integers 
        // before being passed to the constructor of VLMPipeline
        if(std::string(device) == "NPU") {
            if(property.find("MAX_PROMPT_LEN") != property.end()) {
                std::string max_prompt_len = property["MAX_PROMPT_LEN"].as<std::string>();
                property.erase("MAX_PROMPT_LEN");
                property["MAX_PROMPT_LEN"] = std::stoi(max_prompt_len);
            }
            if(property.find("MIN_RESPONSE_LEN") != property.end()) {
                std::string min_response_len = property["MIN_RESPONSE_LEN"].as<std::string>();
                property.erase("MIN_RESPONSE_LEN");
                property["MIN_RESPONSE_LEN"] = std::stoi(min_response_len);
            }
        }
        std::unique_ptr<ov_genai_vlm_pipeline> _pipe = std::make_unique<ov_genai_vlm_pipeline>();
        _pipe->object =
            std::make_shared<ov::genai::VLMPipeline>(std::filesystem::path(models_path), std::string(device), property);
        *pipe = _pipe.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_vlm_pipeline_free(ov_genai_vlm_pipeline* pipe) {
    if (pipe) {
        delete pipe;
    }
}

ov_status_e ov_genai_vlm_pipeline_generate(ov_genai_vlm_pipeline* pipe,
                                           const char* text_inputs,
                                           const ov_tensor_t** rgbs,
                                           size_t num_images,
                                           const ov_genai_generation_config* config,
                                           const streamer_callback* streamer,
                                           ov_genai_vlm_decoded_results** results) {
    if (!pipe || !(pipe->object) || !text_inputs || !(streamer || results)) {
        return ov_status_e::INVALID_C_PARAM;
    }

    std::vector<ov::Tensor> rgbs_cpp;
    rgbs_cpp.reserve(num_images);
    for (size_t i = 0; i < num_images; ++i) {
        const ov_tensor* ct = rgbs[i];

        auto et = ov::element::Type_t::u8;

        ov_shape_t shape_c{};
        ov_tensor_get_shape(ct, &shape_c);
        std::vector<size_t> dims(shape_c.rank);
        for (size_t d = 0; d < shape_c.rank; ++d) dims[d] = shape_c.dims[d];
        ov_shape_free(&shape_c);

        void* data_ptr = nullptr;
        ov_tensor_data(const_cast<ov_tensor*>(ct), &data_ptr);
        if (!data_ptr) {
            return ov_status_e::INVALID_C_PARAM;
        }

        rgbs_cpp.emplace_back(ov::element::Type(et), ov::Shape(dims), data_ptr);
    }

    try {
        std::unique_ptr<ov_genai_vlm_decoded_results> _results = std::make_unique<ov_genai_vlm_decoded_results>();
        _results->object = std::make_shared<ov::genai::VLMDecodedResults>();
        std::string input_str(text_inputs);
        ov::genai::StringInputs input = {input_str};
        
        if (streamer) {
            auto callback = [streamer](std::string word) -> ov::genai::StreamingStatus {
                return static_cast<ov::genai::StreamingStatus>((streamer->callback_func)(word.c_str(), streamer->args));
            };
            if (num_images > 0) {
                *(_results->object) = (config && config->object)
                                        ? pipe->object->generate(input_str, rgbs_cpp, *(config->object), callback)
                                        : pipe->object->generate(input_str, rgbs_cpp, {}, callback);
            }
            if (num_images == 0){
                *(_results->object) = (config && config->object)
                ? pipe->object->generate(input_str, ov::genai::generation_config(*(config->object)), ov::genai::streamer(callback))
                : pipe->object->generate(input_str, ov::genai::streamer(callback));
            }
        } 
        if (results) {
            *results = _results.release();
        }

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
} 

ov_status_e ov_genai_vlm_pipeline_start_chat(ov_genai_vlm_pipeline* pipe) {
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

ov_status_e ov_genai_vlm_pipeline_finish_chat(ov_genai_vlm_pipeline* pipe) {
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

ov_status_e ov_genai_vlm_pipeline_get_generation_config(const ov_genai_vlm_pipeline* pipe,
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

ov_status_e ov_genai_vlm_pipeline_set_generation_config(ov_genai_vlm_pipeline* pipe,
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
