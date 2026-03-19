// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2video_pipeline.h"
#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "openvino/genai/video_generation/generation_config.hpp"
#include "openvino/c/ov_tensor.h"
#include "types_c.h"
#include <stdarg.h>

ov_status_e ov_genai_text2video_pipeline_create(const char* models_path, ov_genai_text2video_pipeline** pipe) {
    if (!models_path || !pipe) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_text2video_pipeline> _pipe = std::make_unique<ov_genai_text2video_pipeline>();
        _pipe->object = std::make_shared<ov::genai::Text2VideoPipeline>(std::filesystem::path(models_path));
        *pipe = _pipe.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2video_pipeline_create_with_device(const char* models_path, const char* device, const size_t property_args_size, ov_genai_text2video_pipeline** pipe, ...) {
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
        
        std::unique_ptr<ov_genai_text2video_pipeline> _pipe = std::make_unique<ov_genai_text2video_pipeline>();
        _pipe->object = std::make_shared<ov::genai::Text2VideoPipeline>(std::filesystem::path(models_path), std::string(device), property);
        *pipe = _pipe.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_text2video_pipeline_free(ov_genai_text2video_pipeline* pipe) {
    if (pipe) {
        delete pipe;
    }
}

ov_status_e ov_genai_text2video_pipeline_compile(ov_genai_text2video_pipeline* pipe,
                                                 const char* device,
                                                 const size_t property_args_size,
                                                 ...) {
    if (!pipe || !pipe->object || !device || property_args_size % 2 != 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        ov::AnyMap property = {};
        va_list args_ptr;
        va_start(args_ptr, property_args_size);
        size_t property_size = property_args_size / 2;
        for (size_t i = 0; i < property_size; i++) {
            GET_PROPERTY_FROM_ARGS_LIST;
        }
        va_end(args_ptr);
        
        pipe->object->compile(std::string(device), property);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2video_pipeline_generate(ov_genai_text2video_pipeline* pipe,
                                                  const char* positive_prompt,
                                                  const size_t property_args_size,
                                                  ov_genai_video_generation_result** result,
                                                  ...) {
    if (!pipe || !pipe->object || !positive_prompt || !result || property_args_size % 2 != 0) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        ov::AnyMap property = {};
        va_list args_ptr;
        va_start(args_ptr, result);
        size_t property_size = property_args_size / 2;
        for (size_t i = 0; i < property_size; i++) {
            GET_PROPERTY_FROM_ARGS_LIST;
        }
        va_end(args_ptr);
        
        std::unique_ptr<ov_genai_video_generation_result> _result = std::make_unique<ov_genai_video_generation_result>();
        _result->object = std::make_shared<ov::genai::VideoGenerationResult>();
        
        *(_result->object) = pipe->object->generate(std::string(positive_prompt), property);
        
        *result = _result.release();

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_video_generation_result_free(ov_genai_video_generation_result* result) {
    if (result) {
        delete result;
    }
}

ov_status_e ov_genai_video_generation_result_get_video(const ov_genai_video_generation_result* result,
                                                       ov_tensor_t** tensor) {
    if (!result || !(result->object) || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        ov::Tensor video_tensor = result->object->video;
        ov::Shape cpp_shape = video_tensor.get_shape();
        std::vector<int64_t> dims(cpp_shape.begin(), cpp_shape.end());
        ov_shape_t shape;
        ov_shape_create(cpp_shape.size(), dims.data(), &shape);

        ov_element_type_e type = static_cast<ov_element_type_e>(video_tensor.get_element_type());
        
        ov_status_e status = ov_tensor_create_from_host_ptr(type, shape, video_tensor.data(), tensor);
        ov_shape_free(&shape);
        return status;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_video_generation_result_get_perf_metrics(const ov_genai_video_generation_result* result,
                                                              ov_genai_video_generation_perf_metrics** metrics) {
    if (!result || !(result->object) || !metrics) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_video_generation_perf_metrics> _metrics = std::make_unique<ov_genai_video_generation_perf_metrics>();
        _metrics->object = std::make_shared<ov::genai::VideoGenerationPerfMetrics>(result->object->performance_stat);
        *metrics = _metrics.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_video_generation_perf_metrics_free(ov_genai_video_generation_perf_metrics* metrics) {
    if (metrics) {
        delete metrics;
    }
}

ov_status_e ov_genai_text2video_pipeline_get_generation_config(const ov_genai_text2video_pipeline* pipe,
                                                               ov_genai_video_generation_config** config) {
    if (!pipe || !pipe->object || !config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_video_generation_config> _config = std::make_unique<ov_genai_video_generation_config>();
        _config->object = std::make_shared<ov::genai::VideoGenerationConfig>(pipe->object->get_generation_config());
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2video_pipeline_set_generation_config(ov_genai_text2video_pipeline* pipe,
                                                               ov_genai_video_generation_config* config) {
    if (!pipe || !pipe->object || !config || !config->object) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        pipe->object->set_generation_config(*(config->object));
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_video_generation_config_create(ov_genai_video_generation_config** config) {
    if (!config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_video_generation_config> _config = std::make_unique<ov_genai_video_generation_config>();
        _config->object = std::make_shared<ov::genai::VideoGenerationConfig>();
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_video_generation_config_free(ov_genai_video_generation_config* config) {
    if (config) {
        delete config;
    }
}
