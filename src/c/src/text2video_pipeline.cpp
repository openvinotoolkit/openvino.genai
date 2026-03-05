// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2video_pipeline.h"
#include "openvino/genai/c/video_generation_config.h"
#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "openvino/genai/video_generation/generation_config.hpp"
// #include ""

#include "types_c.h"

#include <stdarg.h>
#include <memory>
#include <string>

// -------------------------------------------------------------------------
// Pipeline Creation & Destruction
// -------------------------------------------------------------------------

ov_status_e ov_genai_text2video_pipeline_create(const char* models_path, 
                                                const char* device, 
                                                const size_t property_args_size, 
                                                ov_genai_text2video_pipeline** pipe, 
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
            GET_PROPERTY_FROM_ARGS_LIST; // Macro defined in types_c.h
        }
        va_end(args_ptr);

        std::unique_ptr<ov_genai_text2video_pipeline> _pipe = std::make_unique<ov_genai_text2video_pipeline>();
        _pipe->object = std::make_shared<ov::genai::Text2VideoPipeline>(
            std::filesystem::path(models_path), 
            std::string(device), 
            property
        );
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

// -------------------------------------------------------------------------
// Generation
// -------------------------------------------------------------------------

ov_status_e ov_genai_text2video_pipeline_generate(ov_genai_text2video_pipeline* pipe,
                                                  const char* prompt,
                                                  const ov_genai_video_generation_config* config,
                                                  ov_tensor_t** video_tensor) {
    if (!pipe || !(pipe->object) || !prompt || !video_tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::string prompt_str(prompt);
        ov::AnyMap properties;

        // MANUALLY populate properties to avoid the linker error
        if (config && config->object) {
            properties["width"] = config->object->width;
            properties["height"] = config->object->height;
            properties["num_frames"] = config->object->num_frames;
            properties["num_inference_steps"] = config->object->num_inference_steps;
            // Add any other specific video parameters here if needed
        }

        // Call C++ generate - returns VideoGenerationResult
        auto result = pipe->object->generate(prompt_str, properties);

        // Extract the generated tensor
        ov::Tensor cpp_tensor = result.video;

        // Bridge: Convert C++ ov::Tensor to C ov_tensor_t
        ov_shape_t shape;
        shape.rank = cpp_tensor.get_shape().size();
        shape.dims = (int64_t*)malloc(sizeof(int64_t) * shape.rank);
        for (size_t i = 0; i < shape.rank; ++i) {
            shape.dims[i] = (int64_t)cpp_tensor.get_shape()[i];
        }
        
        // Map C++ element type to C element type enum
        ov_element_type_e c_type = static_cast<ov_element_type_e>(
            static_cast<int>(static_cast<ov::element::Type_t>(cpp_tensor.get_element_type()))
        );

        ov_status_e status = ov_tensor_create_from_host_ptr(
            c_type, 
            shape, 
            cpp_tensor.data(), 
            video_tensor
        );
        
        free(shape.dims); 
        
        if (status != ov_status_e::OK) {
            return status;
        }

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

// -------------------------------------------------------------------------
// Configuration Getters & Setters
// -------------------------------------------------------------------------

ov_status_e ov_genai_text2video_pipeline_get_generation_config(const ov_genai_text2video_pipeline* pipe,
                                                               ov_genai_video_generation_config** config) {
    if (!pipe || !(pipe->object) || !config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_video_generation_config> _config = std::make_unique<ov_genai_video_generation_config>();
        // Fetch the video generation config from the C++ pipeline and wrap it
        _config->object = std::make_shared<ov::genai::VideoGenerationConfig>(pipe->object->get_generation_config());
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2video_pipeline_set_generation_config(ov_genai_text2video_pipeline* pipe,
                                                               ov_genai_video_generation_config* config) {
    if (!pipe || !(pipe->object) || !config || !(config->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        // Pass the unwrapped C++ config back to the pipeline
        pipe->object->set_generation_config(*(config->object));
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}