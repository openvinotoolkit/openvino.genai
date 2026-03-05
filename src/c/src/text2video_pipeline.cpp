// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2video_pipeline.h"
#include "openvino/genai/video_generation/text2video_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"

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

        // Convert the VideoGenerationConfig into the AnyMap properties expected by C++
        if (config && config->object) {
            properties.insert(ov::genai::generation_config(*(config->object)));
        }

        // Call generate, which returns a VideoGenerationResult object
        auto result = pipe->object->generate(prompt_str, properties);

        // Extract the first actual tensor from the result struct
        ov::Tensor cpp_tensor = result.video;

        // Bridge: Convert C++ ov::Tensor to C ov_tensor_t
        ov_shape_t shape;
        shape.rank = cpp_tensor.get_shape().size();
        shape.dims = new int64_t[shape.rank];
        for (size_t i = 0; i < shape.rank; ++i) {
            shape.dims[i] = cpp_tensor.get_shape()[i];
        }
        
        // Explicitly cast the C++ type to the C enum type
        ov_element_type_e c_type = static_cast<ov_element_type_e>(
            static_cast<int>(static_cast<ov::element::Type_t>(cpp_tensor.get_element_type()))
        );

        ov_status_e status = ov_tensor_create_from_host_ptr(
            c_type, 
            shape, 
            cpp_tensor.data(), 
            video_tensor
        );
        
        delete[] shape.dims; // Clean up temporary array
        
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
                                                               ov_genai_generation_config** config) {
    if (!pipe || !(pipe->object) || !config) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_generation_config> _config = std::make_unique<ov_genai_generation_config>();
        // Fetch the unified generation config from the C++ pipeline and wrap it
        _config->object = std::make_shared<ov::genai::GenerationConfig>(pipe->object->get_generation_config());
        *config = _config.release();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_text2video_pipeline_set_generation_config(ov_genai_text2video_pipeline* pipe,
                                                               ov_genai_generation_config* config) {
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