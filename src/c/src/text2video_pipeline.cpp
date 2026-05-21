// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2video_pipeline.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/genai/video_generation/text2video_pipeline.hpp"

// OpenVINO Core C API for Tensor and Shape management
#include "openvino/c/ov_shape.h"
#include "openvino/c/ov_tensor.h"

struct ov_genai_text2video_pipeline_opaque {
    std::unique_ptr<ov::genai::Text2VideoPipeline> pipe;
};

extern "C" {

ov_status_e ov_genai_text2video_pipeline_create(const char* model_path,
                                                const char* device,
                                                ov_genai_text2video_pipeline** pipeline) {
    if (!pipeline || !model_path || !device) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto pipe_cpp = std::make_unique<ov::genai::Text2VideoPipeline>(std::string(model_path), std::string(device));
        *pipeline = new ov_genai_text2video_pipeline_opaque{std::move(pipe_cpp)};
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

void ov_genai_text2video_pipeline_free(ov_genai_text2video_pipeline* pipeline) {
    if (pipeline) {
        delete pipeline;
    }
}

ov_status_e ov_genai_text2video_pipeline_generate(ov_genai_text2video_pipeline* pipeline,
                                                  const char* prompt,
                                                  const char* negative_prompt,
                                                  const ov_genai_video_generation_config* config,
                                                  ov_tensor_t** output_tensor) {
    if (!pipeline || !prompt || !output_tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap properties;

        if (config) {
            if (config->width > 0) {
                properties[ov::genai::width.name()] = config->width;
            }
            if (config->height > 0) {
                properties[ov::genai::height.name()] = config->height;
            }
            if (config->num_frames > 0) {
                properties[ov::genai::num_frames.name()] = config->num_frames;
            }
            if (config->frame_rate > 0.0f) {
                properties[ov::genai::frame_rate.name()] = config->frame_rate;
            }
        }

        if (negative_prompt) {
            properties[ov::genai::negative_prompt.name()] = std::string(negative_prompt);
        }

        // Run the C++ inference
        auto result = pipeline->pipe->generate(std::string(prompt), properties);
        ov::Tensor cpp_tensor = result.video;
        const ov::Shape& shape = cpp_tensor.get_shape();

        // Convert C++ ov::Shape to C ov_shape_t
        ov_shape_t c_shape;
        ov_shape_create(shape.size(), &c_shape);
        for (size_t i = 0; i < shape.size(); ++i) {
            c_shape.dims[i] = shape[i];
        }

        // Create the OpenVINO C Tensor (Video outputs are typically U8)
        ov_status_e status = ov_tensor_create(U8, c_shape, output_tensor);

        // Deep copy the data so it safely escapes the C++ scope
        if (status == ov_status_e::OK && *output_tensor) {
            void* c_data = nullptr;
            ov_tensor_data(*output_tensor, &c_data);
            if (c_data) {
                std::memcpy(c_data, cpp_tensor.data(), cpp_tensor.get_byte_size());
            }
        }

        // Clean up the temporary C shape
        ov_shape_free(&c_shape);

        return status;

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
}

}  // extern "C"