// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2video_pipeline.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/genai/video_generation/text2video_pipeline.hpp"

struct ov_genai_text2video_pipeline {
    std::unique_ptr<ov::genai::Text2VideoPipeline> pipe;
};

extern "C" {

int ov_genai_text2video_pipeline_create(
    const char* model_path,
    const char* device,
    ov_genai_text2video_pipeline** pipeline) {
    if (!pipeline || !model_path || !device) {
        return 1;
    }
    try {
        auto pipe_cpp = std::make_unique<ov::genai::Text2VideoPipeline>(std::string(model_path), std::string(device));
        *pipeline = new ov_genai_text2video_pipeline{std::move(pipe_cpp)};
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[DEBUG] OpenVINO Exception: " << e.what() << "\n" << std::endl;
        return 2;
    } catch (...) {
        return 2;
    }
}

void ov_genai_text2video_pipeline_destroy(ov_genai_text2video_pipeline* pipeline) {
    if (pipeline) {
        delete pipeline;
    }
}

void ov_genai_text2video_pipeline_generate(
    ov_genai_text2video_pipeline* pipeline,
    const char* prompt,
    const char* negative_prompt,
    const ov_genai_video_generation_config* config,
    text2video_custom_tensor* output_tensor) {
    if (!pipeline || !prompt || !output_tensor) {
        return;
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
        }

        if (negative_prompt) {
            properties[ov::genai::negative_prompt.name()] = std::string(negative_prompt);
        }

        auto result = pipeline->pipe->generate(std::string(prompt), properties);

        ov::Tensor cpp_tensor = result.video;
        const ov::Shape& shape = cpp_tensor.get_shape();

        if (shape.size() >= 3) {
            output_tensor->height = shape[shape.size() - 3];
            output_tensor->width = shape[shape.size() - 2];
        }

        output_tensor->size = cpp_tensor.get_byte_size();
        output_tensor->data = std::malloc(output_tensor->size);
        if (output_tensor->data) {
            std::memcpy(output_tensor->data, cpp_tensor.data(), output_tensor->size);
        }

    } catch (...) {
        if (output_tensor) {
            output_tensor->data = nullptr;
            output_tensor->size = 0;
        }
    }
}

void ov_genai_text2video_free_tensor(text2video_custom_tensor* tensor) {
    if (tensor && tensor->data) {
        std::free(tensor->data);
        tensor->data = nullptr;
        tensor->size = 0;
    }
}

}  // extern "C"