// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/text2video_pipeline.h"

#include <cstdlib>
#include <cstring>
#include <memory>

#include "openvino/genai/video_generation/text2video_pipeline.hpp"

struct ov_genai_text2video_pipeline {
    std::unique_ptr<ov::genai::Text2VideoPipeline> pipe;
};

extern "C" {

int ov_genai_text2video_pipeline_create(const char* model_path,
                                        const char* device,
                                        ov_genai_text2video_pipeline** pipeline) {
    if (!pipeline || !model_path || !device) {
        return 1;
    }

    try {
        auto pipe_cpp = std::make_unique<ov::genai::Text2VideoPipeline>(model_path, device);
        *pipeline = new ov_genai_text2video_pipeline{std::move(pipe_cpp)};
        return 0;
    } catch (...) {
        return 2;
    }
}

void ov_genai_text2video_pipeline_destroy(ov_genai_text2video_pipeline* pipeline) {
    if (pipeline) {
        delete pipeline;
    }
}

void ov_genai_text2video_pipeline_generate(ov_genai_text2video_pipeline* pipeline,
                                           const char* prompt,
                                           text2video_custom_tensor* output_tensor) {
    if (!pipeline || !prompt || !output_tensor) {
        return;
    }

    try {
        auto result = pipeline->pipe->generate(prompt);
        ov::Tensor cpp_tensor = result.video;

        output_tensor->height = cpp_tensor.get_shape()[2];
        output_tensor->width = cpp_tensor.get_shape()[3];
        output_tensor->size = cpp_tensor.get_byte_size();

        output_tensor->data = malloc(output_tensor->size);
        if (output_tensor->data) {
            std::memcpy(output_tensor->data, cpp_tensor.data(), output_tensor->size);
        }
    } catch (...) {
        output_tensor->data = nullptr;
    }
}

void ov_genai_text2video_free_tensor(text2video_custom_tensor* tensor) {
    if (tensor && tensor->data) {
        free(tensor->data);
        tensor->data = nullptr;
    }
}

}  // extern "C"