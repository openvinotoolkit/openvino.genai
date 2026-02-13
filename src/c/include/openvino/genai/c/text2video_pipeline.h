// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stddef.h> 

#ifdef __cplusplus
extern "C" {
#endif

// FIX: Define the macro to be empty for static compilation
#define OV_GENAI_C_API

typedef struct {
    void* data;      // Raw pixel data
    size_t size;     // Total bytes
    size_t height;
    size_t width;
} text2video_custom_tensor;

typedef struct ov_genai_text2video_pipeline ov_genai_text2video_pipeline;

OV_GENAI_C_API int ov_genai_text2video_pipeline_create(
    const char* model_path,
    const char* device,
    ov_genai_text2video_pipeline** pipeline);

OV_GENAI_C_API void ov_genai_text2video_pipeline_destroy(
    ov_genai_text2video_pipeline* pipeline);

OV_GENAI_C_API void ov_genai_text2video_pipeline_generate(
    ov_genai_text2video_pipeline* pipeline,
    const char* prompt,
    text2video_custom_tensor* output_tensor
);

OV_GENAI_C_API void ov_genai_text2video_free_tensor(
    text2video_custom_tensor* tensor);

#ifdef __cplusplus
}
#endif