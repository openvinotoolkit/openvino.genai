// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/c/openvino.h"
#include "openvino/genai/c/visibility.h" // Use the standard visibility header
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// #include "openvino/genai/c/visibility.h"
#ifdef __cplusplus
extern "C" {
#endif

// --- ADDED EXPORT MACRO DEFINITIONS ---
#ifndef OPENVINO_GENAI_C_EXPORTS
#define OPENVINO_GENAI_C_EXPORTS
#endif
// --------------------------------------

typedef struct ov_genai_video_generation_config_opaque ov_genai_video_generation_config;

OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_video_generation_config_create(ov_genai_video_generation_config** config);
OPENVINO_GENAI_C_EXPORTS void ov_genai_video_generation_config_free(ov_genai_video_generation_config* config);
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_video_generation_config_set_width(ov_genai_video_generation_config* config, size_t width);
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_video_generation_config_set_height(ov_genai_video_generation_config* config, size_t height);
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_video_generation_config_set_num_frames(ov_genai_video_generation_config* config, size_t num_frames);
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_video_generation_config_set_num_inference_steps(ov_genai_video_generation_config* config, size_t num_inference_steps);

#ifdef __cplusplus
}
#endif