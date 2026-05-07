// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for ov::genai::Text2VideoPipeline class.
 *
 * @file text2video_pipeline.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_tensor.h"
#include "openvino/genai/c/visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct ov_genai_video_generation_config
 * @brief Configuration parameters for video generation.
 */
typedef struct {
    size_t width;       //!< Width of the generated video frames
    size_t height;      //!< Height of the generated video frames
    size_t num_frames;  //!< Total number of frames to generate
    float frame_rate;   //!< Frame rate of the generated video
} ov_genai_video_generation_config;

/**
 * @struct ov_genai_text2video_pipeline
 * @brief type define ov_genai_text2video_pipeline from ov_genai_text2video_pipeline_opaque
 */
typedef struct ov_genai_text2video_pipeline_opaque ov_genai_text2video_pipeline;

/**
 * @brief Construct ov_genai_text2video_pipeline.
 *
 * Initializes a ov_genai_text2video_pipeline instance from the specified model directory and device.
 *
 * @param models_path Path to the directory containing the model files.
 * @param device Name of a device to load a model to.
 * @param pipeline A pointer to the newly created ov_genai_text2video_pipeline.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_create(const char* models_path,
                                                                         const char* device,
                                                                         ov_genai_text2video_pipeline** pipeline);

/**
 * @brief Release the memory allocated by ov_genai_text2video_pipeline.
 * @param pipeline A pointer to the ov_genai_text2video_pipeline to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2video_pipeline_free(ov_genai_text2video_pipeline* pipeline);

/**
 * @brief Generate a video based on text inputs.
 *
 * @param pipeline A pointer to the ov_genai_text2video_pipeline instance.
 * @param prompt A pointer to the input text string (positive prompt).
 * @param negative_prompt A pointer to the negative input text string.
 * @param config A pointer to the ov_genai_video_generation_config, the pointer can be NULL.
 * @param output_tensor A pointer to the ov_tensor_t pointer to retrieve the generated video tensor.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2video_pipeline_generate(ov_genai_text2video_pipeline* pipeline,
                                      const char* prompt,
                                      const char* negative_prompt,
                                      const ov_genai_video_generation_config* config,
                                      ov_tensor_t** output_tensor);

#ifdef __cplusplus
}
#endif