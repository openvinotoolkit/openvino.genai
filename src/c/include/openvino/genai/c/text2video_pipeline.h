// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for ov::genai::Text2VideoPipeline class.
 *
 * @file text2video_pipeline.h
 */

#pragma once

#include "openvino/c/openvino.h"
#include "openvino/genai/c/video_generation_config.h"
// #include "openvino/genai/c/generation_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OPENVINO_GENAI_C_EXPORTS
#define OPENVINO_GENAI_C_EXPORTS
#endif

/**
 * @struct ov_genai_text2video_pipeline
 * @brief type define ov_genai_text2video_pipeline from ov_genai_text2video_pipeline_opaque
 */
typedef struct ov_genai_text2video_pipeline_opaque ov_genai_text2video_pipeline;

/**
 * @brief Construct ov_genai_text2video_pipeline.
 *
 * Initializes a ov_genai_text2video_pipeline instance from the specified model directory and device. 
 * Optional property parameters can be passed as key-value pairs.
 *
 * @param models_path Path to the directory containing the model files.
 * @param device Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param pipe A pointer to the newly created ov_genai_text2video_pipeline.
 * @param ... property parameter: Optional pack of pairs: <char* property_key, char* property_value>
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_create(
    const char* models_path,
    const char* device,
    const size_t property_args_size,
    ov_genai_text2video_pipeline** pipe,
    ...);

/**
 * @brief Release the memory allocated by ov_genai_text2video_pipeline.
 * @param pipe A pointer to the ov_genai_text2video_pipeline to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2video_pipeline_free(ov_genai_text2video_pipeline* pipe);

/**
 * @brief Generate video results by ov_genai_text2video_pipeline
 * @param pipe A pointer to the ov_genai_text2video_pipeline instance.
 * @param prompt A pointer to the input text prompt string.
 * @param config A pointer to the ov_genai_video_generation_config, the pointer can be NULL.
 * @param video_tensor A pointer to an ov_tensor_t pointer, which retrieves the generated video frames.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_generate(
    ov_genai_text2video_pipeline* pipe,
    const char* prompt,
    const ov_genai_video_generation_config* config,
    ov_tensor_t** video_tensor);

/**
 * @brief Get the VideoGenerationConfig from ov_genai_text2video_pipeline.
 * @param pipe A pointer to the ov_genai_text2video_pipeline instance.
 * @param config A pointer to the newly created ov_genai_video_generation_config.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_get_generation_config(
    const ov_genai_text2video_pipeline* pipe,
    ov_genai_video_generation_config** config);

/**
 * @brief Set the VideoGenerationConfig to ov_genai_text2video_pipeline.
 * @param pipe A pointer to the ov_genai_text2video_pipeline instance.
 * @param config A pointer to the ov_genai_video_generation_config instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_set_generation_config(
    ov_genai_text2video_pipeline* pipe,
    ov_genai_video_generation_config* config);

#ifdef __cplusplus
}
#endif