// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for ov::genai::Text2VideoPipeline class.
 *
 * @file text2video_pipeline.h
 */

#pragma once
#include "openvino/c/openvino.h"
#include "openvino/genai/c/visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// We need an enum for status if it's not included
typedef struct ov_genai_video_generation_config_opaque ov_genai_video_generation_config;
typedef struct ov_genai_text2video_pipeline_opaque ov_genai_text2video_pipeline;
typedef struct ov_genai_video_generation_perf_metrics_opaque ov_genai_video_generation_perf_metrics;
typedef struct ov_genai_video_generation_result_opaque ov_genai_video_generation_result;

/**
 * @brief Initialize text to video generation pipeline from a folder with models.
 * @param models_path A models path to read models and config files from
 * @param pipe A pointer to the newly created ov_genai_text2video_pipeline.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_create(const char* models_path, ov_genai_text2video_pipeline** pipe);

/**
 * @brief Initializes text to video pipelines from a folder with models and performs compilation after it.
 * @param models_path A models path to read models and config files from
 * @param device A single device used for all models
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param pipe A pointer to the newly created ov_genai_text2video_pipeline.
 * @param ... property parameter: Optional pack of pairs: <char* property_key, char* property_value>
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_create_with_device(const char* models_path, 
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
 * @brief Compiles video generation pipeline for a given device
 * @param pipe A pointer to the ov_genai_text2video_pipeline.
 * @param device A device to compile models with
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param ... property parameter: Optional pack of pairs: <char* property_key, char* property_value>
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_compile(ov_genai_text2video_pipeline* pipe,
                                                                          const char* device,
                                                                          const size_t property_args_size,
                                                                          ...);

/**
 * @brief Generates video(s) based on prompt and other video generation parameters
 * @param pipe A pointer to the ov_genai_text2video_pipeline instance.
 * @param positive_prompt Prompt to generate video(s) from
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param result A pointer to the ov_genai_video_generation_result, which retrieves the results of the generation.
 * @param ... property parameter: Optional pack of pairs: <char* property_key, char* property_value>
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_generate(ov_genai_text2video_pipeline* pipe,
                                                                           const char* positive_prompt,
                                                                           const size_t property_args_size,
                                                                           ov_genai_video_generation_result** result,
                                                                           ...);


/**
 * @brief Release the memory allocated by ov_genai_video_generation_result.
 * @param result A pointer to the ov_genai_video_generation_result to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_video_generation_result_free(ov_genai_video_generation_result* result);

/**
 * @brief Get the Video Tensor from ov_genai_video_generation_result.
 * @param result A pointer to the ov_genai_video_generation_result instance.
 * @param tensor A pointer to the underlying ov_tensor (part of openvino/c/openvino.h).
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_video_generation_result_get_video(const ov_genai_video_generation_result* result,
                                                                                ov_tensor_t** tensor);

/**
 * @brief Get the Performance Metrics from ov_genai_video_generation_result.
 * @param result A pointer to the ov_genai_video_generation_result instance.
 * @param metrics A pointer to the newly created ov_genai_video_generation_perf_metrics.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_video_generation_result_get_perf_metrics(const ov_genai_video_generation_result* result,
                                                                                       ov_genai_video_generation_perf_metrics** metrics);

/**
 * @brief Release the memory allocated by ov_genai_video_generation_perf_metrics.
 * @param metrics A pointer to the ov_genai_video_generation_perf_metrics to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_video_generation_perf_metrics_free(ov_genai_video_generation_perf_metrics* metrics);

/**
 * @brief Get the GenerationConfig from ov_genai_text2video_pipeline.
 * @param pipe A pointer to the ov_genai_text2video_pipeline instance.
 * @param config A pointer to the newly created ov_genai_video_generation_config.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_get_generation_config(const ov_genai_text2video_pipeline* pipe,
                                                                                        ov_genai_video_generation_config** config);

/**
 * @brief Set the GenerationConfig to ov_genai_text2video_pipeline.
 * @param pipe A pointer to the ov_genai_text2video_pipeline instance.
 * @param config A pointer to the ov_genai_video_generation_config instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2video_pipeline_set_generation_config(ov_genai_text2video_pipeline* pipe,
                                                                                        ov_genai_video_generation_config* config);

/**
 * @brief Initialize a ov_genai_video_generation_config.
 * @param config A pointer to the newly created ov_genai_video_generation_config.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_video_generation_config_create(ov_genai_video_generation_config** config);

/**
 * @brief Release the memory allocated by ov_genai_video_generation_config.
 * @param config A pointer to the ov_genai_video_generation_config to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_video_generation_config_free(ov_genai_video_generation_config* config);

#ifdef __cplusplus
}
#endif