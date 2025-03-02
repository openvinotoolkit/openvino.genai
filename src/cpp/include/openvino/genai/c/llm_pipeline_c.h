// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for  ov::genai::LLMPipeline class.
 *
 * @file llm_pipeline_c.h
 */

#pragma once
#include "generation_config_c.h"
#include "perf_metrics_c.h"

/**
 * @struct ov_genai_decoded_results
 * @brief type define ov_genai_decoded_results from ov_genai_decoded_results_opaque
 */
typedef struct ov_genai_decoded_results_opaque ov_genai_decoded_results;

/**
 * @brief Create DecodedResults
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_decoded_results* ov_genai_decoded_results_create();

/**
 * @brief Release the memory allocated by ov_genai_decoded_results.
 * @param model A pointer to the ov_genai_decoded_results to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_decoded_results_free(ov_genai_decoded_results* results);

/**
 * @brief Get performance metrics from ov_genai_decoded_results.
 * @param results A pointer to the ov_genai_decoded_results.
 * @param metrics A pointer to the ov_genai_perf_metrics.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_decoded_results_get_perf_metrics(ov_genai_decoded_results* results,
                                                                        ov_genai_perf_metrics** metrics);

/**
 * @brief Get string result from ov_genai_decoded_results.
 * @param results A pointer to the ov_genai_decoded_results.
 * @param output A pointer to the output string buffer.
 * @param max_size The maximum size of the output buffer.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_decoded_results_get_string(ov_genai_decoded_results* results,
                                                                  char* output,
                                                                  int max_size);

/**
 * @struct ov_genai_llm_pipeline
 * @brief type define ov_genai_llm_pipeline from ov_genai_llm_pipeline_opaque
 */
typedef struct ov_genai_llm_pipeline_opaque ov_genai_llm_pipeline;

/**
 * @brief Construct ov_genai_llm_pipeline.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_llm_pipeline* ov_genai_llm_pipeline_create(const char* models_path,
                                                                             const char* device);

/**
 * @brief Release the memory allocated by ov_genai_llm_pipeline.
 * @param model A pointer to the ov_genai_llm_pipeline.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_llm_pipeline_free(ov_genai_llm_pipeline* pipe);

/**
 * @brief Generate text by ov_genai_llm_pipeline.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @param inputs A pointer to the input string.
 * @param output A pointer to the output string buffer.
 * @param max_size The maximum size of the output buffer.
 * @param config A pointer to the ov_genai_generation_config, the pointer can be NULL.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_llm_pipeline_generate(ov_genai_llm_pipeline* handle,
                                                             const char* inputs,
                                                             char* output,
                                                             int max_size,
                                                             ov_genai_generation_config* config);

/*@brief Generate text by ov_genai_llm_pipeline with Streamer.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @param inputs A pointer to the input string.
 * @param output A pointer to the output string buffer.
 * @param max_size The maximum size of the output buffer.
 * @param config A pointer to the ov_genai_generation_config, the pointer can be NULL.
 * @param buffer A pointer to the stream buffer.
 * @param buffer_size The size of the stream buffer.
 * @param buffer_pos A pointer to the stream buffer position.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_llm_pipeline_generate_stream(ov_genai_llm_pipeline* pipe,
                                                                    const char* inputs,
                                                                    char* output,
                                                                    int max_size,
                                                                    ov_genai_generation_config* config,
                                                                    char* buffer,
                                                                    const int buffer_size,
                                                                    int* buffer_pos);

/**
 * @brief Generate text by ov_genai_llm_pipeline and return ov_genai_decoded_results.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @param inputs A pointer to the input string.
 * @pram config A pointer to the ov_genai_generation_config, the pointer can be NULL.
 * @return ov_genai_decoded_results A pointer to the ov_genai_decoded_results.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_decoded_results* ov_genai_llm_pipeline_generate_decode_results(
    ov_genai_llm_pipeline* handle,
    const char* inputs,
    ov_genai_generation_config* config);
/**
 * @brief Start chat with keeping history in kv cache.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_llm_pipeline_start_chat(ov_genai_llm_pipeline* pipe);

/**
 * @brief Finish chat and clear kv cache.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_llm_pipeline_finish_chat(ov_genai_llm_pipeline* pipe);

/**
 * @brief Get the GenerationConfig from ov_genai_llm_pipeline.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @return ov_genai_generation_config A pointer to the ov_genai_generation_config.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_generation_config* ov_genai_llm_pipeline_get_generation_config(
    ov_genai_llm_pipeline* pipe);

/**
 * @brief Set the GenerationConfig to ov_genai_llm_pipeline.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @param config A pointer to the ov_genai_generation_config.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_llm_pipeline_set_generation_config(ov_genai_llm_pipeline* pipe,
                                                            ov_genai_generation_config* config);
