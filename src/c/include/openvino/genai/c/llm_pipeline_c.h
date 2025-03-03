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
 * @param results A pointer to the ov_genai_decoded_results.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_decoded_results_create(ov_genai_decoded_results** results);

/**
 * @brief Release the memory allocated by ov_genai_decoded_results.
 * @param model A pointer to the ov_genai_decoded_results to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_decoded_results_free(ov_genai_decoded_results* results);

/**
 * @brief Get performance metrics from ov_genai_decoded_results.
 * @param results A pointer to the ov_genai_decoded_results.
 * @param metrics A pointer to the ov_genai_perf_metrics.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_decoded_results_get_perf_metrics(const ov_genai_decoded_results* results,
                                                                               ov_genai_perf_metrics** metrics);

/**
 * @brief Release the memory allocated by ov_genai_perf_metrics.
 * @param model A pointer to the ov_genai_perf_metrics to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics);

/**
 * @brief Get string result from ov_genai_decoded_results.
 * @param results A pointer to the ov_genai_decoded_results.
 * @param output A pointer to the output string buffer.
 * @param max_size The maximum size of the output buffer.
 * @return ov_status_e A status code, return OK(0) if successful. Returns OUT_OF_BOUNDS if output_size is insufficient
 * to store the result.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_decoded_results_get_string(const ov_genai_decoded_results* results,
                                                                         char* output,
                                                                         size_t max_size);

/**
 * @struct ov_genai_llm_pipeline
 * @brief type define ov_genai_llm_pipeline from ov_genai_llm_pipeline_opaque
 * @return ov_status_e A status code, return OK(0) if successful.
 */
typedef struct ov_genai_llm_pipeline_opaque ov_genai_llm_pipeline;

/**
 * @brief Construct ov_genai_llm_pipeline.
 * @param models_path A path to the directory containing the model files.
 * @param device A device name.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_create(const char* models_path,
                                                                  const char* device,
                                                                  ov_genai_llm_pipeline** pipe);

// TODO: Add 'const ov::AnyMap& properties' as an input argument when creating ov_genai_llm_pipeline.

/**
 * @brief Release the memory allocated by ov_genai_llm_pipeline.
 * @param model A pointer to the ov_genai_llm_pipeline.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_llm_pipeline_free(ov_genai_llm_pipeline* pipe);

/**
 * @struct stream_callback
 * @brief Completion callback definition about the function
 */
typedef struct {
    void(OPENVINO_C_API_CALLBACK* callback_func)(const char*);  //!< The callback func
} stream_callback;

/**
 * @brief Generate text by ov_genai_llm_pipeline.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @param inputs A pointer to the input string.
 * @param config A pointer to the ov_genai_generation_config, This is optional, the pointer can be NULL.
 * @param streamer A pointer to the stream callback. This is optional; set to NULL if no callback is needed.
 * @param output A pointer to the output string buffer.
 * @param output_max_size The maximum size of the output buffer.
 * @return ov_status_e A status code, return OK(0) if successful. Returns OUT_OF_BOUNDS if output_size is insufficient
 * to store the result.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_generate(ov_genai_llm_pipeline* pipe,
                                                                    const char* inputs,
                                                                    const ov_genai_generation_config* config,
                                                                    const stream_callback* streamer,
                                                                    char* output,
                                                                    size_t output_max_size);
/**
 * @brief Generate text by ov_genai_llm_pipeline and return ov_genai_decoded_results.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @param inputs A pointer to the input string.
 * @param config A pointer to the ov_genai_generation_config, the pointer can be NULL.
 * @param streamer A pointer to the stream callback. This is optional; set to NULL if no callback is needed.
 * @param ov_genai_decoded_results A pointer to the ov_genai_decoded_results.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_llm_pipeline_generate_decode_results(ov_genai_llm_pipeline* pipe,
                                              const char* inputs,
                                              const ov_genai_generation_config* config,
                                              const stream_callback* streamer,
                                              ov_genai_decoded_results** results);
/**
 * @brief Start chat with keeping history in kv cache.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_start_chat(ov_genai_llm_pipeline* pipe);

/**
 * @brief Finish chat and clear kv cache.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_finish_chat(ov_genai_llm_pipeline* pipe);

/**
 * @brief Get the GenerationConfig from ov_genai_llm_pipeline.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @param ov_genai_generation_config A pointer to the ov_genai_generation_config.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_get_generation_config(const ov_genai_llm_pipeline* pipe,
                                                                                 ov_genai_generation_config** config);

/**
 * @brief Set the GenerationConfig to ov_genai_llm_pipeline.
 * @param pipe A pointer to the ov_genai_llm_pipeline.
 * @param config A pointer to the ov_genai_generation_config.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_llm_pipeline_set_generation_config(ov_genai_llm_pipeline* pipe,
                                                                                 ov_genai_generation_config* config);
