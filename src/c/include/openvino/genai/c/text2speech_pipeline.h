// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for ov::genai::Text2SpeechPipeline class.
 *
 * @file text2speech_pipeline.h
 */

#pragma once

#include "openvino/c/ov_tensor.h"
#include "openvino/genai/c/perf_metrics.h"
#include "openvino/genai/c/speech_generation_config.h"

/**
 * @struct ov_genai_text2speech_decoded_results
 * @brief type define ov_genai_text2speech_decoded_results from ov_genai_text2speech_decoded_results_opaque
 */
typedef struct ov_genai_text2speech_decoded_results_opaque ov_genai_text2speech_decoded_results;

/**
 * @struct ov_genai_text2speech_pipeline
 * @brief type define ov_genai_text2speech_pipeline from ov_genai_text2speech_pipeline_opaque
 */
typedef struct ov_genai_text2speech_pipeline_opaque ov_genai_text2speech_pipeline;

/**
 * @brief Create Text2SpeechDecodedResults
 * @param results A pointer to the newly created ov_genai_text2speech_decoded_results.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_create(ov_genai_text2speech_decoded_results** results);

/**
 * @brief Release the memory allocated by ov_genai_text2speech_decoded_results.
 * @param results A pointer to the ov_genai_text2speech_decoded_results to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2speech_decoded_results_free(ov_genai_text2speech_decoded_results* results);

/**
 * @brief Get performance metrics from ov_genai_text2speech_decoded_results.
 * @param results A pointer to the ov_genai_text2speech_decoded_results instance.
 * @param metrics A pointer to the newly created ov_genai_perf_metrics.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_perf_metrics(const ov_genai_text2speech_decoded_results* results,
                                                      ov_genai_perf_metrics** metrics);

/**
 * @brief Get number of generated waveform tensors from ov_genai_text2speech_decoded_results.
 * @param results A pointer to the ov_genai_text2speech_decoded_results instance.
 * @param count A pointer to the number of generated waveform tensors.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_speeches_count(const ov_genai_text2speech_decoded_results* results,
                                                        size_t* count);

/**
 * @brief Get generated waveform tensor at specific index from ov_genai_text2speech_decoded_results.
 * @param results A pointer to the ov_genai_text2speech_decoded_results instance.
 * @param index The index of the waveform tensor to retrieve.
 * @param speech A pointer to the retrieved ov_tensor_t.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_speech_at(const ov_genai_text2speech_decoded_results* results,
                                                   size_t index,
                                                   ov_tensor_t** speech);

/**
 * @brief Construct ov_genai_text2speech_pipeline.
 *
 * Initializes a ov_genai_text2speech_pipeline instance from the specified model directory and device. Optional property
 * parameters can be passed as key-value pairs.
 *
 * @param models_path Path to the directory containing the model files.
 * @param device Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param pipe A pointer to the newly created ov_genai_text2speech_pipeline.
 * @param ... property parameters: Optional pack of key/value pairs. Each property is specified as a key (char*)
 *            followed by a value whose C type depends on the property key.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2speech_pipeline_create(const char* models_path,
                                                                          const char* device,
                                                                          const size_t property_args_size,
                                                                          ov_genai_text2speech_pipeline** pipe,
                                                                          ...);

/**
 * @brief Release the memory allocated by ov_genai_text2speech_pipeline.
 * @param pipe A pointer to the ov_genai_text2speech_pipeline to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2speech_pipeline_free(ov_genai_text2speech_pipeline* pipe);

/**
 * @brief Generates speeches based on input texts.
 * @param pipe A pointer to the ov_genai_text2speech_pipeline instance.
 * @param texts An array of input text strings.
 * @param texts_size The number of input text strings.
 * @param speaker_embedding Optional speaker embedding tensor. Can be NULL.
 * @param property_args_size How many properties args will be passed.
 * @param results A pointer to the ov_genai_text2speech_decoded_results to retrieve the results.
 * @param ... property parameter: Optional pack of pairs.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_generate(ov_genai_text2speech_pipeline* pipe,
                                       const char** texts,
                                       size_t texts_size,
                                       const ov_tensor_t* speaker_embedding,
                                       const size_t property_args_size,
                                       ov_genai_text2speech_decoded_results** results,
                                       ...);

/**
 * @brief Get the SpeechGenerationConfig from ov_genai_text2speech_pipeline.
 * @param pipe A pointer to the ov_genai_text2speech_pipeline instance.
 * @param config A pointer to the newly created ov_genai_speech_generation_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_get_generation_config(const ov_genai_text2speech_pipeline* pipe,
                                                    ov_genai_speech_generation_config** config);

/**
 * @brief Set the SpeechGenerationConfig to ov_genai_text2speech_pipeline.
 * @param pipe A pointer to the ov_genai_text2speech_pipeline instance.
 * @param config A pointer to the ov_genai_speech_generation_config instance.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_set_generation_config(ov_genai_text2speech_pipeline* pipe,
                                                    const ov_genai_speech_generation_config* config);
