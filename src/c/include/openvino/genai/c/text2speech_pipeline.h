// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/c/ov_shape.h"
#include "openvino/c/ov_tensor.h"
#include "openvino/genai/c/perf_metrics.h"
#include "openvino/genai/c/speech_generation_config.h"

/** @brief Opaque text-to-speech pipeline. */
typedef struct ov_genai_text2speech_pipeline_opaque ov_genai_text2speech_pipeline;
/** @brief Opaque collection of decoded speech waveforms and performance metrics. */
typedef struct ov_genai_text2speech_decoded_results_opaque ov_genai_text2speech_decoded_results;

/**
 * @brief Create a text-to-speech pipeline from an exported model directory.
 * @param[in] models_path Directory containing the exported speech model.
 * @param[in] device OpenVINO device name, such as "CPU" or "GPU".
 * @param[in] property_args_size Number of variadic key/value arguments; must be even. Zero uses defaults.
 * @param[out] pipeline Receives a new pipeline. Release it with ov_genai_text2speech_pipeline_free.
 * @param[in] ... OpenVINO property key/value arguments, when property_args_size is nonzero.
 * @return OK on success, INVALID_C_PARAM for NULL required arguments or odd property_args_size,
 *         or UNKNOW_EXCEPTION if construction fails.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_text2speech_pipeline_create(const char* models_path,
                                                                          const char* device,
                                                                          size_t property_args_size,
                                                                          ov_genai_text2speech_pipeline** pipeline,
                                                                          ...);
/**
 * @brief Release a pipeline created by this API. NULL is allowed.
 * @param[in] pipeline Pipeline to release.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2speech_pipeline_free(ov_genai_text2speech_pipeline* pipeline);

/**
 * @brief Generate one waveform for a text prompt.
 * @param[in] pipeline Pipeline to use.
 * @param[in] text NUL-terminated text prompt.
 * @param[in] speaker_embedding Optional float32 speaker tensor; NULL uses a model's default voice.
 *                              Kokoro requires a speaker tensor. The caller retains ownership.
 * @param[in] config Optional speech generation configuration for this call; NULL uses the pipeline's
 *                   current configuration. The configuration is copied; the caller retains ownership.
 * @param[out] results Receives a new result. Release it with ov_genai_text2speech_decoded_results_free.
 * @return OK on success, INVALID_C_PARAM for invalid arguments or a non-float32 speaker tensor,
 *         or UNKNOW_EXCEPTION if generation fails.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_generate(ov_genai_text2speech_pipeline* pipeline,
                                       const char* text,
                                       const ov_tensor_t* speaker_embedding,
                                       const ov_genai_speech_generation_config* config,
                                       ov_genai_text2speech_decoded_results** results);

/**
 * @brief Generate one waveform per text with a shared speaker embedding.
 * @param[in] pipeline Pipeline to use.
 * @param[in] texts Array of count non-NULL, NUL-terminated text prompts.
 * @param[in] count Number of prompts; must be greater than zero.
 * @param[in] speaker_embedding Optional float32 speaker tensor shared by all prompts; NULL uses a model's
 *                              default voice. Kokoro requires a speaker tensor. The caller retains ownership.
 * @param[in] config Optional speech generation configuration for this call; NULL uses the pipeline's
 *                   current configuration. The configuration is copied; the caller retains ownership.
 * @param[out] results Receives a new result. Release it with ov_genai_text2speech_decoded_results_free.
 * @return OK on success, INVALID_C_PARAM for invalid arguments or a non-float32 speaker tensor,
 *         or UNKNOW_EXCEPTION if generation fails.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_generate_batch(ov_genai_text2speech_pipeline* pipeline,
                                             const char* const* texts,
                                             size_t count,
                                             const ov_tensor_t* speaker_embedding,
                                             const ov_genai_speech_generation_config* config,
                                             ov_genai_text2speech_decoded_results** results);

/**
 * @brief Copy the pipeline's current speech generation configuration.
 * @param[in] pipeline Pipeline to read.
 * @param[out] config Receives a new configuration owned by the caller. Release it with
 *                    ov_genai_speech_generation_config_free.
 * @return OK on success, INVALID_C_PARAM for NULL arguments, or UNKNOW_EXCEPTION on failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_get_generation_config(const ov_genai_text2speech_pipeline* pipeline,
                                                    ov_genai_speech_generation_config** config);
/**
 * @brief Copy a speech generation configuration into the pipeline.
 * @param[in,out] pipeline Pipeline to update.
 * @param[in] config Configuration to copy; the caller retains ownership.
 * @return OK on success, INVALID_C_PARAM for NULL arguments, or UNKNOW_EXCEPTION on failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_set_generation_config(ov_genai_text2speech_pipeline* pipeline,
                                                    const ov_genai_speech_generation_config* config);
/**
 * @brief Get the expected speaker embedding tensor shape.
 * @param[in] pipeline Pipeline to query.
 * @param[out] shape Receives a shape whose dimensions must be released with ov_shape_free.
 * @return OK on success, INVALID_C_PARAM for NULL arguments, or UNKNOW_EXCEPTION on failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_pipeline_get_speaker_embedding_shape(const ov_genai_text2speech_pipeline* pipeline,
                                                          ov_shape_t* shape);

/**
 * @brief Release a decoded result created by this API. NULL is allowed.
 * @param[in] results Result to release.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2speech_decoded_results_free(ov_genai_text2speech_decoded_results* results);
/**
 * @brief Get the number of generated waveforms.
 * @param[in] results Decoded result to query.
 * @param[out] count Receives the waveform count.
 * @return OK on success or INVALID_C_PARAM for NULL arguments.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_speeches_count(const ov_genai_text2speech_decoded_results* results,
                                                        size_t* count);
/**
 * @brief Get the sample rate shared by the generated waveforms.
 * @param[in] results Decoded result to query.
 * @param[out] sample_rate Receives the rate in samples per second.
 * @return OK on success or INVALID_C_PARAM for NULL arguments.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_output_sample_rate(const ov_genai_text2speech_decoded_results* results,
                                                            uint32_t* sample_rate);
/**
 * @brief Copy a generated waveform to a new float32 tensor.
 * @param[in] results Decoded result to query.
 * @param[in] index Zero-based waveform index, less than the speech count.
 * @param[out] speech Receives a new tensor owned by the caller. Release it with ov_tensor_free.
 * @return OK on success, OUT_OF_BOUNDS for an invalid index, INVALID_C_PARAM for NULL arguments,
 *         or UNKNOW_EXCEPTION on copying failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_speech_at(const ov_genai_text2speech_decoded_results* results,
                                                   size_t index,
                                                   ov_tensor_t** speech);
/**
 * @brief Copy base performance metrics from a decoded result.
 *
 * For text-to-speech, ov_genai_perf_metrics_get_throughput reports samples per second.
 * Token-oriented metrics are not populated and return default values. Use
 * ov_genai_text2speech_decoded_results_get_num_generated_samples to get the generated sample count.
 * @param[in] results Decoded result to query.
 * @param[out] metrics Receives new metrics owned by the caller. Release them with
 *                     ov_genai_text2speech_decoded_results_perf_metrics_free.
 * @return OK on success, INVALID_C_PARAM for NULL arguments, or UNKNOW_EXCEPTION on copying failure.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_perf_metrics(const ov_genai_text2speech_decoded_results* results,
                                                      ov_genai_perf_metrics** metrics);
/**
 * @brief Release performance metrics returned by ov_genai_text2speech_decoded_results_get_perf_metrics.
 * @param[in] metrics Metrics to release. NULL is allowed.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_text2speech_decoded_results_perf_metrics_free(ov_genai_perf_metrics* metrics);
/**
 * @brief Get the total number of samples reported by the generation performance metrics.
 * This is the generated sample count for text-to-speech; token-oriented metrics are not populated.
 * @param[in] results Decoded result to query.
 * @param[out] count Receives the generated sample count.
 * @return OK on success or INVALID_C_PARAM for NULL arguments.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_text2speech_decoded_results_get_num_generated_samples(const ov_genai_text2speech_decoded_results* results,
                                                               size_t* count);
