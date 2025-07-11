// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO GenAI C API, which is a C wrapper for ov::genai::PerfMetrics class.
 *
 * @file perf_metrics_c.h
 */
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "openvino/c/ov_common.h"
#include "openvino/genai/c/visibility.h"

/**
 * @struct ov_genai_perf_metrics
 * @brief type define ov_genai_perf_metrics from ov_genai_perf_metrics_opaque.
 */
typedef struct ov_genai_perf_metrics_opaque ov_genai_perf_metrics;

/**
 * @brief Get load time from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param load_time Load time in ms.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_get_load_time(const ov_genai_perf_metrics* metrics,
                                                                         float* load_time);

/**
 * @brief Get the number of generated tokens from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param num_generation_tokens The number of generated tokens.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_perf_metrics_get_num_generation_tokens(const ov_genai_perf_metrics* metrics, size_t* num_generation_tokens);

/**
 * @brief Get the number of input tokens from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param num_input_tokens The number of input tokens.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_get_num_input_tokens(const ov_genai_perf_metrics* metrics,
                                                                                size_t* num_input_tokens);

/**
 * @brief Get the time to first token (in ms) from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param mean Mean of time to first token.
 *  @param std Standard deviation of time to first token.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_get_ttft(const ov_genai_perf_metrics* metrics,
                                                                    float* mean,
                                                                    float* std);

/**
 * @brief Get the time per output token (TPOT in ms) from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param mean Mean of time per output token.
 * @param std Standard deviation of time per output token.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_get_tpot(const ov_genai_perf_metrics* metrics,
                                                                    float* mean,
                                                                    float* std);

/**
 * @brief Get the inference time (in ms) per output token from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param mean Mean of inference time per input token.
 * @param std Standard deviation of inference time per input token.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_get_ipot(const ov_genai_perf_metrics* metrics,
                                                                    float* mean,
                                                                    float* std);

/**
 * @brief Get tokens per second from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param mean Mean of throughput.
 * @param std Standard deviation of throughput.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_get_throughput(const ov_genai_perf_metrics* metrics,
                                                                          float* mean,
                                                                          float* std);

/**
 * @brief Get inference duration (in ms) from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param mean Mean of inference duration.
 * @param std Standard deviation of inference duration.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_get_inference_duration(const ov_genai_perf_metrics* metrics,
                                                                                  float* mean,
                                                                                  float* std);

/**
 * @brief Get generate duration (in ms) from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
.* @param mean Mean of generate duration.
 * @param std Standard deviation of generate duration.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_get_generate_duration(const ov_genai_perf_metrics* metrics,
                                                                                 float* mean,
                                                                                 float* std);

/**
 * @brief Get tokenization duration (in ms) from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param mean Mean of tokenization duration.
 * @param std Standard deviation of tokenization duration.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_perf_metrics_get_tokenization_duration(const ov_genai_perf_metrics* metrics, float* mean, float* std);

/**
 * @brief Get detokenization duration (in ms) from ov_genai_perf_metrics.
 * @param metrics A pointer to the ov_genai_perf_metrics instance.
 * @param mean Mean of detokenization duration.
 * @param std Standard deviation of detokenization duration.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_perf_metrics_get_detokenization_duration(const ov_genai_perf_metrics* metrics, float* mean, float* std);

/**
 * @brief C interface for PerfMetrics& operator+=(const PerfMetrics& right)
 *
 * This function adds the PerfMetrics from 'right' to 'left' in place. Equivalent to ov::genai::PerfMetrics&
 * operator+=(const ov::genai::PerfMetrics&);
 *
 * @param left A pointer to the ov_genai_perf_metrics instance that will be updated.
 * @param right A pointer to the ov_genai_perf_metrics instance whose metrics will be added to 'left'.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_perf_metrics_add_in_place(ov_genai_perf_metrics* left,
                                                                        const ov_genai_perf_metrics* right);
