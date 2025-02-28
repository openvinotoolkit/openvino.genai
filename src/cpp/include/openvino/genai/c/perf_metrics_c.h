// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO GenAI C API, which is a C wrapper for ov::genai::PerfMetrics class.
 *
 * @file perf_metrics_c.h
 */
#pragma once
#include "common_c.h"

/**
 * @brief Structure to store mean and standard deviation values.
 */
typedef struct {
    float mean;
    float std;
} MeanStdPair_C;

/**
 * @struct PerfMetricsHandle
 * @brief type define PerfMetricsHandle from OpaquePerfMetrics
 */
typedef struct OpaquePerfMetrics PerfMetricsHandle;

/**
 * @brief Create PerfMetricsHandle.
 */
OPENVINO_GENAI_C_EXPORTS PerfMetricsHandle* CreatePerfMetrics();

/**
 * @brief Release the memory allocated by PerfMetricsHandle.
 * @param model A pointer to the PerfMetricsHandle to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void DestroyPerfMetics(PerfMetricsHandle* metrics);

/**
 * @brief Get load time from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Load time in ms.
 */
OPENVINO_GENAI_C_EXPORTS float PerfMetricsGetLoadTime(const PerfMetricsHandle* metrics);

/**
 * @brief Get the number of generated tokens from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return The number of generated tokens.
 */
OPENVINO_GENAI_C_EXPORTS size_t PerfMetricsGetNumGeneratedTokens(const PerfMetricsHandle* metrics);

/**
 * @brief Get the number of input tokens from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return The number of input tokens.
 */
OPENVINO_GENAI_C_EXPORTS size_t PerfMetricsGetNumInputTokens(const PerfMetricsHandle* metrics);

/**
 * @brief Get the time to first token (in ms) from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Mean and standard deviation of time to first token.
 */
OPENVINO_GENAI_C_EXPORTS MeanStdPair_C PerfMetricsGetTtft(const PerfMetricsHandle* metrics);

/**
 * @brief Get the time per output token (TPOT in ms) from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Mean and standard deviation of time per output token.
 */
OPENVINO_GENAI_C_EXPORTS MeanStdPair_C PerfMetricsGetTpot(const PerfMetricsHandle* metrics);

/**
 * @brief Get the inference time (in ms) per output token from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Mean and standard deviation of inference time per input token.
 */
OPENVINO_GENAI_C_EXPORTS MeanStdPair_C PerfMetricsGetIpot(const PerfMetricsHandle* metrics);

/**
 * @brief Get tokens per second from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Mean and standard deviation of throughput.
 */
OPENVINO_GENAI_C_EXPORTS MeanStdPair_C PerfMetricsGetThroughput(const PerfMetricsHandle* metrics);

/**
 * @brief Get inference duration (in ms) from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Mean and standard deviation of inference duration.
 */
OPENVINO_GENAI_C_EXPORTS MeanStdPair_C PerfMetricsGetInferenceDuration(const PerfMetricsHandle* metrics);

/**
 * @brief Get generate duration (in ms) from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Mean and standard deviation of generate duration.
 */
OPENVINO_GENAI_C_EXPORTS MeanStdPair_C PerfMetricsGetGenerateDuration(const PerfMetricsHandle* metrics);

/**
 * @brief Get tokenization duration (in ms) from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Mean and standard deviation of tokenization duration.
 */
OPENVINO_GENAI_C_EXPORTS MeanStdPair_C PerfMetricsGetTokenizationDuration(const PerfMetricsHandle* metrics);

/**
 * @brief Get detokenization duration (in ms) from PerfMetricsHandle.
 * @param metrics A pointer to the PerfMetricsHandle.
 * @return Mean and standard deviation of detokenization duration.
 */
OPENVINO_GENAI_C_EXPORTS MeanStdPair_C PerfMetricsGetDetokenizationDuration(const PerfMetricsHandle* metrics);

/**
 * @brief C interface for PerfMetrics& operator+=(const PerfMetrics& right)
 *
 * This function adds the PerfMetrics from 'right' to 'left' in place.
 *
 * @param left A pointer to the PerfMetricsHandle that will be updated.
 * @param right A pointer to the PerfMetricsHandle whose metrics will be added to 'left'.
 */
OPENVINO_GENAI_C_EXPORTS void AddPerfMetricsInPlace(PerfMetricsHandle* left, const PerfMetricsHandle* right);
