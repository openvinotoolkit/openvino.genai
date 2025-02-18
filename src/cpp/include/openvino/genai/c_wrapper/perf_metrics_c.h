// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO GenAI C API, which is a C wrapper for ov::genai::PerfMetrics class.
 *
 * @file perf_metrics_c.h
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "../visibility.hpp"
#ifdef __cplusplus
OPENVINO_EXTERN_C {
#endif

    typedef struct {
        float mean;
        float std;
    } MeanStdPair_C;
    /**
     * @struct PerfMetricsHandle
     * @brief type define PerfMetricsHandle from OpaquePerfMetrics
     */
    typedef struct OpaquePerfMetrics PerfMetricsHandle;

    OPENVINO_GENAI_EXPORTS PerfMetricsHandle* CreatePerfMetrics();
    OPENVINO_GENAI_EXPORTS void DestoryPerfMetics(PerfMetricsHandle * metrics);

    OPENVINO_GENAI_EXPORTS float PerfMetricsGetLoadTime(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS size_t PerfMetricsGetNumGeneratedTokens(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS size_t PerfMetricsGetNumInputTokens(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS MeanStdPair_C PerfMetricsGetTtft(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS MeanStdPair_C PerfMetricsGetTpot(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS MeanStdPair_C PerfMetricsGetIpot(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS MeanStdPair_C PerfMetricsGetThroughput(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS MeanStdPair_C PerfMetricsGetInferenceDuration(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS MeanStdPair_C PerfMetricsGetGenerateDuration(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS MeanStdPair_C PerfMetricsGetTokenizationDuration(const PerfMetricsHandle* metrics);
    OPENVINO_GENAI_EXPORTS MeanStdPair_C PerfMetricsGetDetokenizationDuration(const PerfMetricsHandle* metrics);

    // C interface for PerfMetrics& operator+=(const PerfMetrics& right);
    OPENVINO_GENAI_EXPORTS void AddPerfMetricsInPlace(PerfMetricsHandle * left, const PerfMetricsHandle* right);

#ifdef __cplusplus
}
#endif
