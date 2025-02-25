// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c_wrapper/perf_metrics_c.h"

#include <stdbool.h>
#include <stdint.h>

#include "common_c.hpp"
#include "openvino/genai/perf_metrics.hpp"

#ifdef __cplusplus
OPENVINO_EXTERN_C {
#endif

    MeanStdPair_C convert_to_c(const ov::genai::MeanStdPair& cpp_pair) {
        MeanStdPair_C c_pair;
        c_pair.mean = cpp_pair.mean;
        c_pair.std = cpp_pair.std;
        return c_pair;
    }

    PerfMetricsHandle* CreatePerfMetrics() {
        PerfMetricsHandle* metrics = new PerfMetricsHandle;
        metrics->object = std::make_shared<ov::genai::PerfMetrics>();
        return metrics;
    }
    void DestroyPerfMetics(PerfMetricsHandle * metrics) {
        if (metrics)
            delete metrics;
    }

    float PerfMetricsGetLoadTime(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object)
            return metrics->object->get_load_time();
        return 0.0f;
    }
    size_t PerfMetricsGetNumGeneratedTokens(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object)
            return metrics->object->get_num_generated_tokens();
        return 0;
    }
    size_t PerfMetricsGetNumInputTokens(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return metrics->object->get_num_input_tokens();
        }
        return 0;
    }
    MeanStdPair_C PerfMetricsGetTtft(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return convert_to_c(metrics->object->get_ttft());
        }
        return {0.0f, 0.0f};
    }
    MeanStdPair_C PerfMetricsGetTpot(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return convert_to_c(metrics->object->get_tpot());
        }
        return {0.0f, 0.0f};
    }

    MeanStdPair_C PerfMetricsGetIpot(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return convert_to_c(metrics->object->get_ipot());
        }
        return {0.0f, 0.0f};
    }
    MeanStdPair_C PerfMetricsGetThroughput(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return convert_to_c(metrics->object->get_throughput());
        }
        return {0.0f, 0.0f};
    }

    MeanStdPair_C PerfMetricsGetInferenceDuration(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return convert_to_c(metrics->object->get_inference_duration());
        }
        return {0.0f, 0.0f};
    }

    MeanStdPair_C PerfMetricsGetGenerateDuration(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return convert_to_c(metrics->object->get_generate_duration());
        }
        return {0.0f, 0.0f};
    }
    MeanStdPair_C PerfMetricsGetTokenizationDuration(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return convert_to_c(metrics->object->get_tokenization_duration());
        }
        return {0.0f, 0.0f};
    }
    MeanStdPair_C PerfMetricsGetDetokenizationDuration(const PerfMetricsHandle* metrics) {
        if (metrics && metrics->object) {
            return convert_to_c(metrics->object->get_detokenization_duration());
        }
        return {0.0f, 0.0f};
    }

    // PerfMetrics& operator+=(const PerfMetrics& right);
    void AddPerfMetricsInPlace(PerfMetricsHandle * left, const PerfMetricsHandle* right) {
        if (left && right && left->object && right->object) {
            *(left->object) += *(right->object);
        }
    }

#ifdef __cplusplus
}
#endif