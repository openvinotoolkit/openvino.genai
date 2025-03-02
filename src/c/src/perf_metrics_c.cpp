// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/perf_metrics_c.h"

#include <stdbool.h>
#include <stdint.h>

#include "types_c.h"
#include "openvino/genai/perf_metrics.hpp"

MeanStdPair_C convert_to_c(const ov::genai::MeanStdPair& cpp_pair) {
    MeanStdPair_C c_pair;
    c_pair.mean = cpp_pair.mean;
    c_pair.std = cpp_pair.std;
    return c_pair;
}

ov_genai_perf_metrics* ov_genai_perf_metrics_create() {
    ov_genai_perf_metrics* metrics = new ov_genai_perf_metrics;
    metrics->object = std::make_shared<ov::genai::PerfMetrics>();
    return metrics;
}
void ov_genai_perf_metrics_free(ov_genai_perf_metrics* metrics) {
    if (metrics)
        delete metrics;
}

float ov_genai_perf_metrics_get_load_time(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object)
        return metrics->object->get_load_time();
    return 0.0f;
}
size_t ov_genai_perf_metrics_get_num_generation_tokens(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object)
        return metrics->object->get_num_generated_tokens();
    return 0;
}
size_t ov_genai_perf_metrics_get_num_input_tokens(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return metrics->object->get_num_input_tokens();
    }
    return 0;
}
MeanStdPair_C ov_genai_perf_metrics_get_ttft(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return convert_to_c(metrics->object->get_ttft());
    }
    return {0.0f, 0.0f};
}
MeanStdPair_C ov_genai_perf_metrics_get_tpot(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return convert_to_c(metrics->object->get_tpot());
    }
    return {0.0f, 0.0f};
}

MeanStdPair_C ov_genai_perf_metrics_get_ipot(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return convert_to_c(metrics->object->get_ipot());
    }
    return {0.0f, 0.0f};
}
MeanStdPair_C ov_genai_perf_metrics_get_throughput(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return convert_to_c(metrics->object->get_throughput());
    }
    return {0.0f, 0.0f};
}

MeanStdPair_C ov_genai_perf_metrics_get_inference_duration(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return convert_to_c(metrics->object->get_inference_duration());
    }
    return {0.0f, 0.0f};
}

MeanStdPair_C ov_genai_perf_metrics_get_generate_duration(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return convert_to_c(metrics->object->get_generate_duration());
    }
    return {0.0f, 0.0f};
}
MeanStdPair_C ov_genai_perf_metrics_get_tokenization_duration(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return convert_to_c(metrics->object->get_tokenization_duration());
    }
    return {0.0f, 0.0f};
}
MeanStdPair_C ov_genai_perf_metrics_get_detokenization_duration(const ov_genai_perf_metrics* metrics) {
    if (metrics && metrics->object) {
        return convert_to_c(metrics->object->get_detokenization_duration());
    }
    return {0.0f, 0.0f};
}

// PerfMetrics& operator+=(const PerfMetrics& right);
void ov_genai_perf_metrics_add_in_place(ov_genai_perf_metrics* left, const ov_genai_perf_metrics* right) {
    if (left && right && left->object && right->object) {
        *(left->object) += *(right->object);
    }
}
