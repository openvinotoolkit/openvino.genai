// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/speech_generation_perf_metrics.h"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"

#include <cstdlib>

int speech_generation_perf_metrics_create(speech_generation_perf_metrics_t** metrics) {
    if (!metrics) {
        return 1;  // Invalid argument
    }

    try {
        *metrics = new speech_generation_perf_metrics_t();
        return 0;
    } catch (...) {
        return 1;
    }
}

int speech_generation_perf_metrics_destroy(speech_generation_perf_metrics_t* metrics) {
    if (!metrics) {
        return 1;  // Invalid argument
    }

    try {
        delete metrics;
        return 0;
    } catch (...) {
        return 1;
    }
}

int speech_generation_perf_metrics_get_num_generated_samples(const speech_generation_perf_metrics_t* metrics, int* value) {
    if (!metrics || !value) {
        return 1;  // Invalid argument
    }

    try {
        *value = metrics->num_generated_samples;
        return 0;
    } catch (...) {
        return 1;
    }
}

int speech_generation_perf_metrics_get_generate_duration(const speech_generation_perf_metrics_t* metrics, float* value) {
    if (!metrics || !value) {
        return 1;  // Invalid argument
    }

    try {
        *value = metrics->base.generate_duration;
        return 0;
    } catch (...) {
        return 1;
    }
}

int speech_generation_perf_metrics_get_throughput(const speech_generation_perf_metrics_t* metrics, float* value) {
    if (!metrics || !value) {
        return 1;  // Invalid argument
    }

    try {
        *value = metrics->base.throughput;
        return 0;
    } catch (...) {
        return 1;
    }
}
