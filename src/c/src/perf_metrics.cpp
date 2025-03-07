// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/perf_metrics.h"

#include <stdbool.h>
#include <stdint.h>

#include "openvino/genai/perf_metrics.hpp"
#include "types_c.h"

ov_genai_perf_metrics* ov_genai_perf_metrics_create() {
    ov_genai_perf_metrics* metrics = new ov_genai_perf_metrics;
    metrics->object = std::make_shared<ov::genai::PerfMetrics>();
    return metrics;
}
void ov_genai_perf_metrics_free(ov_genai_perf_metrics* metrics) {
    if (metrics)
        delete metrics;
}

ov_status_e ov_genai_perf_metrics_get_load_time(const ov_genai_perf_metrics* metrics, float* load_time) {
    if (!metrics || !(metrics->object) || !load_time) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *load_time = metrics->object->get_load_time();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_perf_metrics_get_num_generation_tokens(const ov_genai_perf_metrics* metrics,
                                                            size_t* num_generated_tokens) {
    if (!metrics || !(metrics->object) || !num_generated_tokens) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *num_generated_tokens = metrics->object->get_num_generated_tokens();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_perf_metrics_get_num_input_tokens(const ov_genai_perf_metrics* metrics, size_t* num_input_tokens) {
    if (!metrics || !(metrics->object) || !num_input_tokens) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *num_input_tokens = metrics->object->get_num_input_tokens();
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_perf_metrics_get_ttft(const ov_genai_perf_metrics* metrics, float* mean, float* std) {
    if (!metrics || !(metrics->object) || !mean || !std) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto ttft = metrics->object->get_ttft();
        *mean = ttft.mean;
        *std = ttft.std;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_perf_metrics_get_tpot(const ov_genai_perf_metrics* metrics, float* mean, float* std) {
    if (!metrics || !(metrics->object) || !mean || !std) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto tpot = metrics->object->get_tpot();
        *mean = tpot.mean;
        *std = tpot.std;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_perf_metrics_get_ipot(const ov_genai_perf_metrics* metrics, float* mean, float* std) {
    if (!metrics || !(metrics->object) || !mean || !std) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto ipot = metrics->object->get_ipot();
        *mean = ipot.mean;
        *std = ipot.std;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_perf_metrics_get_throughput(const ov_genai_perf_metrics* metrics, float* mean, float* std) {
    if (!metrics || !(metrics->object) || !mean || !std) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto throughput = metrics->object->get_throughput();
        *mean = throughput.mean;
        *std = throughput.std;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_perf_metrics_get_inference_duration(const ov_genai_perf_metrics* metrics,
                                                         float* mean,
                                                         float* std) {
    if (!metrics || !(metrics->object) || !mean || !std) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto inference_duration = metrics->object->get_inference_duration();
        *mean = inference_duration.mean;
        *std = inference_duration.std;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_genai_perf_metrics_get_generate_duration(const ov_genai_perf_metrics* metrics, float* mean, float* std) {
    if (!metrics || !(metrics->object) || !mean || !std) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto generation_duration = metrics->object->get_generate_duration();
        *mean = generation_duration.mean;
        *std = generation_duration.std;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_perf_metrics_get_tokenization_duration(const ov_genai_perf_metrics* metrics,
                                                            float* mean,
                                                            float* std) {
    if (!metrics || !(metrics->object) || !mean || !std) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto tokenization_duration = metrics->object->get_tokenization_duration();
        *mean = tokenization_duration.mean;
        *std = tokenization_duration.std;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_perf_metrics_get_detokenization_duration(const ov_genai_perf_metrics* metrics,
                                                              float* mean,
                                                              float* std) {
    if (!metrics || !(metrics->object) || !mean || !std) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto detokenization_duration = metrics->object->get_detokenization_duration();
        *mean = detokenization_duration.mean;
        *std = detokenization_duration.std;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
ov_status_e ov_genai_perf_metrics_add_in_place(ov_genai_perf_metrics* left, const ov_genai_perf_metrics* right) {
    if (!left || !(left->object) || !right || !(right->object)) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        *(left->object) += *(right->object);
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
