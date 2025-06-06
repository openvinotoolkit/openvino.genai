// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"


namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS SDPerfMetrics : public ov::genai::PerfMetrics {
    ov::genai::MeanStdPair ttst;  // Time to the second token (in ms).
    ov::genai::MeanStdPair ttl;  // latency from the third token (in ms).
    size_t num_accepted_tokens; // num tokens, which was accepted but main model
    size_t num_iterations; // num inference

    ov::genai::MeanStdPair get_ttst();
    ov::genai::MeanStdPair get_latency();

    SDPerfMetrics() = default;

    size_t get_num_accepted_tokens();
    size_t get_num_iterations();

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;
};

struct OPENVINO_GENAI_EXPORTS SDPerModelsPerfMetrics : public ov::genai::SDPerfMetrics {
    ov::genai::SDPerfMetrics main_model_metrics;
    ov::genai::SDPerfMetrics draft_model_metrics;

    SDPerModelsPerfMetrics();

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;
};

}