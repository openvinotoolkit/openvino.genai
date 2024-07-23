// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include "openvino/genai/visibility.hpp"
#include <vector>
#include <memory>
#include <optional>

namespace ov {
namespace genai {

using TimePoint = std::chrono::steady_clock::time_point;
using MicroSeconds = std::chrono::duration<float, std::ratio<1, 1000000>>;

/**
* @brief Structure with raw performance metrics for each generation before any statistics calculated.
*/
struct OPENVINO_GENAI_EXPORTS RawPerfMetrics {
    std::vector<MicroSeconds> generate_durations;
    std::vector<MicroSeconds> tokenization_durations;
    std::vector<MicroSeconds> detokenization_durations;
    
    std::vector<MicroSeconds> m_times_to_first_token;
    std::vector<TimePoint> m_new_token_times;
    std::vector<size_t> m_batch_sizes;
    std::vector<MicroSeconds> m_durations;

    size_t num_generated_tokens;
    size_t num_input_tokens;
};

/**
* @brief Structure to store performance metric for each generation
*
*/
struct OPENVINO_GENAI_EXPORTS PerfMetrics {
    // Load time in ms.
    float load_time;

    // First token time (in ms).
    float mean_ttft;
    float std_ttft;

    // Time (in ms) per output token.
    float mean_tpot;
    float std_tpot;
    
    float mean_generate_duration;
    float std_generate_duration;
    float mean_tokenization_duration = -1;
    float std_tokenization_duration = -1;
    float mean_detokenization_duration = -1;
    float std_detokenization_duration = -1;
     
    // Tokens per second.
    float mean_throughput;
    float std_throughput;

    size_t num_generated_tokens;
    size_t num_input_tokens;

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt);
    static float get_microsec(std::chrono::steady_clock::duration duration);
    PerfMetrics operator+(const PerfMetrics& metrics) const;
    PerfMetrics& operator+=(const PerfMetrics& right);

    RawPerfMetrics raw_metrics;
};

} // namespace genai
} // namespace ov
