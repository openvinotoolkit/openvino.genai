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

/**
* @brief Structure with raw performance metrics for each generation before any statistics calculated.
*/
struct OPENVINO_GENAI_EXPORTS RawPerfMetrics {
    std::vector<float> generate_durations;
    std::vector<float> tokenization_durations;
    std::vector<float> detokenization_durations;
    
    std::vector<float> m_times_to_first_token;
    std::vector<TimePoint> m_new_token_times;
    std::vector<size_t> m_batch_sizes;
    std::vector<float> m_durations;

    size_t num_generated_tokens;
    size_t num_input_tokens;
};

/**
* @brief Structure to store performance metric for each generation
*
*/
struct OPENVINO_GENAI_EXPORTS PerfMetrics {
    // First token time.
    float mean_ttft;
    float std_ttft;

    // Time per output token.
    float mean_tpot;
    float std_tpot;
    
    float load_time;

    float mean_generate_duration;
    float std_generate_duration;
    float mean_tokenization_duration;
    float std_tokenization_duration;
    float mean_detokenization_duration;
    float std_detokenization_duration;
    
    float mean_throughput;
    float std_throughput;

    size_t num_generated_tokens;
    size_t num_input_tokens;

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt);
    static float get_duration_ms(std::chrono::steady_clock::duration duration);
    PerfMetrics operator+(const PerfMetrics& metrics) const;
    PerfMetrics& operator+=(const PerfMetrics& right);

    RawPerfMetrics raw_counters;
};

} // namespace genai
} // namespace ov
