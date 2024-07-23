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
* @brief Structure to store mean and standart deviation values.
*/
struct OPENVINO_GENAI_EXPORTS MeanStdPair {
    float mean;
    float std;
};

/**
* @brief Structure to store performance metric for each generation.
* 
* @param
*/
struct OPENVINO_GENAI_EXPORTS PerfMetrics {
    float load_time;   // Load time in ms.
    MeanStdPair ttft;  // Time to the first token (in ms) (TTTFT).
    MeanStdPair tpot;  // Time (in ms) per output token (TPOT).
    MeanStdPair throughput;  // Tokens per second.
    
    MeanStdPair generate_duration;
    MeanStdPair tokenization_duration = {-1, -1};
    MeanStdPair detokenization_duration = {-1. -1};

    size_t num_generated_tokens;
    size_t num_input_tokens;

    /** 
     * @brief calculates mean/std values from raw_metrics. 
     * 
     * @param start_time optional start_time in case if duration needs to be updated.
     */
    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt);
    
    /** 
     * @brief convert duration to microseconds
     * 
     * @param duration duration in 
     */
    static float get_microsec(std::chrono::steady_clock::duration duration);
    PerfMetrics operator+(const PerfMetrics& metrics) const;
    PerfMetrics& operator+=(const PerfMetrics& right);

    RawPerfMetrics raw_metrics;
};

} // namespace genai
} // namespace ov
