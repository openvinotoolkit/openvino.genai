// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <numeric>
#include <vector>
#include <cmath>

namespace ov {
namespace genai {

using TimePoints = std::vector<std::chrono::steady_clock::time_point>;

struct GenerationMetrics {
    GenerationMetrics() = default;

    GenerationMetrics(const TimePoints& tok_times, size_t batch_size = 1);
    GenerationMetrics(const std::vector<float>& durations, const std::vector<float>& times_to_first_token, size_t batch_size = 1);

    // First token time.
    float mean_ttft;
    float std_ttft;
    std::vector<float> times_to_first_token;

    // Time per output token.
    float mean_tpot;
    float std_tpot;
    std::vector<float> durations;
    
    std::pair<float, float> get_tokens_per_sec() const;
    size_t batch_size;
    float load_time;

    GenerationMetrics operator+(GenerationMetrics const& metrics) const;
};

} // namespace genai
} // namespace ov
