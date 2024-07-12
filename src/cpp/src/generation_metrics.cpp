// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/generation_metrics.hpp"
#include <tuple>

namespace {

std::pair<float, float> calc_mean_and_std(const std::vector<float>& durations) {
    float mean = std::accumulate(durations.begin(), durations.end(), 0.0f) / durations.size();
    
    float sum_square_durations = std::accumulate(durations.begin(), durations.end(), 0.0f,
        [](const float& acc, const float& duration) -> float {
            return acc + duration * duration;
        });
    float std = std::sqrt(sum_square_durations / durations.size() - mean * mean);      
    return {mean, std};
}

} // namespace

namespace ov {
namespace genai {
    

GenerationMetrics::GenerationMetrics(const TimePoints& tok_times, size_t batch_size) {
    this->batch_size = batch_size;
    durations = std::vector<float>(tok_times.size() - 1);
    for (size_t i = 1; i < tok_times.size(); ++i) {
        durations[i - 1] = std::chrono::duration_cast<std::chrono::milliseconds>(tok_times[i] - tok_times[i - 1]).count();
    }
    times_to_first_token.emplace_back(durations[0]);

    std::tie(mean_tpot, std_tpot) = calc_mean_and_std(durations);
    std::tie(mean_ttft, std_ttft) = calc_mean_and_std(times_to_first_token);
}

GenerationMetrics::GenerationMetrics(const std::vector<float>& durations_, const std::vector<float>& times_to_first_token_, size_t batch_size)
    : durations(durations_), times_to_first_token(times_to_first_token_) {
    this->batch_size = batch_size;
    std::tie(mean_tpot, std_tpot) = calc_mean_and_std(durations);
    std::tie(mean_ttft, std_ttft) = calc_mean_and_std(times_to_first_token);
}

GenerationMetrics GenerationMetrics::operator+(GenerationMetrics const& metrics) const {
    std::vector<float> new_durations = durations;
    std::vector<float> new_times_to_first_token = times_to_first_token;
    new_durations.insert(new_durations.end(), metrics.durations.begin(), metrics.durations.end());
    new_times_to_first_token.insert(new_times_to_first_token.end(), metrics.times_to_first_token.begin(), metrics.times_to_first_token.end());
    
    return GenerationMetrics(new_durations, new_times_to_first_token);
}

std::pair<float, float> GenerationMetrics::get_tokens_per_sec() const {
   auto mean_tps = 1000.0f * batch_size / mean_tpot;
   auto std_tps = 1000.0f * std_tpot / (mean_tpot * mean_tpot);
   return {mean_tps, std_tps};
}


} // namespace genai
} // namespace ov
