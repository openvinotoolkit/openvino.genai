// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/perf_metrics.hpp"
#include "perf_counters.hpp"
#include "openvino/openvino.hpp"
#include <tuple>
#include <numeric>
#include <cmath>

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
    
void PerfMetrics::evaluate(TimePoint start_time) {

    auto& tok_times = m_counters->m_new_token_times;
    auto& batch_sizes = m_counters->m_batch_sizes;
    m_counters->m_durations = std::vector<float>(tok_times.size());

    auto ttft = std::chrono::duration_cast<std::chrono::milliseconds>(tok_times[0] - start_time).count();
    m_counters->m_times_to_first_token.emplace_back(ttft);
    
    for (size_t i = 0; i < tok_times.size(); ++i) {
        m_counters->m_durations[i] = std::chrono::duration_cast<std::chrono::milliseconds>(tok_times[i] - start_time).count();
        // If in 10 ms a batch of 5 new tokens is generated then TTOT is 10 ms / 5.
        // todo: float check that it's valid for batch > 1.
        m_counters->m_durations[i] /= batch_sizes[i];
        start_time = tok_times[i];
    }

    std::tie(mean_tpot, std_tpot) = calc_mean_and_std(m_counters->m_durations);
    std::tie(mean_ttft, std_ttft) = calc_mean_and_std(m_counters->m_times_to_first_token);
}

PerfMetrics PerfMetrics::operator+(const PerfMetrics& metrics) const {
    PerfMetrics nm;  // new metrics
    nm.m_counters = m_counters;
    auto& new_counters = nm.m_counters;

    auto& new_durations = new_counters->m_durations;
    auto& new_times_to_first_token = new_counters->m_times_to_first_token;
    
    auto& counters_to_appnd = metrics.m_counters;
    new_durations.insert(new_durations.end(), counters_to_appnd->m_durations.begin(), counters_to_appnd->m_durations.end());
    new_times_to_first_token.insert(new_times_to_first_token.end(), counters_to_appnd->m_times_to_first_token.begin(), counters_to_appnd->m_times_to_first_token.end());
    
    OPENVINO_ASSERT(metrics.load_time == load_time, "generation metrics can be accumulated only for the same pipeline");
    
    std::tie(nm.mean_tpot, nm.std_tpot) = calc_mean_and_std(new_counters->m_durations);
    std::tie(nm.mean_ttft, nm.std_ttft) = calc_mean_and_std(new_counters->m_times_to_first_token);
    
    // todo: add tokenization statistics concatenation.
    
    return nm;
}

PerfMetrics& PerfMetrics::operator+=(const PerfMetrics& right) {
    *this = *this + right;
    return *this;
}



} // namespace genai
} // namespace ov
