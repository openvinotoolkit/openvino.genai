// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <vector>
#include <openvino/genai/perf_metrics.hpp>

namespace ov {
namespace genai {

struct PerfCounters {
    std::vector<float> generate_durations;
    std::vector<float> tokenization_duration;
    std::vector<float> detokenization_duration;
    size_t num_generated_tokens;
    size_t num_input_tokens;

    std::vector<size_t> m_batch_sizes;
    std::vector<float> m_durations;
    std::vector<float> m_times_to_first_token;
    std::vector<TimePoint> m_new_token_times;
    void add_timestamp(size_t batch_size);
    // void add_gen_finish_timestamp(size_t batch_size);

};

// class StopWatch {
//     TimePoint m_start;
// public:
//     StopWatch& start() {
//         m_start = std::chrono::steady_clock::now();
//         return *this;
//     }

//     float split() {
//         std::chrono::steady_clock::time_point curr_time = std::chrono::steady_clock::now();
//         return std::chrono::duration_cast<std::chrono::milliseconds>(curr_time - m_start).count();
//     }
// };

} // namespace genai
} // namespace ov
