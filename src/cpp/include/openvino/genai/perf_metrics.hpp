// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include "openvino/genai/visibility.hpp"
#include <vector>
#include <memory>

namespace ov {
namespace genai {

using TimePoint = std::chrono::steady_clock::time_point;

struct PerfCounters;

struct OPENVINO_GENAI_EXPORTS PerfMetrics {
    // First token time.
    float mean_ttft;
    float std_ttft;

    // Time per output token.
    float mean_tpot;
    float std_tpot;
    
    float load_time;
    float start_time;

    float mean_generate_duration;
    float mean_decoding_duration;
    float mean_encoding_duration;
    
    float mean_throughput;
    float std_throughput;

    size_t num_generated_tokens;
    size_t num_input_tokens;

    std::shared_ptr<PerfCounters> m_counters;
    void evaluate(TimePoint start_time);

    PerfMetrics operator+(const PerfMetrics& metrics) const;
    PerfMetrics& operator+=(const PerfMetrics& right);

    
};

} // namespace genai
} // namespace ov
