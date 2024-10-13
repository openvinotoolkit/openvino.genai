// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <chrono>
#include <map>

namespace ov::genai {

class Timer {
    double m_total;
    decltype(std::chrono::steady_clock::now()) m_start;

public:
    Timer() :
        m_total(0.) {
    }

    void start() {
        m_start = std::chrono::steady_clock::now();
    }

    void end() {
        auto m_end = std::chrono::steady_clock::now();
        m_total += std::chrono::duration<double, std::milli>(m_end - m_start).count();
    }

    float get_duration_ms() {
        return m_total / 1000.;
    }
};

class SpeculativeDecodingMetrics {
    // percent of draft model using time + draft model gen tokens
    using AcceptanceRate = std::vector<float>;
    // { request_id, acceptance_rate }
    std::map<int64_t, AcceptanceRate> m_acceptance_rate;

    std::map<int64_t, size_t> m_draft_accepted_tokens;
    std::map<int64_t, size_t> m_generated_len;

public:
    float draft_duration = 0, main_duration = 0, total_duration = 0;

    float get_avg_acceptance_rate(int64_t request_id);
    void update_acceptance_rate(int64_t request_id, float acceptance_rate);

    float get_draft_accepted_tokens_percentage(int64_t request_id);
    void update_draft_accepted_tokens(int64_t request_id, size_t num_matches);

    void set_generated_len(int64_t request_id, size_t generated_len);

    size_t get_iteration_number(int64_t request_id);

    float get_draft_duration_percentage();
    float get_main_duration_percentage();
    float get_inference_duration_percentage();

};

}