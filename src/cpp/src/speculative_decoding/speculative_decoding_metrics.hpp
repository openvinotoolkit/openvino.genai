// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <chrono>
#include <map>

namespace ov::genai {
class SpeculativeDecodingMetrics {
    // percent of draft model using time + draft model gen tokens
    using AcceptanceRate = std::vector<float>;
    // { request_id, acceptance_rate }
    std::map<int64_t, AcceptanceRate> m_acceptance_rate;

    std::map<int64_t, size_t> m_draft_accepted_tokens;
    std::map<int64_t, size_t> m_draft_generated_len;

public:
    float draft_duration = 0, main_duration = 0, total_duration = 0;
    size_t m_generated_len = 0;

    float get_avg_acceptance_rate(int64_t request_id);
    void update_acceptance_rate(int64_t request_id, float acceptance_rate);

    float get_draft_accepted_tokens_percentage(int64_t request_id);
    size_t get_draft_accepted_tokens_counter(int64_t request_id);
    void update_draft_accepted_tokens(int64_t request_id, size_t num_matches);

    void update_draft_generated_len(int64_t request_id, size_t generated_len);
    void update_generated_len(size_t generated_len);
    size_t get_draft_generated_len(int64_t request_id);

    size_t get_iteration_number(int64_t request_id);

    float get_draft_duration_percentage();
    float get_main_duration_percentage();
    float get_inference_duration_percentage();

    std::vector<int64_t> get_requests_id();

    void print_acceptance_rates();
    void print(bool is_printing_per_request = false);

    void clean_up();
};
}