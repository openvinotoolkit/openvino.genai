// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "openvino/runtime/exception.hpp"

namespace ov::genai {

float SpeculativeDecodingMetrics::get_avg_acceptance_rate(int64_t request_id) {
    float avg_acceptance_rate = 0.f;
    if (request_id == -1) {
        size_t total_iteration_cnt = 0;
        for (const auto& acceptance_rate : m_acceptance_rate) {
            avg_acceptance_rate += std::accumulate(acceptance_rate.second.begin(), acceptance_rate.second.end(), 0);
            total_iteration_cnt += acceptance_rate.second.size();
        }
        if (total_iteration_cnt == 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        avg_acceptance_rate /= total_iteration_cnt;
    } else {
        auto iter = m_acceptance_rate.find(request_id);
        if (iter == m_acceptance_rate.end()) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        const auto& acceptance_rate = iter->second;
        avg_acceptance_rate = std::accumulate(acceptance_rate.begin(), acceptance_rate.end(), 0);
        if (acceptance_rate.size() == 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        avg_acceptance_rate /= acceptance_rate.size();
    }
    return avg_acceptance_rate;
}

void SpeculativeDecodingMetrics::update_acceptance_rate(int64_t request_id, float acceptance_rate) {
    if (m_acceptance_rate.count(request_id)) {
        m_acceptance_rate[request_id].push_back(acceptance_rate);
    } else {
        m_acceptance_rate.insert({ request_id, std::vector<float>{acceptance_rate} });
    }
}

size_t SpeculativeDecodingMetrics::get_iteration_number(int64_t request_id) {
    return m_acceptance_rate[request_id].size();
}

float SpeculativeDecodingMetrics::get_draft_duration_percentage() {
    return (draft_duration / total_duration) * 100;
}

float SpeculativeDecodingMetrics::get_main_duration_percentage() {
    return (main_duration / total_duration) * 100;
}

float SpeculativeDecodingMetrics::get_inference_duration_percentage() {
    return ((draft_duration + main_duration) / total_duration) * 100;
}

float SpeculativeDecodingMetrics::get_draft_accepted_tokens_percentage(int64_t request_id) {
    float avg_acceptance_rate = 0.f;
    if (request_id == -1) {
        size_t total_iteration_cnt = 0;
        for (const auto& accepten_token_cnt : m_draft_accepted_tokens) {
            avg_acceptance_rate += accepten_token_cnt.second;
            total_iteration_cnt += m_draft_generated_len[request_id];
        }
        OPENVINO_ASSERT(total_iteration_cnt > 0);
        avg_acceptance_rate /= total_iteration_cnt;
    } else {
        auto iter = m_draft_accepted_tokens.find(request_id);
        if (iter == m_draft_accepted_tokens.end()) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        avg_acceptance_rate = m_draft_accepted_tokens[request_id];
        OPENVINO_ASSERT(m_draft_generated_len[request_id] > 0);
        avg_acceptance_rate /= m_draft_generated_len[request_id];
    }
    return avg_acceptance_rate * 100;
}

size_t SpeculativeDecodingMetrics::get_draft_accepted_tokens_counter(int64_t request_id) {
    size_t avg_acceptance_rate = 0;
    if (request_id == -1) {
        size_t total_iteration_cnt = 0;
        for (const auto& accepten_token_cnt : m_draft_accepted_tokens) {
            avg_acceptance_rate += accepten_token_cnt.second;
        }
    } else {
        OPENVINO_ASSERT(m_draft_accepted_tokens.count(request_id));
        const auto& acceptance_rate = m_draft_accepted_tokens[request_id];
        avg_acceptance_rate = acceptance_rate;
    }
    return avg_acceptance_rate;
}

void SpeculativeDecodingMetrics::update_draft_accepted_tokens(int64_t request_id, size_t num_matches) {
    m_draft_accepted_tokens[request_id] += num_matches;
}

void SpeculativeDecodingMetrics::update_draft_generated_len(int64_t request_id, size_t generated_len) {
    m_draft_generated_len[request_id] += generated_len;
}

size_t SpeculativeDecodingMetrics::get_draft_generated_len(int64_t request_id) {
    return m_draft_generated_len.at(request_id);
}

void SpeculativeDecodingMetrics::update_generated_len(size_t generated_len) {
    m_generated_len += generated_len;
}

std::vector<int64_t> SpeculativeDecodingMetrics::get_requests_id() {
    std::vector<int64_t> result;
    for (const auto& req : m_draft_generated_len) {
        result.push_back(req.first);
    }
    return result;
}

void SpeculativeDecodingMetrics::print_acceptance_rates() {
    for (const auto& a : m_acceptance_rate) {
        std::cout << "Request_id: " << a.first << " ||| ";
        for (const auto& b : a.second) {
            std::cout << b << " ";
        }
        std::cout << std::endl;
    }
}

void SpeculativeDecodingMetrics::print(bool is_printing_per_request) {
    if (total_duration == 0) {
        total_duration = draft_duration + main_duration;
    }
    std::cout << "\n=============================== " << std::endl;
    std::cout << "Generated tokens: " << m_generated_len << std::endl;
    std::cout << "Total duration, sec: " << total_duration << std::endl;
    std::cout << "Draft model duration, sec: " << draft_duration << std::endl;
    std::cout << "Main model duration, sec: " << main_duration << std::endl;
    std::cout << "Draft model duration, %: " << get_draft_duration_percentage() << std::endl;
    std::cout << "Main model duration, %: " << get_main_duration_percentage() << std::endl;
    std::cout << "Token per sec: " << float(m_generated_len) / total_duration << std::endl;
    std::cout << "AVG acceptance rate, %: " << get_avg_acceptance_rate(-1) << std::endl;
    std::cout << "=============================== " << std::endl;
    if (is_printing_per_request) {
        for (const auto& i : get_requests_id()) {
            std::cout << "REQUEST_ID: " << i << std::endl;
            std::cout << "Main model iterations: " << get_iteration_number(i) << std::endl;
            std::cout << "AVG acceptance rate, %: " << get_avg_acceptance_rate(i) << std::endl;
            std::cout << "Generated tokens by draft model: " << get_draft_generated_len(i) << std::endl;
            std::cout << "Accepted token rate, %: " << get_draft_accepted_tokens_percentage(i) << std::endl;
            std::cout << "=============================== " << std::endl;
        }
        print_acceptance_rates();
    }

}

void SpeculativeDecodingMetrics::clean_up() {
    m_acceptance_rate.clear();
    m_draft_accepted_tokens.clear();
    m_draft_generated_len.clear();
    m_generated_len = 0;
    draft_duration = 0;
    main_duration = 0;
    total_duration = 0;
}

}
