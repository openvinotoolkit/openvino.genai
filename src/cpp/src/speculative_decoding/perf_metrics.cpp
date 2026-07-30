// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// #pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <ostream>
#include <iostream>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

namespace ov::genai {

MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

MeanStdPair ov::genai::SDPerfMetrics::get_ttst() {
    evaluate_statistics();
    return ttst;
};

MeanStdPair ov::genai::SDPerfMetrics::get_latency() {
    evaluate_statistics();
    return avg_latency;
};

void ov::genai::SDPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated)
        return;

    std::vector<ov::genai::MicroSeconds> second_token_duration;
    std::vector<ov::genai::MicroSeconds> topt_durations;
    std::vector<ov::genai::MicroSeconds> latency_durations;
    std::vector<ov::genai::MicroSeconds> ttft_durations;
    if (raw_metrics.m_durations.size() > 0 && raw_metrics.m_batch_sizes.size() > 0) {
        auto& durations = raw_metrics.m_durations;
        auto& batch_sizes = raw_metrics.m_batch_sizes;

        const auto& generate_durations = raw_metrics.generate_durations;
        const size_t durations_count = durations.size();

        size_t generate_idx = 0;
        ov::genai::MicroSeconds generate_accumulated_duration = ov::genai::MicroSeconds(0.0f);
        size_t step_in_generate = 0;

        num_generated_tokens = 0;

        for (size_t i = 0; i < durations_count; ++i) {
            if (step_in_generate == 0) {
                ttft_durations.emplace_back(durations[i]);
            } else if (step_in_generate == 1) {
                second_token_duration.emplace_back(durations[i]);
            } else {
                latency_durations.emplace_back(durations[i]);
                if (i < batch_sizes.size() && batch_sizes[i] > 0) {
                    topt_durations.emplace_back(durations[i] / batch_sizes[i]);
                }
            }
            if (i < batch_sizes.size()) {
                num_generated_tokens += batch_sizes[i];
            }
            ++step_in_generate;

            if (!generate_durations.empty() && generate_idx < generate_durations.size()) {
                generate_accumulated_duration += durations[i];
                if (generate_accumulated_duration.count() >= generate_durations[generate_idx].count()) {
                    generate_accumulated_duration = ov::genai::MicroSeconds(0.0f);
                    ++generate_idx;
                    step_in_generate = 0;
                }
            }
        }

        if (!generate_durations.empty() || raw_metrics.m_times_to_first_token.empty()) {
            raw_metrics.m_times_to_first_token = ttft_durations;
        }
    }

    std::vector<ov::genai::MicroSeconds> generate_durations_for_stats = raw_metrics.generate_durations;
    if (generate_durations_for_stats.empty() && !raw_metrics.m_durations.empty()) {
        ov::genai::MicroSeconds total_generate_duration(0.0f);
        for (const auto& duration : raw_metrics.m_durations) {
            total_generate_duration += duration;
        }
        generate_durations_for_stats.emplace_back(total_generate_duration);
    }
    generate_duration = calc_mean_and_std(generate_durations_for_stats);

    ttft = ov::genai::calc_mean_and_std(raw_metrics.m_times_to_first_token);
    ttst = ov::genai::calc_mean_and_std(second_token_duration);
    tpot = ov::genai::calc_mean_and_std(topt_durations);

    avg_latency = ov::genai::calc_mean_and_std(latency_durations);

    inference_duration = ov::genai::calc_mean_and_std(raw_metrics.m_inference_durations);

    throughput = {1000.0f / tpot.mean, (tpot.std * 1000.0f) / (tpot.mean * tpot.mean)};

    m_evaluated = true;
}

ov::genai::SDPerfMetrics ov::genai::SDPerfMetrics::operator+(const SDPerfMetrics& right) const {
    PerfMetrics base_result = PerfMetrics::operator+(right);
    SDPerfMetrics result;
    static_cast<PerfMetrics&>(result) = base_result;

    // Some SD raw producers do not fill per-generate totals for sub-model metrics.
    // Synthesize one total per generate call to preserve boundaries after accumulation.
    if (raw_metrics.generate_durations.empty() && !raw_metrics.m_durations.empty()) {
        std::cout << "operator+ raw_metrics generate_durations empty" << std::endl;
        ov::genai::MicroSeconds total_generate_duration(0.0f);
        for (const auto& duration : raw_metrics.m_durations) {
            total_generate_duration += duration;
        }
        result.raw_metrics.generate_durations.emplace_back(total_generate_duration);
    }
    if (right.raw_metrics.generate_durations.empty() && !right.raw_metrics.m_durations.empty()) {
        std::cout << "operator+ right raw_metrics generate_durations empty" << std::endl;
        ov::genai::MicroSeconds total_generate_duration(0.0f);
        for (const auto& duration : right.raw_metrics.m_durations) {
            total_generate_duration += duration;
        }
        result.raw_metrics.generate_durations.emplace_back(total_generate_duration);
    }

    result.m_evaluated = false;
    return result;
}

ov::genai::SDPerfMetrics& ov::genai::SDPerfMetrics::operator+=(const SDPerfMetrics& right) {
    *this = *this + right;
    return *this;
}

ov::genai::SDPerModelsPerfMetrics::SDPerModelsPerfMetrics() : num_accepted_tokens(0) {
    raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
    main_model_metrics.raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
    draft_model_metrics.raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
}

size_t ov::genai::SDPerModelsPerfMetrics::get_num_accepted_tokens() {
    evaluate_statistics();
    return num_accepted_tokens;
};
    
void ov::genai::SDPerModelsPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated)
        return;

    ov::genai::PerfMetrics::evaluate_statistics(start_time);
    // recalculate tpot to take into account all generated tokens and calculate num_accepted_tokens
    if (raw_metrics.m_new_token_times.size() > 0 && raw_metrics.m_batch_sizes.size() > 0) {
        auto& tok_times = raw_metrics.m_new_token_times;
        auto& batch_sizes = raw_metrics.m_batch_sizes;
        num_accepted_tokens = 0;

        for (size_t i = 1; i < tok_times.size(); ++i) {
            num_accepted_tokens += batch_sizes[i] - 1;
        }
    }
    main_model_metrics.evaluate_statistics(start_time);
    draft_model_metrics.evaluate_statistics(start_time);

    m_evaluated = true;
}

ov::genai::SDPerModelsPerfMetrics ov::genai::SDPerModelsPerfMetrics::operator+(const SDPerModelsPerfMetrics& right) const {
    SDPerfMetrics base_result = SDPerfMetrics::operator+(right);
    SDPerModelsPerfMetrics result;
    static_cast<SDPerfMetrics&>(result) = base_result;

    result.main_model_metrics = main_model_metrics + right.main_model_metrics;
    result.draft_model_metrics = draft_model_metrics + right.draft_model_metrics;

    // Keep aggregated accepted-token count even when token timestamps are unavailable after accumulation.
    result.num_accepted_tokens = num_accepted_tokens + right.num_accepted_tokens;
    result.raw_metrics.m_new_token_times.clear();
    result.main_model_metrics.raw_metrics.m_new_token_times.clear();
    result.draft_model_metrics.raw_metrics.m_new_token_times.clear();

    result.m_evaluated = false;
    return result;
}

ov::genai::SDPerModelsPerfMetrics& ov::genai::SDPerModelsPerfMetrics::operator+=(const SDPerModelsPerfMetrics& right) {
    *this = *this + right;
    return *this;
}

}  // namespace ov::genai
