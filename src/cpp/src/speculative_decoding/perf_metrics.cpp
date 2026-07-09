// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// #pragma once

#include <chrono>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <ostream>
#include <iostream>

#include "openvino/runtime/exception.hpp"
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
    std::vector<ov::genai::MicroSeconds> apt_durations;
    if (raw_metrics.m_durations.size() > 0 && raw_metrics.m_batch_sizes.size() > 0) {
        auto& durations = raw_metrics.m_durations;
        auto& batch_sizes = raw_metrics.m_batch_sizes;

        raw_metrics.m_times_to_first_token.clear();
        raw_metrics.m_times_to_first_token.emplace_back(durations[0]);

        num_generated_tokens = batch_sizes[0];

        raw_metrics.generate_durations.clear();
        raw_metrics.generate_durations.emplace_back(durations[0]);

        for (size_t i = 1; i < durations.size(); ++i) {
            if (i == 1) {
                second_token_duration.emplace_back(durations[i]);
            } else {
                latency_durations.emplace_back(durations[i]);
                topt_durations.emplace_back(durations[i] / batch_sizes[i]);
            }

            raw_metrics.generate_durations[0] += durations[i];
            num_generated_tokens += batch_sizes[i];
        }
    }

    generate_duration = calc_mean_and_std(raw_metrics.generate_durations);

    ttft = ov::genai::calc_mean_and_std(raw_metrics.m_times_to_first_token);
    ttst = ov::genai::calc_mean_and_std(second_token_duration);
    tpot = ov::genai::calc_mean_and_std(topt_durations);

    avg_latency = ov::genai::calc_mean_and_std(latency_durations);

    inference_duration = ov::genai::calc_mean_and_std(raw_metrics.m_inference_durations);

    throughput = {1000.0f / tpot.mean, (tpot.std * 1000.0f) / (tpot.mean * tpot.mean)};

    m_evaluated = true;
}

ov::genai::SDPerModelsPerfMetrics::SDPerModelsPerfMetrics() : num_accepted_tokens(0), num_draft_tokens(0) {
    raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
    main_model_metrics.raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
    draft_model_metrics.raw_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
}

size_t ov::genai::SDPerModelsPerfMetrics::get_num_accepted_tokens() {
    evaluate_statistics();
    return num_accepted_tokens;
};

size_t ov::genai::SDPerModelsPerfMetrics::get_num_draft_tokens() {
    evaluate_statistics();
    return num_draft_tokens;
};

size_t ov::genai::SDPerModelsPerfMetrics::get_num_rejected_tokens() {
    evaluate_statistics();
    if (num_draft_tokens == 0) {
        return 0;
    }
    OPENVINO_ASSERT(num_draft_tokens >= num_accepted_tokens,
                    "The number of accepted draft tokens cannot exceed the number of draft candidate tokens.");
    return num_draft_tokens - num_accepted_tokens;
};

float ov::genai::SDPerModelsPerfMetrics::get_draft_acceptance_rate() {
    evaluate_statistics();
    if (num_draft_tokens == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(num_accepted_tokens) / static_cast<float>(num_draft_tokens);
};
    
void ov::genai::SDPerModelsPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated)
        return;

    ov::genai::PerfMetrics::evaluate_statistics(start_time);
    // Recalculate tpot to take into account all generated tokens. Older speculative
    // paths did not fill num_draft_tokens, so keep the batch-size based accepted
    // token estimate as a fallback for those paths only.
    if (raw_metrics.m_new_token_times.size() > 0 && raw_metrics.m_batch_sizes.size() > 0) {
        auto& tok_times = raw_metrics.m_new_token_times;
        auto& batch_sizes = raw_metrics.m_batch_sizes;
        if (num_draft_tokens == 0) {
            num_accepted_tokens = 0;
        }

        for (size_t i = 1; i < tok_times.size(); ++i) {
            if (num_draft_tokens == 0) {
                num_accepted_tokens += batch_sizes[i] - 1;
            }
        }
    }
    main_model_metrics.evaluate_statistics(start_time);
    draft_model_metrics.evaluate_statistics(start_time);

    m_evaluated = true;
}

}  // namespace ov::genai
