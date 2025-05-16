// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"

#include <cmath>
#include <numeric>

namespace ov {
namespace genai {

MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

void SpeechGenerationPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    generate_duration = calc_mean_and_std(raw_metrics.generate_durations);
    tokenization_duration = calc_mean_and_std(raw_metrics.tokenization_durations);

    // tokens per second

    float throughput_mean = static_cast<float>(num_generated_samples) * 1000.0f / generate_duration.mean;
    float throughput_std = (generate_duration.std * 1000.0f * static_cast<float>(num_generated_samples)) /
                           (generate_duration.mean * generate_duration.mean);
    throughput = {throughput_mean, throughput_std};
    m_evaluated = true;
}

}  // namespace genai
}  // namespace ov
