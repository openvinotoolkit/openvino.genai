// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"

#include <cmath>
#include <numeric>

namespace ov {
namespace genai {

MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

void SpeechGenerationPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    (void)start_time;
    if (m_evaluated) {
        return;
    }

    generate_duration = calc_mean_and_std(raw_metrics.generate_durations);
    tokenization_duration = calc_mean_and_std(raw_metrics.tokenization_durations);

    if (!raw_ttfa_durations.empty()) {
        ttfa = calc_mean_and_std(raw_ttfa_durations);
    }
    if (!raw_codegen_durations.empty()) {
        codegen_duration = calc_mean_and_std(raw_codegen_durations);
    }
    if (!raw_audio_decode_durations.empty()) {
        audio_decode_duration = calc_mean_and_std(raw_audio_decode_durations);
    }

    // Samples per second
    if (generate_duration.mean > 0.0f) {
        float throughput_mean = static_cast<float>(num_generated_samples) * 1000.0f / generate_duration.mean;
        float throughput_std = (generate_duration.std * 1000.0f * static_cast<float>(num_generated_samples)) /
                               (generate_duration.mean * generate_duration.mean);
        throughput = {throughput_mean, throughput_std};
    } else {
        throughput = {0.0f, 0.0f};
    }

    if (num_generated_samples > 0 && output_sample_rate > 0 && generate_duration.mean > 0.0f) {
        const float audio_duration_ms =
            (static_cast<float>(num_generated_samples) / static_cast<float>(output_sample_rate)) * 1000.0f;
        rtf = audio_duration_ms > 0.0f ? (generate_duration.mean / audio_duration_ms) : 0.0f;
    } else {
        rtf = 0.0f;
    }

    m_evaluated = true;
}

}  // namespace genai
}  // namespace ov
