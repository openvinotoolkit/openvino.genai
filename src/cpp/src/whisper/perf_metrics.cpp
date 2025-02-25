// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

MeanStdPair WhisperPerfMetrics::get_features_extraction_duration() {
    evaluate_statistics();
    return features_extraction_duration;
}

void WhisperPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    features_extraction_duration = ov::genai::calc_mean_and_std(whisper_raw_metrics.features_extraction_durations);
    PerfMetrics::evaluate_statistics(start_time);
};

WhisperPerfMetrics WhisperPerfMetrics::operator+(const WhisperPerfMetrics& right) const {
    PerfMetrics base_result = PerfMetrics::operator+(right);
    WhisperPerfMetrics result{base_result};

    // copy left whisper raw metrics
    result.whisper_raw_metrics = whisper_raw_metrics;

    // insert right metrics
    auto& result_features_extraction_durations = result.whisper_raw_metrics.features_extraction_durations;
    auto& right_features_extraction_durations = right.whisper_raw_metrics.features_extraction_durations;
    result_features_extraction_durations.insert(result_features_extraction_durations.end(),
                                                right_features_extraction_durations.begin(),
                                                right_features_extraction_durations.end());
    return result;
}

WhisperPerfMetrics& WhisperPerfMetrics::operator+=(const WhisperPerfMetrics& right) {
    *this = *this + right;
    return *this;
}

}  // namespace genai
}  // namespace ov
