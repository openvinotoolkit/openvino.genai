// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/perf_metrics.hpp"

#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

MeanStdPair WhisperPerfMetrics::get_features_extraction_diration() {
    evaluate_statistics();
    return features_extraction_diration;
}

void WhisperPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    features_extraction_diration = ov::genai::calc_mean_and_std(whisper_raw_metrics.features_extraction_durations);
    PerfMetrics::evaluate_statistics(start_time);
};

}  // namespace genai
}  // namespace ov
