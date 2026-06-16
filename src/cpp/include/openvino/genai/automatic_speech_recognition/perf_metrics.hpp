// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {
struct ASRRawPerfMetrics {
    std::vector<MicroSeconds> features_extraction_durations;
    std::vector<MicroSeconds> word_level_timestamps_processing_durations;
    std::vector<MicroSeconds> encode_inference_durations;
    std::vector<MicroSeconds> decode_inference_durations;
};

struct OPENVINO_GENAI_EXPORTS ASRPerfMetrics : public PerfMetrics {
    ASRPerfMetrics() = default;
    ASRPerfMetrics(const PerfMetrics& perf_metrics) : PerfMetrics(perf_metrics) {}

    MeanStdPair features_extraction_duration;
    MeanStdPair word_level_timestamps_processing_duration;
    MeanStdPair encode_inference_duration;
    MeanStdPair decode_inference_duration;

    MeanStdPair get_features_extraction_duration();
    MeanStdPair get_word_level_timestamps_processing_duration();
    MeanStdPair get_encode_inference_duration();
    MeanStdPair get_decode_inference_duration();

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;

    ASRPerfMetrics operator+(const ASRPerfMetrics& metrics) const;
    ASRPerfMetrics& operator+=(const ASRPerfMetrics& right);

    ASRRawPerfMetrics asr_raw_metrics;
};
}  // namespace ov::genai
