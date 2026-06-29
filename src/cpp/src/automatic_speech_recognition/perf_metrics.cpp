// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/perf_metrics.hpp"

namespace ov::genai {

MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

MeanStdPair ASRPerfMetrics::get_features_extraction_duration() {
    evaluate_statistics();
    return features_extraction_duration;
}

MeanStdPair ASRPerfMetrics::get_word_level_timestamps_processing_duration() {
    evaluate_statistics();
    return word_level_timestamps_processing_duration;
}

MeanStdPair ASRPerfMetrics::get_encode_inference_duration() {
    evaluate_statistics();
    return encode_inference_duration;
}

MeanStdPair ASRPerfMetrics::get_decode_inference_duration() {
    evaluate_statistics();
    return decode_inference_duration;
}

void ASRPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    features_extraction_duration = calc_mean_and_std(asr_raw_metrics.features_extraction_durations);
    word_level_timestamps_processing_duration =
        calc_mean_and_std(asr_raw_metrics.word_level_timestamps_processing_durations);
    encode_inference_duration = calc_mean_and_std(asr_raw_metrics.encode_inference_durations);
    decode_inference_duration = calc_mean_and_std(asr_raw_metrics.decode_inference_durations);

    PerfMetrics::evaluate_statistics(start_time);
}

ASRPerfMetrics ASRPerfMetrics::operator+(const ASRPerfMetrics& right) const {
    PerfMetrics base_result = PerfMetrics::operator+(right);
    ASRPerfMetrics result{base_result};

    result.asr_raw_metrics = asr_raw_metrics;

    auto& result_features = result.asr_raw_metrics.features_extraction_durations;
    const auto& right_features = right.asr_raw_metrics.features_extraction_durations;
    result_features.insert(result_features.end(), right_features.begin(), right_features.end());

    auto& result_wlt = result.asr_raw_metrics.word_level_timestamps_processing_durations;
    const auto& right_wlt = right.asr_raw_metrics.word_level_timestamps_processing_durations;
    result_wlt.insert(result_wlt.end(), right_wlt.begin(), right_wlt.end());

    auto& result_encode = result.asr_raw_metrics.encode_inference_durations;
    const auto& right_encode = right.asr_raw_metrics.encode_inference_durations;
    result_encode.insert(result_encode.end(), right_encode.begin(), right_encode.end());

    auto& result_decode = result.asr_raw_metrics.decode_inference_durations;
    const auto& right_decode = right.asr_raw_metrics.decode_inference_durations;
    result_decode.insert(result_decode.end(), right_decode.begin(), right_decode.end());

    return result;
}

ASRPerfMetrics& ASRPerfMetrics::operator+=(const ASRPerfMetrics& right) {
    *this = *this + right;
    return *this;
}

}  // namespace ov::genai
