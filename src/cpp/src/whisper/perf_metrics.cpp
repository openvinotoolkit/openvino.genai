// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/whisper_pipeline.hpp"

namespace ov {
namespace genai {

MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

MeanStdPair WhisperPerfMetrics::get_features_extraction_duration() {
    evaluate_statistics();
    return features_extraction_duration;
}

MeanStdPair WhisperPerfMetrics::get_word_level_timestamps_processing_duration() {
    evaluate_statistics();
    return word_level_timestamps_processing_duration;
}

MeanStdPair WhisperPerfMetrics::get_encode_inference_duration() {
    evaluate_statistics();
    return encode_inference_duration;
}

MeanStdPair WhisperPerfMetrics::get_decode_inference_duration() {
    evaluate_statistics();
    return decode_inference_duration;
}

void WhisperPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    features_extraction_duration = ov::genai::calc_mean_and_std(whisper_raw_metrics.features_extraction_durations);

    word_level_timestamps_processing_duration =
        ov::genai::calc_mean_and_std(whisper_raw_metrics.word_level_timestamps_processing_durations);

    encode_inference_duration = ov::genai::calc_mean_and_std(whisper_raw_metrics.encode_inference_durations);

    decode_inference_duration = ov::genai::calc_mean_and_std(whisper_raw_metrics.decode_inference_durations);

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

    auto& result_word_level_timestamps_processing_durations =
        result.whisper_raw_metrics.word_level_timestamps_processing_durations;
    auto& right_word_level_timestamps_processing_durations =
        right.whisper_raw_metrics.word_level_timestamps_processing_durations;
    result_word_level_timestamps_processing_durations.insert(result_word_level_timestamps_processing_durations.end(),
                                                             right_word_level_timestamps_processing_durations.begin(),
                                                             right_word_level_timestamps_processing_durations.end());

    auto& result_encode_inference_durations = result.whisper_raw_metrics.encode_inference_durations;
    auto& right_encode_inference_durations = right.whisper_raw_metrics.encode_inference_durations;
    result_encode_inference_durations.insert(result_encode_inference_durations.end(),
                                            right_encode_inference_durations.begin(),
                                            right_encode_inference_durations.end());

    auto& result_decode_inference_durations = result.whisper_raw_metrics.decode_inference_durations;
    auto& right_decode_inference_durations = right.whisper_raw_metrics.decode_inference_durations;
    result_decode_inference_durations.insert(result_decode_inference_durations.end(),
                                            right_decode_inference_durations.begin(),
                                            right_decode_inference_durations.end());
    return result;
}

WhisperPerfMetrics& WhisperPerfMetrics::operator+=(const WhisperPerfMetrics& right) {
    *this = *this + right;
    return *this;
}

}  // namespace genai
}  // namespace ov
