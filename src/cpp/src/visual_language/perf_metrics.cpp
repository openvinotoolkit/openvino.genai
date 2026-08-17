// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/perf_metrics.hpp"

namespace ov::genai {
MeanStdPair calc_mean_and_std(const std::vector<MicroSeconds>& durations);

MeanStdPair VLMPerfMetrics::get_prepare_embeddings_duration() {
    evaluate_statistics();
    return prepare_embeddings_duration;
}

MeanStdPair VLMPerfMetrics::get_vision_encoder_duration() {
    evaluate_statistics();
    return vision_encoder_duration;
}

MeanStdPair VLMPerfMetrics::get_text_embedding_duration() {
    evaluate_statistics();
    return text_embedding_duration;
}

size_t VLMPerfMetrics::get_total_image_slice_count() {
    evaluate_statistics();
    return total_image_slice_count;
}

void VLMPerfMetrics::evaluate_statistics(std::optional<TimePoint> start_time) {
    if (m_evaluated) {
        return;
    }

    prepare_embeddings_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.prepare_embeddings_durations);
    vision_encoder_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.vision_encoder_durations);
    text_embedding_duration = ov::genai::calc_mean_and_std(vlm_raw_metrics.text_embedding_durations);

    total_image_slice_count = 0;
    for (const auto count : vlm_raw_metrics.per_image_slice_counts) {
        total_image_slice_count += count;
    }
    PerfMetrics::evaluate_statistics(start_time);
};

VLMPerfMetrics VLMPerfMetrics::operator+(const VLMPerfMetrics& right) const {
    PerfMetrics base_result = PerfMetrics::operator+(right);
    VLMPerfMetrics result{std::move(base_result)};

    result.vlm_raw_metrics = vlm_raw_metrics;

    auto& result_prepare_embeddings_durations = result.vlm_raw_metrics.prepare_embeddings_durations;
    auto& right_prepare_embeddings_durations = right.vlm_raw_metrics.prepare_embeddings_durations;
    result_prepare_embeddings_durations.insert(
        result_prepare_embeddings_durations.end(),
        right_prepare_embeddings_durations.begin(),
        right_prepare_embeddings_durations.end()
    );

    auto& result_vision_encoder_durations = result.vlm_raw_metrics.vision_encoder_durations;
    auto& right_vision_encoder_durations = right.vlm_raw_metrics.vision_encoder_durations;
    result_vision_encoder_durations.insert(
        result_vision_encoder_durations.end(),
        right_vision_encoder_durations.begin(),
        right_vision_encoder_durations.end()
    );

    auto& result_text_embedding_durations = result.vlm_raw_metrics.text_embedding_durations;
    auto& right_text_embedding_durations = right.vlm_raw_metrics.text_embedding_durations;
    result_text_embedding_durations.insert(
        result_text_embedding_durations.end(),
        right_text_embedding_durations.begin(),
        right_text_embedding_durations.end()
    );

    auto& result_per_image_slice_counts = result.vlm_raw_metrics.per_image_slice_counts;
    const auto& right_per_image_slice_counts = right.vlm_raw_metrics.per_image_slice_counts;
    result_per_image_slice_counts.insert(
        result_per_image_slice_counts.end(),
        right_per_image_slice_counts.begin(),
        right_per_image_slice_counts.end()
    );
    return result;
}
}  // namespace ov::genai
