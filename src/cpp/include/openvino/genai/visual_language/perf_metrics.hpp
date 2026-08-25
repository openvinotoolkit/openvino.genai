// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS VLMRawPerfMetrics {
    /** @brief Duration of preparation of embeddings */
    std::vector<MicroSeconds> prepare_embeddings_durations;
    /** @brief Duration of vision encoding */
    std::vector<MicroSeconds> vision_encoding_durations;
    /** @brief Duration of audio encoding */
    std::vector<MicroSeconds> audio_encoding_durations;
    /** @brief Duration of text embedding */
    std::vector<MicroSeconds> text_embedding_durations;
    /** @brief Number of image slices produced for each input image */
    std::vector<size_t> per_image_slice_counts;
};

struct OPENVINO_GENAI_EXPORTS VLMPerfMetrics : public PerfMetrics {
    /** @brief Mean and standard deviation of preparation of embeddings in milliseconds */
    MeanStdPair prepare_embeddings_duration;
    /** @brief Mean and standard deviation of vision encoding in milliseconds */
    MeanStdPair vision_encoding_duration;
    /** @brief Mean and standard deviation of audio encoding in milliseconds */
    MeanStdPair audio_encoding_duration;
    /** @brief Mean and standard deviation of text embedding in milliseconds */
    MeanStdPair text_embedding_duration;
    /** @brief Total number of image slices produced for the request */
    size_t total_image_slice_count = 0;

    MeanStdPair get_prepare_embeddings_duration();
    MeanStdPair get_vision_encoding_duration();
    MeanStdPair get_audio_encoding_duration();
    MeanStdPair get_text_embedding_duration();
    size_t get_total_image_slice_count();

    VLMPerfMetrics() = default;

    explicit VLMPerfMetrics(const PerfMetrics& perf_metrics)
        : PerfMetrics(perf_metrics),
          prepare_embeddings_duration(),
          total_image_slice_count(0){};

    explicit VLMPerfMetrics(PerfMetrics&& perf_metrics) noexcept
        : PerfMetrics(std::move(perf_metrics)),
          prepare_embeddings_duration(),
          total_image_slice_count(0){};

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;

    VLMPerfMetrics operator+(const VLMPerfMetrics& metrics) const;

    VLMRawPerfMetrics vlm_raw_metrics;
};

}  // namespace ov::genai
