// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

struct WhisperFeatures {
    size_t feature_size;

    // total frames extracted from the audio including padding
    size_t n_frames;

    // active frames corresponding to the original audio length
    size_t n_active_frames;

    // flattened 2d array with shape [feature_size, n_frames]
    std::vector<float> data;

    /**
     * Return frames with specific offset
     * Pad to min_frames if needed
     *
     *     v offset
     * ****xxxxx****
     * ****xxxxx****
     * ****xxxxx****
     *
     */
    std::vector<float> get_data_with_offset(const size_t frame_offset, const size_t min_frames);
};

/// HuggingFace WhisperFeatureExtractor implementation: log-mel spectrogram
/// with Hann windowing, slaney mel filter bank, max-8 clamp and (x+4)/4
/// normalization. Output is a flat [feature_size, n_frames] row-major buffer
/// wrapped in WhisperFeatures.
///
/// Used by both Whisper and Qwen3-Omni audio preprocessing — both pipelines
/// declare `"feature_extractor_type": "WhisperFeatureExtractor"` in their
/// HuggingFace preprocessor configs and share this single implementation
/// with model-specific config values.
class WhisperFeatureExtractor {
public:
    size_t feature_size = 80;
    size_t sampling_rate = 16000;
    size_t hop_length = 160;
    size_t n_fft = 400;
    size_t chunk_length = 30;
    size_t n_samples = 480000;
    size_t nb_max_frames = 3000;

    /// Load configuration from a HuggingFace `preprocessor_config.json`. Missing
    /// fields fall back to the Whisper defaults declared above. After load,
    /// `n_samples` is reset to `sampling_rate * chunk_length` so the extractor
    /// pre-pads to the configured Whisper chunk length (default 30 s = 480 000
    /// samples at 16 kHz).
    explicit WhisperFeatureExtractor(const std::filesystem::path& preprocessor_json_path);

    /// Construct with explicit numeric configuration. Used by pipelines that
    /// declare `WhisperFeatureExtractor` in their preprocessor config but do
    /// not fix a `chunk_length` (e.g. Qwen3-Omni). `n_samples` is set to 0,
    /// disabling pre-padding so the produced frame count tracks the raw
    /// signal length.
    WhisperFeatureExtractor(size_t feature_size, size_t sampling_rate, size_t n_fft, size_t hop_length);

    /// Compute the log-mel spectrogram for `raw_speech`. `raw_speech` must have
    /// size > n_fft/2 to satisfy the reflect-padding window.
    WhisperFeatures extract(const std::vector<float>& raw_speech);

private:
    void init_parameters(const std::filesystem::path& preprocessor_json_path);
    void rebuild_tables();

    std::vector<float> pad_with_reflect(const std::vector<float>& raw_speech) const;
    std::vector<float> compute_mel(const std::vector<float>& padded, size_t n_samples_real, size_t n_frames) const;

    std::vector<float> m_sin_vals;
    std::vector<float> m_cos_vals;
    std::vector<float> m_mel_filter;
};

}  // namespace genai
}  // namespace ov
