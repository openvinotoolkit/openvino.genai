// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

struct WhisperFeatures {
    size_t feature_size;
    size_t n_frames;

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

class WhisperFeatureExtractor {
public:
    size_t feature_size = 80;
    size_t sampling_rate = 16000;
    size_t hop_length = 160;
    size_t n_fft = 400;
    size_t chunk_length = 30;
    size_t n_samples = 480000;
    size_t nb_max_frames = 3000;

    explicit WhisperFeatureExtractor(const std::filesystem::path& preprocessor_json_path);

    /**
     * @brief Create a flattened 2d log-mel spectrogram [feature_size, n_frames] from raw speech data
     *
     * @see [huggingface introduction to audio
     * data](https://huggingface.co/learn/audio-course/chapter1/audio_data#mel-spectrogram)
     */
    WhisperFeatures extract(const std::vector<float>& raw_speech);

private:
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
    std::vector<float> mel_filter;

    void init_mel_filter();
    void init_parameters(const std::filesystem::path& preprocessor_json_path);
};

}  // namespace genai
}  // namespace ov
