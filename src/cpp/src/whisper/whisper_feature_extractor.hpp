// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

class WhisperFeatureExtractor {
public:
    static constexpr size_t feature_size = 80;
    static constexpr size_t sampling_rate = 16000;
    static constexpr size_t hop_length = 160;
    static constexpr size_t n_fft = 400;
    static constexpr size_t chunk_length = 30;
    static constexpr size_t chunk_size = sampling_rate * chunk_length;

    WhisperFeatureExtractor();

    /**
     * @brief Create 2d log-mel spectrogram from raw speech data
     *
     * @see [huggingface introduction to audio
     * data](https://huggingface.co/learn/audio-course/chapter1/audio_data#mel-spectrogram)
     */
    std::vector<float> extract(const std::vector<float>& raw_speech);

private:
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
    std::vector<float> mel_filter;
};

}  // namespace genai
}  // namespace ov
