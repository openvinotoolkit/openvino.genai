// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

struct Gemma4AudioFeatures {
    ov::Tensor input_features;
    ov::Tensor input_features_mask;
};

class Gemma4AudioFeatureExtractor {
public:
    Gemma4AudioFeatureExtractor();
    explicit Gemma4AudioFeatureExtractor(const std::filesystem::path& config_dir_path);

    Gemma4AudioFeatures extract(const ov::Tensor& audio) const;

private:
    size_t m_sampling_rate = 16'000;
    size_t m_feature_size = 128;
    size_t m_frame_length = 320;
    size_t m_hop_length = 160;
    size_t m_fft_length = 512;
    double m_min_frequency = 0.0;
    double m_max_frequency = 8'000.0;
    double m_mel_floor = 0.001;
    std::vector<double> m_mel_filters;

    void validate_parameters() const;
    void validate_audio_input(const ov::Tensor& audio) const;
};

}  // namespace ov::genai
