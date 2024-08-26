// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS WhisperFeatureExtractor {
public:
    static constexpr size_t feature_size = 80;
    static constexpr size_t sampling_rate = 16000;
    static constexpr size_t hop_length = 160;
    static constexpr size_t n_fft = 400;
    static constexpr size_t chunk_length = 30;
    static constexpr size_t chunk_size = sampling_rate * chunk_length;

    WhisperFeatureExtractor();

    std::vector<float> extract(const std::vector<float>& raw_speech);

private:
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
};

}  // namespace genai
}  // namespace ov
