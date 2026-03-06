// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <vector>

#include "openvino/runtime/tensor.hpp"

#include "whisper/feature_extractor.hpp"


namespace ov::genai::module {

struct WhisperFeatureExtractorOutput {
    ov::Tensor input_features;
    std::optional<ov::Tensor> attention_mask;
    size_t num_frames = 0;
};

class WhisperFeatureExtractor {
public:
    WhisperFeatureExtractor() = delete;
    explicit WhisperFeatureExtractor(const std::filesystem::path& model_path);

    // Pads/truncates to the model's configured maximum length (typically 30s @ 16kHz)
    // and returns log-mel filterbank features with shape [feature_size, n_frames].
    WhisperFeatureExtractorOutput extract(const ov::Tensor& raw_speech,
                                         std::optional<size_t> sampling_rate = std::nullopt,
                                         bool return_attention_mask = false,
                                         float dither = 0.0f);

    size_t feature_size() const noexcept;
    size_t sampling_rate() const noexcept;
    size_t hop_length() const noexcept;
    size_t n_fft() const noexcept;
    size_t n_samples() const noexcept;

private:
    ov::genai::WhisperFeatureExtractor m_impl;
};

} // namespace ov::genai::module


