// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "automatic_speech_recognition/models/fun-asr/feature_extractor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * Extracts input features for SenseVoiceSmall.
 *
 * Reuses FunASRFeatureExtractor for Kaldi-compatible filter-bank extraction and
 * low-frame-rate (LFR) stacking, then applies the CMVN statistics loaded from
 * `am.mvn` to the resulting 560-dimensional feature vectors:
 *
 *     normalized = (feature + shift) * scale
 *
 * The output is a float32 tensor of shape [1, T, feature_size].
 */
class SenseVoiceSmallFeatureExtractor {
public:
    static constexpr size_t feature_size = FunASRFeatureExtractor::mel_bins * FunASRFeatureExtractor::lfr_window;

    explicit SenseVoiceSmallFeatureExtractor(const std::filesystem::path& cmvn_file);

    ov::Tensor extract(const std::vector<float>& audio) const;

private:
    FunASRFeatureExtractor m_fbank_lfr;
    std::vector<float> m_shifts;
    std::vector<float> m_scales;
};

}  // namespace ov::genai
