// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/sensevoice/feature_extractor.hpp"

namespace ov::genai {

SenseVoiceSmallFeatureExtractor::SenseVoiceSmallFeatureExtractor(const std::filesystem::path& cmvn_file, float dither)
    : m_fbank_lfr(dither),
      m_cmvn(cmvn_file, feature_size) {}

ov::Tensor SenseVoiceSmallFeatureExtractor::extract(const std::vector<float>& audio) const {
    ov::Tensor features = m_fbank_lfr.extract(audio);
    const ov::Shape shape = features.get_shape();
    m_cmvn.apply(features.data<float>(), shape[0] * shape[1]);
    return features;
}

}  // namespace ov::genai
