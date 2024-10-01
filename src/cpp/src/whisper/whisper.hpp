// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "whisper_config.hpp"
#include "whisper_feature_extractor.hpp"
#include "whisper_models.hpp"

namespace ov {
namespace genai {

struct Segment {
    float m_start;
    float m_end;
    std::vector<int64_t> m_tokens;
};

std::pair<std::vector<int64_t>, std::optional<std::vector<Segment>>> whisper_generate(
    const ov::genai::WhisperGenerationConfig& config,
    const ov::genai::WhisperConfig& model_config,
    const ov::genai::RawSpeechInput& raw_speech,
    ov::genai::WhisperInitializedModels& models,
    ov::genai::WhisperFeatureExtractor& feature_extractor,
    const std::shared_ptr<StreamerBase> streamer);

}  // namespace genai
}  // namespace ov
