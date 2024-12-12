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

struct WhisperGenerateResult {
    std::vector<int64_t> output_tokens;
    std::optional<std::vector<Segment>> segments = std::nullopt;
    WhisperPerfMetrics perf_metrics;
};

struct WhisperContextTokens {
    std::vector<int64_t> initial_prompt;
    std::vector<int64_t> hotwords;
};

WhisperGenerateResult whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                       const ov::genai::WhisperConfig& model_config,
                                       const WhisperContextTokens& context_tokens,
                                       const ov::genai::RawSpeechInput& raw_speech,
                                       ov::genai::WhisperInitializedModels& models,
                                       ov::genai::WhisperFeatureExtractor& feature_extractor,
                                       const std::shared_ptr<ChunkStreamerBase> streamer);

}  // namespace genai
}  // namespace ov
