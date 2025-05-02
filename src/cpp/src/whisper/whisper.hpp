// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "context_tokens.hpp"
#include "models/decoder.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "sampling/sampler.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/models.hpp"

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

WhisperGenerateResult whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                       const ov::genai::WhisperConfig& model_config,
                                       const WhisperContextTokens& context_tokens,
                                       const RawSpeechInput& raw_speech,
                                       ov::InferRequest& encoder,
                                       std::shared_ptr<WhisperDecoder> decoder,
                                       WhisperFeatureExtractor& feature_extractor,
                                       const std::shared_ptr<StreamerBase> streamer,
                                       Sampler& sampler);

}  // namespace genai
}  // namespace ov
