// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

#include "automatic_speech_recognition/audios_view.hpp"
#include "context_tokens.hpp"
#include "models/decoder.hpp"
#include "openvino/genai/tokenizer.hpp"
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

struct SotTokensResult {
    std::vector<int64_t> tokens;
    std::string language;  // language string, e.g. "en"
};

struct WhisperGenerateResult {
    std::vector<int64_t> output_tokens;
    std::string language;
    std::optional<std::vector<Segment>> segments = std::nullopt;
    std::optional<std::vector<WhisperWordTiming>> words = std::nullopt;
    WhisperPerfMetrics perf_metrics;
};

/**
 * Generates one result per audio, in input order. A single audio is decoded with the scalar decoder, which supports
 * beam search, sampling, streaming and word timestamps. Several audios are decoded together greedily with a decoder
 * that supports_dynamic_batch(), without a streamer or word timestamps.
 */
std::vector<WhisperGenerateResult> whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                                    const ov::genai::WhisperConfig& model_config,
                                                    const WhisperContextTokens& context_tokens,
                                                    const AudiosView& audios,
                                                    ov::InferRequest& encoder,
                                                    const std::shared_ptr<WhisperDecoder>& decoder,
                                                    WhisperFeatureExtractor& feature_extractor,
                                                    const std::shared_ptr<StreamerBase>& streamer,
                                                    Sampler& sampler,
                                                    Tokenizer& tokenizer);

}  // namespace genai
}  // namespace ov
