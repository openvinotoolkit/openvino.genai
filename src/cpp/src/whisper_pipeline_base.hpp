// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/whisper_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"

#include "whisper/whisper_feature_extractor.hpp"

namespace ov {
namespace genai {

class WhisperPipelineImplBase {
public:
    WhisperPipelineImplBase(const WhisperGenerationConfig& config,
                            const Tokenizer& tokenizer,
                            const WhisperFeatureExtractor& feature_extractor)
    : m_generation_config(config), m_tokenizer(tokenizer), m_feature_extractor(feature_extractor) {
    }

    virtual DecodedResults generate(const RawSpeechInput& raw_speech_input,
                                    OptionalWhisperGenerationConfig generation_config,
                                    StreamerVariant streamer) = 0;

    virtual ~WhisperPipelineImplBase() = default;

    WhisperGenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    WhisperFeatureExtractor m_feature_extractor;

    float m_load_time_ms = 0;
};

}  // namespace genai
}  // namespace ov
