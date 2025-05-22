// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/whisper_pipeline.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov {
namespace genai {

class WhisperPipeline::WhisperPipelineImplBase {
public:
    WhisperGenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    WhisperFeatureExtractor m_feature_extractor;
    WhisperConfig m_model_config;

    float m_load_time_ms = 0;

    WhisperPipelineImplBase(const std::filesystem::path& models_path)
        : m_generation_config(utils::from_config_json_if_exists<WhisperGenerationConfig>(models_path)),
          m_tokenizer{models_path},
          m_feature_extractor{models_path / "preprocessor_config.json"},
          m_model_config{models_path / "config.json"} {}

    virtual WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                           OptionalWhisperGenerationConfig generation_config,
                                           const std::shared_ptr<StreamerBase> streamer) = 0;

    virtual ~WhisperPipelineImplBase() = default;
};

}  // namespace genai
}  // namespace ov
