// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/whisper_pipeline.hpp"
#include "whisper/whisper_config.hpp"
#include "whisper/whisper_feature_extractor.hpp"

namespace {
ov::genai::WhisperGenerationConfig from_config_json_if_exists(const std::filesystem::path& model_path) {
    auto config_file_path = model_path / "generation_config.json";
    if (std::filesystem::exists(config_file_path)) {
        return ov::genai::WhisperGenerationConfig((config_file_path).string());
    } else {
        return ov::genai::WhisperGenerationConfig{};
    }
}
}  // namespace

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
        : m_generation_config(from_config_json_if_exists(models_path)),
          m_tokenizer{models_path},
          m_feature_extractor{(models_path / "preprocessor_config.json").string()},
          m_model_config{(models_path / "config.json").string()} {}

    virtual WhisperDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                           OptionalWhisperGenerationConfig generation_config,
                                           ChunkStreamerVariant streamer) = 0;

    virtual ~WhisperPipelineImplBase() = default;
};

}  // namespace genai
}  // namespace ov
