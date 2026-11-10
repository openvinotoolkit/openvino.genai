// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "automatic_speech_recognition/audios_view.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/whisper_utils.hpp"

namespace ov {
namespace genai {

class WhisperPipelineImplBase {
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

    // Returns one result per audio, in input order. The audios are read only during this call.
    virtual std::vector<WhisperDecodedResults> generate(const AudiosView& audios,
                                                        OptionalWhisperGenerationConfig generation_config,
                                                        const std::shared_ptr<StreamerBase> streamer) = 0;

    void set_generation_config(const WhisperGenerationConfig& config) {
        m_generation_config = utils::prepare_per_generate_config(m_generation_config, config);
    }

    virtual ~WhisperPipelineImplBase() = default;
};

// Does not measure load time; callers report their own.
std::unique_ptr<WhisperPipelineImplBase> create_whisper_pipeline_impl(const std::filesystem::path& models_path,
                                                                      const std::string& device,
                                                                      const ov::AnyMap& properties);

}  // namespace genai
}  // namespace ov
