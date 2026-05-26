// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "utils.hpp"

namespace ov::genai {

class ASRPipelineImplBase {
public:
    ASRGenerationConfig m_generation_config;
    Tokenizer m_tokenizer;

    ASRPipelineImplBase(const std::filesystem::path& models_path)
        : m_generation_config(utils::from_config_json_if_exists<ASRGenerationConfig>(models_path)),
          m_tokenizer{models_path} {}

    virtual ASRDecodedResults generate(const RawSpeechInput& raw_speech_input,
                                       std::optional<ASRGenerationConfig> generation_config,
                                       const std::shared_ptr<StreamerBase> streamer) = 0;

    virtual ~ASRPipelineImplBase() = default;
};

}  // namespace ov::genai
