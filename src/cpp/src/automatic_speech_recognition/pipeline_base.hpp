// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "streaming_session_impl_base.hpp"
#include "utils.hpp"

namespace ov::genai {

void erase_allowed_asr_ctor_properties(ov::AnyMap& properties);

class ASRPipelineImplBase {
public:
    ASRGenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    float m_load_time_ms = 0.0f;

    ASRPipelineImplBase(const std::filesystem::path& models_path)
        : m_generation_config(utils::from_config_json_if_exists<ASRGenerationConfig>(models_path)),
          m_tokenizer{models_path} {}

    virtual ASRDecodedResults generate(const AudioInputs& audio_inputs,
                                       const std::optional<ASRGenerationConfig>& generation_config,
                                       const std::shared_ptr<StreamerBase> streamer) = 0;

    virtual std::unique_ptr<ASRStreamingSession::Impl> create_streaming_session_impl(
        const ASRStreamingConfig& streaming_config,
        const ASRGenerationConfig& generation_config,
        ASRPartialResultCallback callback) {
        OPENVINO_THROW("Streaming is not supported for this ASR model type");
    }

    virtual void set_generation_config(const ASRGenerationConfig& config) {
        m_generation_config = config;
        m_generation_config.validate();
    }

    virtual ~ASRPipelineImplBase() = default;
};

}  // namespace ov::genai
