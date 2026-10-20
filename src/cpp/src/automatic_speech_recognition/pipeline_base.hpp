// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "utils.hpp"

namespace ov::genai {

void erase_allowed_asr_ctor_properties(ov::AnyMap& properties);

class ASRPipelineImplBase {
public:
    ASRGenerationConfig m_generation_config;
    Tokenizer m_tokenizer;
    float m_load_time_ms = 0.0f;

    ASRPipelineImplBase(const std::filesystem::path& models_path)
        : ASRPipelineImplBase(models_path, Tokenizer{models_path}) {}

    ASRPipelineImplBase(const std::filesystem::path& models_path, const Tokenizer& tokenizer)
        : m_generation_config(utils::from_config_json_if_exists<ASRGenerationConfig>(models_path)),
          m_tokenizer{tokenizer} {}

    virtual ASRDecodedResults generate(const AudioInputs& audio_inputs,
                                       const std::optional<ASRGenerationConfig>& generation_config,
                                       const std::shared_ptr<StreamerBase> streamer) = 0;

    virtual ASRDecodedResults generate_batch(const std::vector<std::vector<float>>& audio_batch,
                                             const std::optional<ASRGenerationConfig>& generation_config,
                                             const std::shared_ptr<StreamerBase> streamer) {
        OPENVINO_ASSERT(audio_batch.size() == 1,
                        "This ASR model supports only one utterance per call, but received ",
                        audio_batch.size(),
                        ".");
        return generate(AudioInputs{audio_batch.front()}, generation_config, streamer);
    }

    virtual void set_generation_config(const ASRGenerationConfig& config) {
        m_generation_config = config;
        m_generation_config.validate();
    }

    virtual ~ASRPipelineImplBase() = default;
};

}  // namespace ov::genai
