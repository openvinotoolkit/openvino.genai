// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "utils.hpp"

namespace ov::genai {

void erase_allowed_asr_ctor_properties(ov::AnyMap& properties);

using AudiosView = std::vector<std::reference_wrapper<const std::vector<float>>>;

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

    virtual ASRDecodedResults generate(const AudiosView& audios,
                                       const std::optional<ASRGenerationConfig>& generation_config,
                                       const std::shared_ptr<StreamerBase> streamer) = 0;

    virtual void set_generation_config(const ASRGenerationConfig& incoming_config) {
        ASRGenerationConfig config = incoming_config;
        if (config.stop_token_ids.empty()) {
            config.stop_token_ids = m_generation_config.stop_token_ids;
        }
        if (config.eos_token_id == -1) {
            config.set_eos_token_id(m_generation_config.eos_token_id);
        }
        config.validate();
        m_generation_config = std::move(config);
    }

    virtual ~ASRPipelineImplBase() = default;

protected:
    static const std::vector<float>& get_single_audio(const AudiosView& audios) {
        OPENVINO_ASSERT(audios.size() == 1,
                        "This ASR model supports only one audio per generate, but received ",
                        audios.size(),
                        ".");
        return audios.front();
    }
};

}  // namespace ov::genai
