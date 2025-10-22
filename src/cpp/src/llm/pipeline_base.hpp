// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/streamer_base.hpp"

namespace ov {
namespace genai {

class LLMPipelineImplBase {
public:
    LLMPipelineImplBase(const Tokenizer& tokenizer,
                        const GenerationConfig& config)
    : m_tokenizer(tokenizer), m_generation_config(config) { }

    Tokenizer get_tokenizer() {
        return m_tokenizer;
    }

    GenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(GenerationConfig config) {
        int64_t default_eos_token_id = m_generation_config.eos_token_id;
        auto default_stop_token_ids = m_generation_config.stop_token_ids;
        m_generation_config = config;

        // If stop_token_ids were not provided, take value from default config
        if (m_generation_config.stop_token_ids.empty())
            m_generation_config.stop_token_ids = default_stop_token_ids;
        // if eos_token_id was not provided in config forward from default config
        if (m_generation_config.eos_token_id == -1)
            m_generation_config.set_eos_token_id(default_eos_token_id);

        m_generation_config.validate();
    }

    virtual DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) = 0;

    virtual DecodedResults generate(
        const ChatHistory& history,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) = 0;

    virtual EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) = 0;

    virtual void start_chat(const std::string& system_message) = 0;
    virtual void finish_chat() = 0;

    virtual ~LLMPipelineImplBase() = default;

    void save_load_time(std::chrono::steady_clock::time_point start_time) {
        auto stop_time = std::chrono::steady_clock::now();
        m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
    }

protected:
    Tokenizer m_tokenizer;
    GenerationConfig m_generation_config;
    std::optional<AdapterController> m_adapter_controller;

    float m_load_time_ms = 0.0f;
};

}  // namespace genai
}  // namespace ov
