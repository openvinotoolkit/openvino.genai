// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "llm/pipeline_base.hpp"
#include "sampling/sampler.hpp"

namespace ov {
namespace genai {
namespace static_llm {

struct LLMPipelineFactory {
    static std::unique_ptr<LLMPipelineImplBase> create(const std::filesystem::path& models_path,
                                                       const ov::genai::Tokenizer& tokenizer,
                                                       const ov::AnyMap& config);

    static std::unique_ptr<LLMPipelineImplBase> create(const std::filesystem::path& models_path,
                                                       const ov::AnyMap& config);

    static std::unique_ptr<LLMPipelineImplBase> create(const std::shared_ptr<ov::Model>& model,
                                                       const ov::genai::Tokenizer& tokenizer,
                                                       const ov::AnyMap& properties,
                                                       const ov::genai::GenerationConfig& generation_config);
};

class StatefulLLMPipeline : public LLMPipelineImplBase {
public:
    StatefulLLMPipeline(
        const std::filesystem::path& path,
        const ov::genai::Tokenizer& tokenizer,
        const ov::AnyMap& config
    );

    StatefulLLMPipeline(
        const std::shared_ptr<ov::Model>& model,
        const ov::genai::Tokenizer& tokenizer,
        const ov::AnyMap& properties,
        const ov::genai::GenerationConfig& generation_config
    );

    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    DecodedResults generate(
        const ChatHistory& history,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    void start_chat(const std::string& system_message) override;
    void finish_chat() override;
    ~StatefulLLMPipeline();

private:
    uint32_t m_max_prompt_len = 0u;
    uint32_t m_kvcache_total = 0u;
    ov::InferRequest m_request;

    Sampler m_sampler;

    bool m_is_chat_conversation = false;
    ChatHistory m_history;
    ov::genai::GenerationStatus m_chat_generation_finish_status = ov::genai::GenerationStatus::RUNNING;
};

}  // namespace static_llm
}  // namespace genai
}  // namespace ov
