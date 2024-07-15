// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llm_pipeline_base.hpp"

namespace ov {
namespace genai {

class StaticLLMPipeline final : public LLMPipelineImplBase {
public:
    StaticLLMPipeline(
        const std::filesystem::path& path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& config
    );

    StaticLLMPipeline(
        const std::filesystem::path& path,
        const std::string& device,
        const ov::AnyMap& config
    );

    DecodedResults generate(
        StringInputs inputs,
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
private:
    void prepare_for_new_conversation();

private:
    struct KVCacheDesc {
        uint32_t total_size;
        uint32_t num_stored_tokens;
    };

    KVCacheDesc m_kvcache_desc;
    ov::InferRequest m_kvcache_request;
    ov::InferRequest m_prefill_request;

    bool m_is_chat_conversation = false;
    ChatHistory m_history;
};

}  // namespace genai
}  // namespace ov
