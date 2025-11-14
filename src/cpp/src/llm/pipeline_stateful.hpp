// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <limits>

#include "llm/pipeline_base.hpp"
#include "lm_encoding.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"

namespace ov::genai {

class StatefulLLMPipeline final : public LLMPipelineImplBase {
    ov::InferRequest m_model_runner;
    Sampler m_sampler;

    // Chat scenario specific parameters
    bool is_chat_conversation = false;
    ChatHistory m_history;
    std::vector<int64_t> m_tokenized_chat_history;
    ov::genai::utils::GenerationChatInputsType m_chat_input_type = ov::genai::utils::GenerationChatInputsType::UNDEF;
    // Finish reason of last generation for chat scenario
    ov::genai::GenerationStatus m_chat_generation_finish_status = ov::genai::GenerationStatus::RUNNING;
    // if True, full history will be used as prompt on each chat generation
    bool m_use_full_chat_history = false;
    size_t m_max_prompt_len = std::numeric_limits<size_t>::max();
    size_t m_max_kv_cache_size = std::numeric_limits<size_t>::max();
    bool m_is_npu = false;
    // include reflection of tokens contained in the kv cache and amount of tokens, which are needed to trim from kv cache on the next step of chat
    utils::KVCacheState m_kv_cache_state;

    void reset_kv_state();
public:

    StatefulLLMPipeline(
        const ov::InferRequest& request,
        const ov::genai::Tokenizer& tokenizer,
        OptionalGenerationConfig generation_config = std::nullopt
    );

    StatefulLLMPipeline(
        const std::filesystem::path& models_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& plugin_config
    );

    StatefulLLMPipeline(
        const std::shared_ptr<ov::Model>& model,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& config,
        const ov::genai::GenerationConfig& generation_config
    );

    StatefulLLMPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& plugin_config
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

    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) const;

    DecodedResults get_decoded_results(
        TokenizedInputs encoded_input,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer,
        std::chrono::steady_clock::time_point start_time
    );

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    ~StatefulLLMPipeline();
};

} // namespace ov::genai
