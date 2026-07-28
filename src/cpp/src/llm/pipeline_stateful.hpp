// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


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
    bool m_is_npu = false;
    // include reflection of tokens contained in the kv cache and amount of tokens, which are needed to trim from kv cache on the next step of chat
    utils::CacheState m_cache_state;

    // True when the NPU plugin advertises continuous prefill for this compiled model.
    // Chat turns then negotiate a keep through the npuw_stored_tokens_state variable
    // state and send only the delta instead of resending the full history.
    bool m_npu_continuous_prefill = false;
    // Past KV capacity of the largest generate variant, used to validate the response
    // budget before proposing.
    size_t m_kv_cache_capacity = std::numeric_limits<size_t>::max();
    // Set after a failed turn. The next attempt skips the proposal and sends the full
    // history, which satisfies the plugin's pending reset.
    bool m_forced_full_prefill = false;
    // Guards against nested arming when generate(EncodedInputs) runs inside the
    // string or ChatHistory entry points.
    bool m_npu_turn_guard_active = false;

    void reset_state();

    // Reads the continuous prefill capability from the compiled model and flips
    // m_use_full_chat_history accordingly. Called from every construction path.
    void init_npu_continuous_prefill(ov::CompiledModel& compiled_model);
    // Proposes the post-alignment common prefix to the plugin, reads the grant back
    // and resizes the cache state to it, so slicing happens at the granted value.
    void negotiate_npu_history_reuse(size_t full_history_len, const GenerationConfig& config);
    // Single recovery path for any failure between proposal and inference. Resets the
    // plugin channel, restores the pre-turn history and forces a full prefill next.
    void on_npu_turn_failure(ChatHistory history_snapshot, std::vector<int64_t> tokenized_history_snapshot);

    // Restores conversation state when a turn throws anywhere between the proposal
    // and the end of inference. Armed only at the outermost generate entry point.
    class NPUTurnGuard {
    public:
        NPUTurnGuard(StatefulLLMPipeline& pipeline) : m_pipeline(pipeline) {
            if (pipeline.m_npu_continuous_prefill && pipeline.is_chat_conversation &&
                !pipeline.m_npu_turn_guard_active) {
                m_armed = true;
                pipeline.m_npu_turn_guard_active = true;
                // ChatHistory copies share their JsonContainer storage, so the
                // snapshot needs explicit deep copies to survive in-place mutation.
                m_history_snapshot = ChatHistory(pipeline.m_history.get_messages().copy());
                m_history_snapshot.set_tools(pipeline.m_history.get_tools().copy());
                m_history_snapshot.set_extra_context(pipeline.m_history.get_extra_context().copy());
                m_tokenized_history_snapshot = pipeline.m_tokenized_chat_history;
            }
        }
        void disarm() {
            if (m_armed) {
                m_pipeline.m_npu_turn_guard_active = false;
                m_armed = false;
            }
        }
        // Runs during stack unwinding, so the rollback must never throw out of here.
        ~NPUTurnGuard() {
            if (m_armed) {
                m_pipeline.m_npu_turn_guard_active = false;
                try {
                    m_pipeline.on_npu_turn_failure(std::move(m_history_snapshot),
                                                   std::move(m_tokenized_history_snapshot));
                } catch (...) {
                }
            }
        }
    private:
        StatefulLLMPipeline& m_pipeline;
        bool m_armed = false;
        ChatHistory m_history_snapshot;
        std::vector<int64_t> m_tokenized_history_snapshot;
    };
    friend class NPUTurnGuard;
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
        std::chrono::steady_clock::time_point start_time,
        std::chrono::steady_clock::time_point tokenization_start_time,
        std::optional<float> chat_template_duration_us = std::nullopt
    );

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    ~StatefulLLMPipeline();
};

} // namespace ov::genai
