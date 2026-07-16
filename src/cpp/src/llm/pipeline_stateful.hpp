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
    size_t m_max_kv_cache_size = std::numeric_limits<size_t>::max();
    bool m_is_npu = false;
    // LongRoPE mitigation, NPU only (see pipeline_stateful.cpp). The 0-based sequence
    // position at which the model switches from its "short" to "long" RoPE factor table
    // (model's "original_max_position_embeddings"). std::nullopt if the model doesn't use
    // LongRoPE or config.json wasn't available.
    std::optional<size_t> m_longrope_threshold;
    // Set by the StringInputs/ChatHistory generate() overloads before delegating to the
    // EncodedInputs overload, when the current turn's cumulative history crosses
    // m_longrope_threshold. Consumed (read then cleared) at the top of the EncodedInputs
    // overload.
    bool m_force_longrope_reprefill = false;
    // Set when a mid-generation LongRoPE threshold crossing happens to land on the very last
    // token allowed by max_new_tokens, so no continuation re-prefill runs within that same
    // call (see the generate(EncodedInputs...) crossing-handling block). The KV cache is left
    // holding that last token still encoded under the pre-crossing RoPE factor, so the very
    // next call for this conversation must force a reprefill regardless of what
    // should_force_longrope_reprefill()'s normal transition check would otherwise say.
    // Consumed (read then cleared) inside should_force_longrope_reprefill(). Must be reset to
    // false anywhere m_cache_state is invalidated/reset for an unrelated conversation (see
    // finish_chat() and the ChatHistory "not a continuation" branch) so it can never leak into
    // a different, later conversation.
    bool m_longrope_reprefill_pending = false;
    // include reflection of tokens contained in the kv cache and amount of tokens, which are needed to trim from kv cache on the next step of chat
    utils::CacheState m_cache_state;

    void reset_state();
    // True if a chat turn whose cumulative token count is `new_total_len` crosses
    // m_longrope_threshold for the first time. Always false on non-NPU devices.
    bool should_force_longrope_reprefill(size_t new_total_len);
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

    // Detects whether the model uses LongRoPE (from "rope_parameters" in config.json under
    // models_path) and, if so, records the switch-over threshold in m_longrope_threshold.
    // No-op (leaves m_longrope_threshold unset) if models_path is empty, config.json is
    // missing/unparseable, or the model doesn't use LongRoPE. Called wherever a
    // StatefulLLMPipeline is constructed with a models_path available: the
    // (models_path, tokenizer, device, properties) constructor above, and the
    // StatefulLLMPipeline construction sites in llm/pipeline.cpp.
    void set_longrope_threshold(const std::filesystem::path& models_path);

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
