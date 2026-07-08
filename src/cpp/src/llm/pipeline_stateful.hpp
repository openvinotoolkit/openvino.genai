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
    // Set from the model's config.json ("rope_parameters") when it uses LongRoPE - the
    // 0-based sequence position at which the model switches from its "short" to "long"
    // RoPE factor table (== "original_max_position_embeddings"). std::nullopt if the model
    // doesn't use LongRoPE or config.json wasn't available to inspect.
    std::optional<size_t> m_longrope_threshold;
    // Transient, single-use: set by the StringInputs/ChatHistory generate() overloads right
    // before delegating to the EncodedInputs overload, when this specific turn's cumulative
    // history was detected to cross m_longrope_threshold. Consumed (read then cleared) at the
    // top of the EncodedInputs overload - unlike m_use_full_chat_history (permanent, NPU-only)
    // this only forces a full-history resend + cache reset for the ONE turn that actually
    // crosses the threshold, so CPU/GPU keep using incremental caching otherwise.
    bool m_force_longrope_reprefill = false;
    // include reflection of tokens contained in the kv cache and amount of tokens, which are needed to trim from kv cache on the next step of chat
    utils::CacheState m_cache_state;

    void reset_state();
    // True if a chat turn whose cumulative token count is `new_total_len` would cross
    // m_longrope_threshold for the first time (cache is currently entirely below the
    // threshold, and this turn's total reaches/exceeds it). False if the model isn't
    // LongRoPE, or the threshold was already crossed by a previous turn (no NEW crossing -
    // incremental caching remains consistent once fully in one regime).
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
    // models_path) and, if so, records the short/long-factor switch-over threshold so that a
    // chat turn crossing it triggers a one-off full-history resend + cache reset (see
    // should_force_longrope_reprefill) and a mid-generation crossing within a turn triggers
    // the LongRopeThresholdStreamer mitigation. Unlike NPU's m_use_full_chat_history this does
    // NOT force full-history resend on every turn - incremental caching is kept otherwise.
    // No-op if models_path is empty, config.json is missing/unparseable, or the model doesn't
    // use LongRoPE. Needed because the real construction path (StatefulPipeline::create in
    // llm/pipeline.cpp) already has models_path available but doesn't go through the
    // models_path-taking constructor above.
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
