// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llm/pipeline_base.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"

namespace ov {
namespace genai {

/**
 * @brief Base class for stateful speculative decoding pipelines
 * 
 * Provides common functionality for Fast Draft and Eagle3 pipelines including:
 * - Chat history management
 * - Token tokenization/detokenization with timing
 * - Performance metrics collection and aggregation
 * - Common generate() method implementations
 */
class StatefulSpeculativePipelineBase : public LLMPipelineImplBase {
public:
    /** @brief Default number of assistant tokens for speculative decoding. */
    static constexpr std::size_t DEFAULT_NUM_ASSISTANT_TOKENS = 5;

protected:
    StatefulSpeculativePipelineBase(const Tokenizer& tokenizer, const GenerationConfig& generation_config);

public:
    virtual ~StatefulSpeculativePipelineBase() = default;

    // LLMPipelineImplBase interface implementations
    DecodedResults generate(StringInputs inputs,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) override;

    DecodedResults generate(const ChatHistory& history,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) override;

    EncodedResults generate(const EncodedInputs& inputs,
                            OptionalGenerationConfig generation_config,
                            StreamerVariant streamer) override;

    void start_chat(const std::string& system_message) override;
    void finish_chat() override;

protected:
    /**
     * @brief Ensures num_assistant_tokens is set and validates generation config.
     *
     * This function validates that assistant_confidence_threshold is 0 (unsupported in stateful mode)
     * and sets num_assistant_tokens to DEFAULT_NUM_ASSISTANT_TOKENS if not already specified.
     *
     * @param config Generation configuration to validate and potentially modify.
     * @throws Exception if assistant_confidence_threshold is non-zero.
     */
    static void ensure_num_assistant_tokens_is_set(GenerationConfig& config);

    /**
     * @brief Resolve and validate generation configuration
     * 
     * Child classes should override this to apply their specific defaults
     * and validations (e.g., num_assistant_tokens handling).
     * 
     * @param generation_config Optional config from user
     * @return Fully resolved GenerationConfig
     */
    virtual GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config);

    /**
     * @brief Core token generation logic - must be implemented by child classes
     * 
     * @param inputs Tokenized input (ov::Tensor or TokenizedInputs)
     * @param config Resolved generation configuration
     * @param streamer Optional streamer for real-time output
     * @return Encoded results with tokens and performance metrics
     */
    virtual EncodedResults generate_tokens(const EncodedInputs& inputs,
                                          const GenerationConfig& config,
                                          StreamerVariant streamer) = 0;

    /**
     * @brief Tokenize prompt with timing measurement
     * 
     * Handles chat mode, chat templates, and special tokens correctly.
     * Updates m_raw_perf_metrics with tokenization duration.
     * 
     * @param prompt Raw text prompt
     * @param config Generation configuration (for chat template settings)
     * @return Tokenized inputs ready for model inference
     */
    TokenizedInputs tokenize(const std::string& prompt, const GenerationConfig& config);

    /**
     * @brief Detokenize tokens with timing measurement
     * 
     * Updates m_raw_perf_metrics with detokenization duration.
     * 
     * @param tokens Token IDs to decode
     * @return Decoded text
     */
    std::vector<std::string> detokenize(const std::vector<std::vector<int64_t>>& tokens);

    /**
     * @brief Update decoded results with final performance metrics
     *
     * Updates the decoded results with metrics from encoded_results (model-level),
     * m_sd_perf_metrics (tokenization/detokenization), and outer timing.
     *
     * @param decoded_results DecodedResults to update
     * @param encoded_results EncodedResults containing model-level metrics
     * @param generate_duration_us Total generation duration in microseconds
     * @param generate_start_time Start time of generation for statistics evaluation
     */
    void update_decoded_results_with_perf_metrics(DecodedResults& decoded_results,
                                                  const EncodedResults& encoded_results,
                                                  float generate_duration_us,
                                                  TimePoint generate_start_time);

protected:
    // Chat state management
    bool m_is_chat_active = false;
    ChatHistory m_chat_history;
    bool m_streaming_was_cancelled = false;

    // Performance metrics (to be populated by child classes)
    SDPerModelsPerfMetrics m_sd_perf_metrics;
    SpeculativeDecodingMetrics m_sd_metrics;
};

}  // namespace genai
}  // namespace ov
