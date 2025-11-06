// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "speculative_decoding_metrics.hpp"
#include "llm/pipeline_base.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"
#include <openvino/genai/perf_metrics.hpp>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>
#include <chrono>
#include <memory>

namespace ov {
namespace genai {

/**
 * @brief Output structure for Eagle3 model inference
 * 
 * Contains all outputs from a single inference call, including logits and hidden features.
 * This structure makes it easy to handle multiple outputs and allows for future extensions.
 */
struct InferenceOutput {
    ov::Tensor logits;
    ov::Tensor hidden_features;
};

/**
 * @brief Base class for Eagle3 model inference with common functionality
 * 
 * This abstract base class provides shared functionality for both target and draft model wrappers,
 * including sequence management, tensor building, and performance metrics.
 */
class Eagle3InferWrapperBase {
public:
    explicit Eagle3InferWrapperBase(const ov::genai::ModelDesc& model_desc);
    virtual ~Eagle3InferWrapperBase() = default;

    // Configuration methods
    std::string device() const { return m_device; }
    ov::genai::GenerationConfig get_generation_config() const { return m_generation_config; }
    void set_generation_config(ov::genai::GenerationConfig cfg) { m_generation_config = std::move(cfg); }
    void set_verbose(bool verbose) { m_verbose = verbose; }
    bool is_verbose() const { return m_verbose; }

    // Sequence management
    void append_tokens(const std::vector<int64_t>& tokens);
    void truncate_sequence(std::size_t size);
    void trim_kv_cache(std::size_t tokens_to_remove);
    void reset_state();
    void release_memory();
    
    // Sequence access
    std::size_t get_sequence_length() const { return m_tokens.size(); }
    const std::vector<int64_t>& get_tokens() const { return m_tokens; }
    const std::vector<int64_t>& get_positions() const { return m_positions; }
    int64_t get_last_sampled_token() const { return m_last_sampled_token; }
    
    // Output access
    ov::Tensor get_logits() const;
    ov::Tensor get_hidden_features() const;
    
    // Tensor utilities
    // Builds model inputs by extracting the last token_count tokens from the sequence
    void build_model_inputs(std::size_t token_count,
                           ov::Tensor& input_ids, ov::Tensor& attention_mask, ov::Tensor& position_ids);
    ov::Tensor create_hidden_state_placeholder(const ov::Shape& shape) const;
    
    // Sampling
    std::variant<int64_t, std::vector<int64_t>> sample_tokens(const ov::Tensor& logits, std::size_t count);

    // Performance metrics
    ov::genai::RawPerfMetrics& get_raw_perf_metrics() { return m_raw_perf_metrics; }

protected:
    static constexpr std::size_t BATCH_SIZE = 1;
    
    // Core inference helper
    uint64_t execute_inference();
    void update_performance_metrics(uint64_t inference_time_us, std::size_t tokens_count);
    
    // Debug logging functions
    void log_debug(const std::string& message) const;
    void log_tensor_info(const std::string& name, const ov::Tensor& tensor) const;
    void log_tensor_content(const std::string& name, const ov::Tensor& tensor, std::size_t max_elements = 10) const;
    void log_model_inputs(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, const ov::Tensor& position_ids) const;
    void log_model_outputs(const ov::Tensor& logits, const ov::Tensor& hidden_features) const;

    // Model and configuration
    std::string m_device;
    ov::AnyMap m_properties;
    ov::genai::GenerationConfig m_generation_config;
    ov::genai::Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;
    
    // Device-specific limits
    std::size_t m_max_prompt_len = 0;
    std::size_t m_kv_cache_capacity = 0;
    
    // Temporary token sequences for validation window
    // NOTE: These are speculative tokens that may be rolled back
    // Verified tokens are stored in Pipeline's m_verified_tokens
    std::vector<int64_t> m_tokens;      // Current speculative token sequence
    std::vector<int64_t> m_positions;   // Corresponding position IDs
    
    // State tracking
    std::size_t m_processed_tokens = 0;
    int64_t m_last_sampled_token = -1;
    
    // Performance tracking
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    
    // Configuration
    bool m_verbose = true;
};

/**
 * @brief Target model wrapper for Eagle3 speculative decoding
 * 
 * Handles inference for the main/target model which validates draft predictions
 * and generates the final output tokens.
 */
class Eagle3TargetModelWrapper : public Eagle3InferWrapperBase {
public:
    explicit Eagle3TargetModelWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3TargetModelWrapper() = default;

    // Initialize sequence with full input
    void initialize_sequence(const ov::Tensor& input_ids, const ov::Tensor& position_ids);

    // Target model inference - returns both logits and hidden features
    InferenceOutput infer(const ov::Tensor& input_ids, 
                         const ov::Tensor& attention_mask, 
                         const ov::Tensor& position_ids);
};

/**
 * @brief Draft model wrapper for Eagle3 speculative decoding
 * 
 * Handles inference for the draft model which generates candidate tokens
 * using hidden states from the target model or its own previous predictions.
 */
class Eagle3DraftModelWrapper : public Eagle3InferWrapperBase {
public:
    explicit Eagle3DraftModelWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3DraftModelWrapper() = default;

    // Initialize sequence with input starting from second token (Eagle3 specific)
    // Draft model uses tokens[1:], position_ids will be [0, 1, 2, ..., seq_len-2]
    void initialize_sequence(const ov::Tensor& input_ids, const ov::Tensor& position_ids);

    // Draft model inference with hidden state inputs - returns both logits and hidden features
    InferenceOutput infer(const ov::Tensor& input_ids,
                         const ov::Tensor& attention_mask,
                         const ov::Tensor& position_ids,
                         const ov::Tensor& target_hidden_features,
                         const ov::Tensor& internal_hidden_features);
};

/**
 * @brief Stateful Eagle3 LLM Pipeline for speculative decoding
 * 
 * Implements the Eagle3 speculative decoding algorithm using a draft model
 * to generate candidate tokens and a main model to validate them.
 */
class StatefulEagle3LLMPipeline : public ov::genai::LLMPipelineImplBase {
public:
    StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& main_model_desc, 
                              const ov::genai::ModelDesc& draft_model_desc,
                              const std::vector<int>& hidden_layers_to_abstract = {});
    ~StatefulEagle3LLMPipeline();

    // LLMPipelineImplBase interface
    DecodedResults generate(StringInputs inputs, OptionalGenerationConfig generation_config, StreamerVariant streamer) override;
    DecodedResults generate(const ChatHistory& history, OptionalGenerationConfig generation_config, StreamerVariant streamer) override;
    EncodedResults generate(const EncodedInputs& inputs, OptionalGenerationConfig generation_config, StreamerVariant streamer) override;
    void start_chat(const std::string& system_message) override;
    void finish_chat() override;

    // Eagle3-specific configuration
    void set_draft_target_mapping(const std::shared_ptr<ov::Model>& draft_model);
    void set_verbose(bool verbose);
    bool is_verbose() const { return m_main_model ? m_main_model->is_verbose() : false; }
    
    // Configuration resolution
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config);
    
    // Performance metrics
    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;
    
    // Debug logging
    void log_generation_step(const std::string& step_name, std::size_t step_number) const;
    void log_sequence_state(const std::string& context) const;

private:
    struct SpeculativeResult {
        ov::Tensor next_hidden_window;          // Hidden states for next iteration
        std::size_t accepted_tokens_count = 0;  // Number of accepted tokens
        std::size_t next_window_size = 0;       // Size of window for next iteration  
        int64_t new_token = -1;                 // Newly generated token (-1 if none)
        bool eos_reached = false;               // Whether EOS was generated
    };

    // Core Eagle3 algorithm
    SpeculativeResult run_speculative_iteration(const ov::Tensor& target_hidden_states, std::size_t token_count, int64_t eos_token_id);
    
    // Draft-to-target token mapping
    int64_t map_draft_token(int64_t draft_token) const;
    std::vector<int64_t> map_draft_tokens(const std::vector<int64_t>& draft_tokens) const;
    
    // Logging utilities
    void log_info(const std::string& message) const;
    void log_debug(const std::string& message) const;
    void log_performance_summary() const;
    
    // Helper methods for tensor manipulation
    ov::Tensor slice_hidden_features(const ov::Tensor& hidden_features, std::size_t start_pos, std::size_t length) const;
    ov::Tensor combine_hidden_windows(const ov::Tensor& confirmed_hidden, const ov::Tensor& new_hidden) const;

    // Model wrappers
    std::unique_ptr<Eagle3DraftModelWrapper> m_draft_model;
    std::unique_ptr<Eagle3TargetModelWrapper> m_main_model;
    
    // Algorithm parameters
    std::size_t m_draft_iterations = 5;  // Number of draft iterations (configurable via num_assistant_tokens)
    
    // Draft-to-target token mapping
    ov::Tensor m_draft_target_mapping;
    
    // Hidden layers configuration
    std::vector<int> m_hidden_layers_to_abstract;
    
    // Token storage - Pipeline is the source of truth for verified tokens
    std::vector<int64_t> m_verified_tokens;  // All verified tokens (prompt + generated)
    std::size_t m_prompt_length = 0;         // Length of initial prompt
    
    // Performance tracking - using standard speculative decoding metrics
    // m_sd_metrics: Algorithm-specific metrics (acceptance rate, draft efficiency, etc.)
    // m_sd_perf_metrics: Complete performance statistics for both main and draft models
    //   - main_model_metrics: TTFT, TPOT, throughput for main/target model
    //   - draft_model_metrics: TTFT, TPOT, throughput for draft model  
    //   - num_accepted_tokens: Number of draft tokens accepted by main model (NOT including main's own predictions)
    ov::genai::SpeculativeDecodingMetrics m_sd_metrics;
    ov::genai::SDPerModelsPerfMetrics m_sd_perf_metrics;
    
    // Chat state
    bool m_is_chat_active = false;
    ChatHistory m_chat_history;
    bool m_streaming_was_cancelled = false;
};

}  // namespace genai
}  // namespace ov
