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
 * @brief Wrapper for Eagle3 model inference with performance tracking
 * 
 * This class handles both draft and target model inference for Eagle3 speculative decoding,
 * providing sequence management, tensor building, and performance metrics.
 */
class Eagle3InferWrapper {
public:
    explicit Eagle3InferWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3InferWrapper() = default;

    // Configuration methods
    std::string device() const { return m_device; }
    ov::genai::GenerationConfig get_generation_config() const { return m_generation_config; }
    void set_generation_config(ov::genai::GenerationConfig cfg) { m_generation_config = std::move(cfg); }
    void set_verbose(bool verbose) { m_verbose = verbose; }
    bool is_verbose() const { return m_verbose; }

    // Sequence management
    void initialize_sequence(const ov::Tensor& input_ids, const ov::Tensor& position_ids);
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

    // Core inference methods
    ov::Tensor infer_target_model(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, const ov::Tensor& position_ids);
    ov::Tensor infer_draft_model(const ov::Tensor& input_ids, const ov::Tensor& attention_mask, const ov::Tensor& position_ids,
                                 const ov::Tensor& target_hidden_features, const ov::Tensor& internal_hidden_features);
    
    // Output access
    ov::Tensor get_logits() const;
    ov::Tensor get_hidden_features() const;
    
    // Tensor utilities
    void build_model_inputs(int64_t begin_idx, std::size_t size,
                           ov::Tensor& input_ids, ov::Tensor& attention_mask, ov::Tensor& position_ids, 
                           bool reset_positions = false, bool full_attention_mask = false);
    ov::Tensor create_hidden_state_placeholder(const ov::Shape& shape) const;
    
    // Sampling
    std::variant<int64_t, std::vector<int64_t>> sample_tokens(const ov::Tensor& logits, std::size_t count);

    // Performance metrics
    struct InferenceMetrics {
        uint64_t total_inference_time_us = 0;
        uint64_t last_inference_time_us = 0;
        std::size_t total_inferences = 0;
        std::size_t total_tokens_processed = 0;
        
        double get_average_inference_time_us() const {
            return total_inferences > 0 ? static_cast<double>(total_inference_time_us) / total_inferences : 0.0;
        }
        
        double get_tokens_per_second() const {
            return total_inference_time_us > 0 ? (total_tokens_processed * 1000000.0) / total_inference_time_us : 0.0;
        }
        
        void reset() {
            total_inference_time_us = 0;
            last_inference_time_us = 0;
            total_inferences = 0;
            total_tokens_processed = 0;
        }
    };
    
    const InferenceMetrics& get_metrics() const { return m_metrics; }
    ov::genai::RawPerfMetrics& get_raw_perf_metrics() { return m_raw_perf_metrics; }

private:
    static constexpr std::size_t BATCH_SIZE = 1;
    
    // Core inference helper
    uint64_t execute_inference(const ov::Tensor& input_ids);
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
    
    // Token sequences - each wrapper only maintains its own sequences
    std::vector<int64_t> m_tokens;      // Either target or draft tokens depending on wrapper type
    std::vector<int64_t> m_positions;   // Corresponding position IDs
    
    // State tracking
    std::size_t m_processed_tokens = 0;
    int64_t m_last_sampled_token = -1;
    
    // Performance tracking
    InferenceMetrics m_metrics;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    
    // Configuration
    bool m_verbose = true;
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
    
    // Performance metrics
    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;
    
    // Debug logging
    void log_generation_step(const std::string& step_name, std::size_t step_number) const;
    void log_sequence_state(const std::string& context) const;
    
    struct PerformanceSummary {
        uint64_t total_generation_time_us = 0;
        uint64_t draft_inference_time_us = 0;
        uint64_t main_inference_time_us = 0;
        uint64_t validation_time_us = 0;
        
        std::size_t prompt_tokens = 0;
        std::size_t generated_tokens = 0;
        std::size_t draft_iterations = 0;
        std::size_t validation_rounds = 0;
        std::size_t accepted_tokens = 0;
        std::size_t rejected_tokens = 0;
        
        double get_acceptance_rate() const {
            std::size_t total = accepted_tokens + rejected_tokens;
            return total > 0 ? static_cast<double>(accepted_tokens) / total : 0.0;
        }
        
        double get_speedup() const {
            return main_inference_time_us > 0 ? 
                static_cast<double>(generated_tokens * main_inference_time_us) / (draft_inference_time_us + main_inference_time_us) : 1.0;
        }
        
        void reset() {
            *this = PerformanceSummary{};
        }
    };
    
    const PerformanceSummary& get_performance_summary() const { return m_perf_summary; }

private:
    struct SpeculativeResult {
        ov::Tensor next_hidden_window;          // Hidden states for next iteration
        std::size_t accepted_tokens_count = 0;  // Number of accepted tokens
        std::size_t next_window_size = 0;       // Size of window for next iteration  
        int64_t new_token = -1;                 // Newly generated token (-1 if none)
        bool eos_reached = false;               // Whether EOS was generated
    };

    // Core Eagle3 algorithm
    SpeculativeResult run_speculative_iteration(const ov::Tensor& hidden_window, std::size_t window_size, int64_t eos_token_id);
    
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
    std::unique_ptr<Eagle3InferWrapper> m_draft_model;
    std::unique_ptr<Eagle3InferWrapper> m_main_model;
    
    // Algorithm parameters
    static constexpr std::size_t DEFAULT_DRAFT_ITERATIONS = 3;
    static constexpr std::size_t DEFAULT_VALIDATION_WINDOW = 5;
    static constexpr std::size_t MAX_CANDIDATES = 10;  // For NPU compatibility
    
    // Draft-to-target token mapping
    ov::Tensor m_draft_target_mapping;
    
    // Hidden layers configuration
    std::vector<int> m_hidden_layers_to_abstract;
    
    // Performance tracking
    PerformanceSummary m_perf_summary;
    ov::genai::SpeculativeDecodingMetrics m_sd_metrics;
    ov::genai::SDPerModelsPerfMetrics m_sd_perf_metrics;
    
    // Chat state
    bool m_is_chat_active = false;
    ChatHistory m_chat_history;
    bool m_streaming_was_cancelled = false;
    
    // Configuration
    bool m_verbose = true;
};

}  // namespace genai
}  // namespace ov
