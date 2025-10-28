// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/genai/perf_metrics.hpp>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>

#include "llm/pipeline_base.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding_metrics.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

enum class LogCategory;

/// @brief Eagle3 model inference output
struct InferenceOutput {
    ov::Tensor logits;           ///< Output logits [batch, seq_len, vocab_size]
    ov::Tensor hidden_features;  ///< Hidden states for draft model input
};

/// @brief Context for a single forward pass
struct InferContext {
    size_t input_token_count = 0;             ///< Number of input tokens for this inference
    size_t sample_count = 1;                  ///< Number of positions to sample from
    bool use_target_hidden = false;           ///< Whether to use hidden states from source_sequence
    Sequence::Ptr source_sequence = nullptr;  ///< Source sequence for hidden states
    size_t num_tokens_to_validate = 0;        ///< Number of draft tokens to validate
};

/// @brief Result from a forward pass
struct InferResult {
    InferenceOutput output;                                      ///< Raw model outputs
    std::variant<int64_t, std::vector<int64_t>> sampled_tokens;  ///< Sampled token(s)
};

/**
 * @brief Base class for Eagle3 model inference wrappers
 *
 * Provides shared functionality for target and draft model wrappers including
 * sequence management, KV cache operations, tensor building, and token sampling
 */
class Eagle3InferWrapperBase {
public:
    explicit Eagle3InferWrapperBase(const ov::genai::ModelDesc& model_desc);
    virtual ~Eagle3InferWrapperBase() = default;

    std::string device() const {
        return m_device;
    }
    void set_verbose(bool verbose) {
        m_verbose = verbose;
    }
    bool is_verbose() const {
        return m_verbose;
    }

    /// @brief Sets draft-to-target token mapping for sampler
    void set_draft_target_mapping(std::shared_ptr<ov::op::v0::Constant> d2t_mapping) {
        m_sampler.set_d2t_for_decoding(d2t_mapping);
    }

    void append_tokens(const std::vector<int64_t>& tokens);
    void truncate_sequence(size_t size);
    void trim_kv_cache(size_t tokens_to_remove);
    void reset_state();
    void release_memory();

    /// @brief Returns total sequence length (prompt + generated)
    /// @note Currently only supports single sequence (top-1)
    size_t get_sequence_length() const {
        if (auto seq = get_sequence(0)) {
            return m_sequence_group->get_prompt_len() + seq->get_generated_len();
        }
        return 0;
    }

    /// @brief Returns only generated tokens
    /// @note Currently only supports single sequence (top-1)
    const std::vector<int64_t>& get_generated_tokens() const {
        static const std::vector<int64_t> empty;
        if (auto seq = get_sequence(0)) {
            return seq->get_generated_ids();
        }
        return empty;
    }

    SequenceGroup::Ptr get_sequence_group() const {
        return m_sequence_group;
    }

    void set_sequence_group(SequenceGroup::Ptr sequence_group) {
        m_sequence_group = sequence_group;
        if (m_sequence_group) {
            // TODO(top-k): Remove this assertion when top-k is supported
            OPENVINO_ASSERT(get_running_sequence_count() == 1,
                            "Eagle3 currently only supports top-1 (single sequence), got ",
                            get_running_sequence_count());
        }
    }

    /// @brief Returns number of running sequences in the group
    size_t get_running_sequence_count() const {
        return m_sequence_group ? m_sequence_group->get_running_sequences().size() : 0;
    }

    /// @brief Returns sequence at given index with bounds checking
    /// @param index Sequence index (0 for top-1)
    /// @return Sequence pointer or nullptr if index out of bounds
    Sequence::Ptr get_sequence(size_t index) const {
        if (m_sequence_group) {
            auto sequences = m_sequence_group->get_running_sequences();
            if (index < sequences.size()) {
                return sequences[index];
            }
        }
        return nullptr;
    }

    /// @brief Returns current (first) sequence - convenience method for top-1
    /// @deprecated Use get_sequence(0) for clarity, will be removed when top-k is implemented
    Sequence::Ptr get_current_sequence() const {
        return get_sequence(0);
    }

    ov::Tensor get_logits() const;
    ov::Tensor get_hidden_features() const;

    void build_model_inputs(size_t token_count,
                            ov::Tensor& input_ids,
                            ov::Tensor& attention_mask,
                            ov::Tensor& position_ids);

    ov::Tensor create_hidden_state_placeholder(const ov::Shape& shape) const;

    /// @brief Samples tokens from logits
    /// @param num_tokens_to_validate Draft tokens to validate (0 for standard sampling)
    /// @return Single token or vector of validated tokens
    std::variant<int64_t, std::vector<int64_t>> sample_tokens(const ov::Tensor& logits,
                                                              size_t input_token_count,
                                                              size_t sample_count,
                                                              size_t num_tokens_to_validate = 0);

    /// @brief Executes forward pass: prepare inputs, infer, and sample
    virtual InferResult forward(const InferContext& ctx) = 0;

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

protected:
    static constexpr size_t BATCH_SIZE = 1;

    uint64_t execute_inference();
    void update_performance_metrics(uint64_t inference_time_us, size_t tokens_count);

    void log_debug(LogCategory category, const std::string& message) const;
    void log_tensor_info(const std::string& name, const ov::Tensor& tensor) const;
    void log_tensor_content(const std::string& name, const ov::Tensor& tensor, size_t max_elements = 10) const;
    void log_model_inputs(const ov::Tensor& input_ids,
                          const ov::Tensor& attention_mask,
                          const ov::Tensor& position_ids) const;
    void log_model_outputs(const ov::Tensor& logits, const ov::Tensor& hidden_features) const;

    std::string m_device;
    ov::AnyMap m_properties;
    ov::genai::Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;
    size_t m_max_prompt_len = 0;

    SequenceGroup::Ptr m_sequence_group;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    bool m_verbose = false;
};

/**
 * @brief Target model wrapper for Eagle3
 *
 * Validates draft predictions and generates final output
 */
class Eagle3TargetModelWrapper : public Eagle3InferWrapperBase {
public:
    explicit Eagle3TargetModelWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3TargetModelWrapper() = default;

    /// @brief Initializes sequence with prompt tokens
    void initialize_sequence(const ov::Tensor& input_ids,
                             const ov::Tensor& position_ids,
                             const ov::genai::GenerationConfig& config);

    InferenceOutput infer(const ov::Tensor& input_ids,
                          const ov::Tensor& attention_mask,
                          const ov::Tensor& position_ids);

    InferResult forward(const InferContext& ctx) override;
};

/**
 * @brief Draft model wrapper for Eagle3
 *
 * Generates candidate tokens using target or internal hidden states
 * Uses tokens[1:] per Eagle3 specification
 */
class Eagle3DraftModelWrapper : public Eagle3InferWrapperBase {
public:
    explicit Eagle3DraftModelWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3DraftModelWrapper() = default;

    /// @brief Initializes sequence using tokens[1:] per Eagle3 spec
    void initialize_sequence(const ov::Tensor& input_ids,
                             const ov::Tensor& position_ids,
                             const ov::genai::GenerationConfig& config);

    /// @brief Runs inference with hidden states (from target or internal source)
    InferenceOutput infer(const ov::Tensor& input_ids,
                          const ov::Tensor& attention_mask,
                          const ov::Tensor& position_ids,
                          const ov::Tensor& hidden_states);

    InferResult forward(const InferContext& ctx) override;
};

/**
 * @brief Stateful Eagle3 speculative decoding pipeline
 *
 * Implements Eagle3 algorithm: draft model generates candidates,
 * main model validates them in parallel, accepts valid tokens and samples new ones
 */
class StatefulEagle3LLMPipeline : public ov::genai::LLMPipelineImplBase {
public:
    StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& main_model_desc,
                              const ov::genai::ModelDesc& draft_model_desc,
                              const std::vector<int>& hidden_layers_to_abstract = {});
    ~StatefulEagle3LLMPipeline();

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

    void set_verbose(bool verbose);
    bool is_verbose() const {
        return m_main_model ? m_main_model->is_verbose() : false;
    }

    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config);
    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;

private:
    struct SpeculativeResult {
        size_t accepted_tokens_count = 0;
        size_t next_window_size = 0;
        bool eos_reached = false;
        std::vector<int64_t> validated_tokens;
    };

    SpeculativeResult run_speculative_iteration(size_t token_count, int64_t eos_token_id);

    void log_info(const std::string& message) const;
    void log_debug(const std::string& message) const;
    void log_generation_step(const std::string& step_name, size_t step_number) const;
    void log_sequence_state(const std::string& context) const;

    std::unique_ptr<Eagle3DraftModelWrapper> m_draft_model;
    std::unique_ptr<Eagle3TargetModelWrapper> m_main_model;

    size_t m_draft_iterations = 5;
    std::vector<int> m_hidden_layers_to_abstract;
    size_t m_prompt_length = 0;

    ov::genai::SpeculativeDecodingMetrics m_sd_metrics;
    ov::genai::SDPerModelsPerfMetrics m_sd_perf_metrics;

    bool m_is_chat_active = false;
    ChatHistory m_chat_history;
    bool m_streaming_was_cancelled = false;
};

}  // namespace genai
}  // namespace ov
