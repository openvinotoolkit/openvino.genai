// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/genai/perf_metrics.hpp>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>

#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "stateful_pipeline_base.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

/// @brief Eagle3 model inference output
struct InferenceOutput {
    ov::Tensor logits;           ///< Output logits [batch, seq_len, vocab_size]
    ov::Tensor hidden_features;  ///< Hidden states for draft model input
};

/// @brief Context for a single forward pass
struct InferContext {
    size_t input_token_count = 0;             ///< Number of input tokens for this inference
    size_t sample_count = 1;                  ///< Number of positions to sample from
    bool use_target_hidden = false;           ///< Whether to use hidden states from target_sequence
    Sequence::Ptr target_sequence = nullptr;  ///< Source sequence for hidden states
    size_t num_tokens_to_validate = 0;        ///< Number of draft tokens to validate
};

/// @brief Result from a forward pass
struct InferResult {
    InferenceOutput output;               ///< Raw model outputs
    std::vector<int64_t> sampled_tokens;  ///< Sampled token(s)
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
            OPENVINO_ASSERT(get_running_sequence_count() == 1,
                            "Eagle3 only supports single sequence, got ",
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

    /// @brief Returns current (first) sequence
    Sequence::Ptr get_current_sequence() const {
        return get_sequence(0);
    }

    ov::Tensor get_logits() const;
    ov::Tensor get_hidden_features() const;

    void build_model_inputs(const size_t token_count,
                            ov::Tensor& input_ids,
                            ov::Tensor& attention_mask,
                            ov::Tensor& position_ids);

    /// @brief Samples tokens from logits
    /// @param num_tokens_to_validate Draft tokens to validate (0 for standard sampling)
    /// @return Vector of sampled token(s) - size 1 for standard sampling, size N for validation mode
    std::vector<int64_t> sample_tokens(const ov::Tensor& logits,
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
    void update_inference_time(uint64_t inference_time_us);
    void record_generated_tokens(size_t actual_generated_count);

    std::string m_device;
    ov::AnyMap m_properties;
    ov::genai::Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;
    size_t m_max_prompt_len = 0;

    SequenceGroup::Ptr m_sequence_group;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
};

/**
 * @brief Target model wrapper for Eagle3
 *
 * Validates draft predictions and generates final output
 */
class Eagle3TargetWrapper : public Eagle3InferWrapperBase {
public:
    explicit Eagle3TargetWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3TargetWrapper() = default;

    /// @brief Initializes sequence with prompt tokens
    void initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config);

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
class Eagle3DraftWrapper : public Eagle3InferWrapperBase {
public:
    explicit Eagle3DraftWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3DraftWrapper() = default;

    /// @brief Initializes sequence using tokens[1:] per Eagle3 spec
    void initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config);

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
 * target model validates them in parallel, accepts valid tokens and samples new ones
 */
class StatefulEagle3LLMPipeline : public StatefulSpeculativePipelineBase {
public:
    StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                              const ov::genai::ModelDesc& draft_model_desc);
    ~StatefulEagle3LLMPipeline();

    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;

    std::vector<float> get_next_token_log_probs(
        const std::string& prompt,
        const std::vector<int64_t>& token_ids
    );

    // Override to reset model states
    void finish_chat() override;

protected:
    // Override base class methods
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) override;

    EncodedResults generate_tokens(const EncodedInputs& inputs,
                                   const GenerationConfig& config,
                                   StreamerVariant streamer) override;

private:
    struct SpeculativeResult {
        size_t accepted_tokens_count = 0;
        size_t next_window_size = 0;
        bool eos_reached = false;
        std::vector<int64_t> validated_tokens;
    };

    SpeculativeResult run_speculative_iteration(size_t token_count, int64_t eos_token_id);

    std::unique_ptr<Eagle3DraftWrapper> m_draft;
    std::unique_ptr<Eagle3TargetWrapper> m_target;

    size_t m_draft_iterations = 5;
    std::vector<int32_t> m_hidden_layers_to_abstract;
    size_t m_prompt_length = 0;
};

}  // namespace genai
}  // namespace ov
