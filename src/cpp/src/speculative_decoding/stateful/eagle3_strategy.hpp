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

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// @brief Bundled model input tensors produced by Eagle3InputBuilder.
struct InputTensors {
    ov::Tensor input_ids;        ///< Token ids          [1, seq_len]
    ov::Tensor attention_mask;   ///< Attention mask      [1, attn_len]
    ov::Tensor position_ids;     ///< Position ids        [1, seq_len]
    ov::Tensor eagle_tree_mask;  ///< Tree attention mask [1, 1, seq_len, attn_len] or [1,1,1,1]
};

/// @brief Eagle3 model inference output.
struct InferenceOutput {
    ov::Tensor logits;           ///< Output logits      [1, seq_len, vocab_size]
    ov::Tensor hidden_features;  ///< Hidden states      [1, seq_len, hidden_size]
};

/// @brief Context for a single forward pass.
struct InferContext {
    size_t input_token_count = 0;             ///< Number of input tokens for this inference
    size_t sample_count = 1;                  ///< Number of positions to sample from
    bool use_target_hidden = false;           ///< Whether to use hidden states from target_sequence
    Sequence::Ptr target_sequence = nullptr;  ///< Source sequence for hidden states (DRAFT_INITIAL only)
    size_t num_tokens_to_validate = 0;        ///< Number of draft tokens to validate (TARGET_VALIDATION only)
    size_t past_accepted_token_count = 0;     ///< Tokens already accepted by target before this speculative window
};

/// @brief Result from a forward pass.
struct InferResult {
    InferenceOutput output;               ///< Raw model outputs
    std::vector<int64_t> sampled_tokens;  ///< Sampled token(s)
};

/// @brief Result from tree-search validation.
struct ValidationResult {
    std::vector<int64_t> validated_tokens;  ///< [accepted_draft_tokens..., bonus_token]
    size_t accepted_count = 0;              ///< Number of accepted draft tokens (excludes bonus)
    size_t num_candidates = 0;              ///< Total candidates submitted (N+1)
    InferenceOutput output;                 ///< Raw target model outputs
};

// ---------------------------------------------------------------------------
// Eagle3InputBuilder — stateless input tensor construction
// ---------------------------------------------------------------------------

/// @brief Constructs model input tensors for each Eagle3 inference phase.
///
/// Each method reads sequence state (prompt ids, generated ids, EagleMetaData)
/// and produces a complete InputTensors bundle.  The builder is stateless; it
/// holds a non-owning reference to the active SequenceGroup.
class Eagle3InputBuilder {
public:
    explicit Eagle3InputBuilder(const SequenceGroup::Ptr& sequence_group) : m_sequence_group(sequence_group) {}

    /// @brief Builds inputs for TARGET_PREFILL and DRAFT_INITIAL phases (causal, no tree mask).
    InputTensors build_prefill_inputs(size_t input_token_count) const;

    /// @brief Builds inputs for DRAFT_ITERATION: flat-concatenated branch tokens across all sequences.
    /// @param past_accepted_token_count Tokens already in the draft KV cache (history + root).
    InputTensors build_draft_iteration_inputs(size_t past_accepted_token_count) const;

    /// @brief Builds inputs for TARGET_VALIDATION: all N+1 tree candidates with tree attention mask.
    InputTensors build_target_validation_inputs() const;

private:
    SequenceGroup::Ptr m_sequence_group;
};

// ---------------------------------------------------------------------------
// Eagle3InferWrapperBase — shared model-wrapper infrastructure
// ---------------------------------------------------------------------------

/**
 * @brief Base class for Eagle3 model inference wrappers.
 *
 * Provides shared functionality for target and draft model wrappers including
 * sequence management, KV cache operations, and token sampling.
 */
class Eagle3InferWrapperBase {
public:
    explicit Eagle3InferWrapperBase(const ov::genai::ModelDesc& model_desc);
    virtual ~Eagle3InferWrapperBase() = default;

    const std::string& device() const {
        return m_device;
    }

    /// @brief Sets draft-to-target token mapping for sampler.
    void set_draft_target_mapping(std::shared_ptr<ov::op::v0::Constant> d2t_mapping) {
        m_sampler.set_d2t_for_decoding(std::move(d2t_mapping));
    }

    void append_tokens(const std::vector<int64_t>& tokens);
    void truncate_sequence(size_t size);
    void trim_kv_cache(size_t tokens_to_remove);

    /// @brief Communicates the Eagle3 tree-search sampling result to NPUW via VariableState.
    ///
    /// On NPU devices the KV cache is managed internally by NPUW.  Instead of calling
    /// trim_kv_cache (which is a no-op on NPU), the pipeline writes the acceptance mask
    /// into the "npuw_eagle3_sampling_result" VariableState before the next infer call so
    /// that NPUW can prune the correct (non-contiguous) tree positions.
    ///
    /// @param num_candidates   Total number of candidate tokens fed to the model (N+1).
    /// @param accepted_indices Candidate indices that were accepted (from EagleMetaData::validated_indices).
    ///                         Index 0 is always the root; subsequent entries are accepted draft nodes.
    void set_npu_sampling_result(size_t num_candidates, const std::vector<int64_t>& accepted_indices);

    void reset_state();
    void release_memory();

    /// @brief Returns total sequence length (prompt + generated) for sequence 0.
    ///
    /// For the target wrapper sequence 0 is the sole sequence.
    /// For the draft wrapper all sequences share the same history and root tokens,
    /// so sequence 0 length equals the shared KV-cache baseline used as the
    /// starting offset for DRAFT_ITERATION input construction.
    size_t get_sequence_length() const {
        if (const auto seq = get_sequence(0)) {
            return m_sequence_group->get_prompt_len() + seq->get_generated_len();
        }
        return 0;
    }

    /// @brief Returns generated token ids for sequence 0.
    ///
    /// Intended for the target wrapper (which always holds a single sequence).
    /// For the draft wrapper this returns the generated ids of sequence 0 only,
    /// which is sufficient for history-length queries and debug logging.
    const std::vector<int64_t>& get_generated_tokens() const {
        static const std::vector<int64_t> empty;
        if (const auto seq = get_sequence(0)) {
            return seq->get_generated_ids();
        }
        return empty;
    }

    SequenceGroup::Ptr get_sequence_group() const {
        return m_sequence_group;
    }

    /// @brief Returns the sequence at the given index (status-agnostic).
    ///
    /// Uses get_sequences() instead of get_running_sequences() so that sequences
    /// temporarily marked FINISHED by the sampler (e.g. on EOS) remain accessible.
    /// Eagle3 pipeline manages generation termination externally.
    /// @param index Zero-based index into the sequence list.
    /// @return Sequence pointer, or nullptr if the index is out of range.
    Sequence::Ptr get_sequence(size_t index) const {
        if (m_sequence_group) {
            const auto& sequences = m_sequence_group->get_sequences();
            if (index < sequences.size()) {
                return sequences[index];
            }
        }
        return nullptr;
    }

    /// @brief Returns sequence 0 (the sole target sequence, or the shared base sequence for the draft).
    Sequence::Ptr get_current_sequence() const {
        return get_sequence(0);
    }

    /// @brief Returns raw logits tensor from model output.
    /// On NPU, the output length is a fixed NPUW_LLM_MAX_GENERATION_TOKEN_LEN (possibly
    /// aligned) which may differ from input length.  Useful positions are always at the tail.
    ov::Tensor get_logits() const;

    /// @brief Extracts the hidden features from the model output.
    /// @param actual_seq_len  The actual input sequence length (shape[1] of the input_ids
    ///                        tensor that was fed to the model).  Pass 0 to let the method
    ///                        re-query the input_ids tensor.
    ov::Tensor get_hidden_features(size_t actual_seq_len = 0) const;

    /// @brief Tail-slices a 3-D tensor to the last `useful_len` positions along dim 1.
    /// On NPU, both logits and hidden-state outputs may be longer than the actual useful
    /// data due to hardware alignment.  The real data always occupies the tail.
    /// Returns the tensor unchanged if its dim-1 already equals useful_len.
    static ov::Tensor trim_tensor_tail(const ov::Tensor& tensor, size_t useful_len);

    /// @brief Executes forward pass: prepare inputs, infer, and sample.
    virtual InferResult forward(const InferContext& ctx) = 0;

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

protected:
    static constexpr size_t BATCH_SIZE = 1;

    uint64_t execute_inference();
    void update_inference_time(uint64_t inference_time_us);
    void record_generated_tokens(size_t actual_generated_count);

    // ---- Sampling helpers (split by mode) ----

    /// @brief Configures the SequenceGroup for sampling and invokes the sampler.
    /// @param logits           Logit tensor to feed to the sampler.
    /// @param input_token_count  Number of input tokens (for scheduling).
    /// @param sample_count     Number of output positions to sample.
    /// @param is_validation    Whether the sampler should run in tree-validation mode.
    void invoke_sampler(const ov::Tensor& logits, size_t input_token_count, size_t sample_count, bool is_validation);

    /// @brief Samples tokens in draft mode (non-validation). Collects the last generated
    ///        token from each running sequence after the sampler has run.
    /// @return One token per running sequence.
    std::vector<int64_t> sample_draft_tokens(const ov::Tensor& logits, size_t input_token_count);

    /// @brief Samples tokens in validation mode. The sampler's validate_tree_candidates
    ///        rewrites generated_ids; this method extracts the accepted + bonus tokens.
    /// @return [acc_1, ..., acc_k, bonus_token]
    std::vector<int64_t> sample_and_validate(const ov::Tensor& logits,
                                             size_t input_token_count,
                                             size_t num_candidates,
                                             size_t num_tokens_to_validate);

    /// @brief Restores all sequences in the group to RUNNING status.
    ///
    /// The sampler's _try_finish_generation() may mark sequences as FINISHED when it
    /// encounters an EOS token or max_new_tokens limit.  In Eagle3 speculative decoding
    /// the pipeline controls generation termination externally — only the target model's
    /// validated EOS should truly end generation.  Call this after every sampler invocation
    /// to undo any premature FINISHED status.
    void restore_running_status();

    std::string m_device;
    ov::AnyMap m_properties;
    ov::genai::Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;
    size_t m_max_prompt_len = 0;

    SequenceGroup::Ptr m_sequence_group;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    ov::genai::utils::CacheTypes m_cache_types;
};

// ---------------------------------------------------------------------------
// Eagle3TargetWrapper — target model
// ---------------------------------------------------------------------------

/**
 * @brief Target model wrapper for Eagle3.
 *
 * Validates draft predictions and generates final output.
 */
class Eagle3TargetWrapper : public Eagle3InferWrapperBase {
public:
    explicit Eagle3TargetWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3TargetWrapper() override = default;

    /// @brief Initializes sequence with prompt tokens.
    void initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config);

    /// @brief Runs target model inference.
    InferenceOutput infer(const InputTensors& inputs);

    InferResult forward(const InferContext& ctx) override;
};

// ---------------------------------------------------------------------------
// Eagle3DraftWrapper — draft model with pre-allocated buffers
// ---------------------------------------------------------------------------

/**
 * @brief Draft model wrapper for Eagle3.
 *
 * Generates candidate tokens using target or internal hidden states.
 * Uses tokens[1:] per Eagle3 specification.
 *
 * Pre-allocates reusable tensor buffers for the DRAFT_ITERATION hot path
 * to avoid repeated heap allocation during the speculative loop.
 */
class Eagle3DraftWrapper : public Eagle3InferWrapperBase {
public:
    explicit Eagle3DraftWrapper(const ov::genai::ModelDesc& model_desc);
    ~Eagle3DraftWrapper() override = default;

    /// @brief Initializes sequence using tokens[1:] per Eagle3 spec.
    void initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config);

    /// @brief Runs draft model inference with hidden states.
    InferenceOutput infer(const InputTensors& inputs, const ov::Tensor& hidden_states);

    InferResult forward(const InferContext& ctx) override;

    /// @brief Pre-allocates reusable buffers once hidden_size and vocab_size are known.
    /// @param max_sequences   Maximum number of concurrent beam sequences (branching_factor).
    /// @param max_depth       Maximum tree depth (number of DRAFT_ITERATION passes).
    /// @param hidden_size     Hidden state dimension from the model.
    /// @param vocab_size      Vocabulary size from the model.
    void allocate_buffers(size_t max_sequences, size_t max_depth, size_t hidden_size, size_t vocab_size);

private:
    /// @brief Prepares hidden states for the current forward pass.
    /// DRAFT_INITIAL: returns the target model's hidden state directly.
    /// DRAFT_ITERATION: concatenates per-sequence hidden states into a flat tensor.
    ov::Tensor prepare_hidden_states(const InferContext& ctx);

    /// @brief Updates per-sequence hidden states from model output after inference.
    /// DRAFT_INITIAL: broadcasts root hidden state to all beam sequences.
    /// DRAFT_ITERATION: appends each sequence's new hidden token in-place.
    void update_hidden_states(const InferenceOutput& output, const InferContext& ctx);

    /// @brief Gathers last-position logits for each sequence in DRAFT_ITERATION mode.
    /// Returns the raw logits tensor unchanged in DRAFT_INITIAL mode.
    ov::Tensor gather_logits_for_sampling(const InferenceOutput& output, const InferContext& ctx);

    // Pre-allocated buffers (initialized by allocate_buffers).
    ov::Tensor m_logits_gather_buf;  ///< {1, max_sequences, vocab_size} for logit gathering
    ov::Tensor m_hidden_concat_buf;  ///< {1, max_sequences * max_depth, hidden_size} for concat
    size_t m_max_sequences = 0;
    size_t m_max_depth = 0;
    size_t m_hidden_size = 0;
    size_t m_vocab_size = 0;
    bool m_buffers_allocated = false;

public:
    /// @brief Returns true if pre-allocated buffers have been initialized.
    bool buffers_allocated() const {
        return m_buffers_allocated;
    }
};

// ---------------------------------------------------------------------------
// StatefulEagle3LLMPipeline — top-level pipeline
// ---------------------------------------------------------------------------

/**
 * @brief Stateful Eagle3 speculative decoding pipeline.
 *
 * Implements the Eagle3 algorithm:
 *   1. DRAFT_INITIAL  — draft model generates the first tree level using target hidden states.
 *   2. DRAFT_ITERATION — draft model expands the tree for (tree_depth - 1) more levels.
 *   3. TARGET_VALIDATION — target model validates all N+1 tree candidates in one pass.
 *   4. Accepted tokens are committed; KV caches and hidden states are synchronized.
 */
class StatefulEagle3LLMPipeline : public StatefulSpeculativePipelineBase {
public:
    /** @brief Default branching factor (top-k candidates per tree node) when not user-specified. */
    static constexpr size_t DEFAULT_EAGLE_BRANCHING_FACTOR = 2;
    /** @brief Default tree depth (number of DRAFT_ITERATION passes) when not user-specified. */
    static constexpr size_t DEFAULT_EAGLE_TREE_DEPTH = 4;
    /** @brief Default number of speculative tokens in the tree when not user-specified. */
    static constexpr size_t DEFAULT_EAGLE_NUM_SPECULATIVE_TOKENS = 7;

    StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                              const ov::genai::ModelDesc& draft_model_desc);
    ~StatefulEagle3LLMPipeline();

    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;

    void finish_chat() override;

protected:
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

    // ---- Speculative iteration sub-steps ----

    /// @brief Runs a complete speculative iteration (draft + validate + sync).
    SpeculativeResult run_speculative_iteration(size_t token_count, int64_t eos_token_id);

    /// @brief Step 1: Generates the first draft tree level using target hidden states.
    InferResult generate_initial_draft(size_t input_token_count, size_t past_accepted_token_count);

    /// @brief Step 2: Expands the draft tree for (m_draft_iterations - 1) more levels.
    void expand_draft_tree(size_t past_accepted_token_count);

    /// @brief Step 3: Validates draft candidates with the target model.
    ValidationResult validate_draft_with_target();

    /// @brief Step 4: Synchronizes sequences and KV caches after validation.
    void synchronize_after_validation(const ValidationResult& validation, size_t pre_draft_token_len);

    /// @brief Step 5: Gathers accepted hidden states for the next iteration.
    void gather_accepted_hidden_states(const ValidationResult& validation);

    /// @brief Applies default eagle_tree_params if the user did not set them.
    static void ensure_eagle_tree_params_is_set(GenerationConfig& config);

    std::unique_ptr<Eagle3DraftWrapper> m_draft;
    std::unique_ptr<Eagle3TargetWrapper> m_target;

    size_t m_draft_iterations = 5;
    size_t m_prompt_length = 0;
};

}  // namespace genai
}  // namespace ov
