// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

#include "continuous_batching/timer.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "utils.hpp"

namespace {

/// @brief Streams generated tokens to the user-provided streamer callback.
ov::genai::StreamingStatus stream_generated_tokens(const std::shared_ptr<ov::genai::StreamerBase>& streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

/// @brief Extracts hidden state at a specific position; returns a tensor view [1, 1, hidden_size].
ov::Tensor slice_hidden_state(const ov::Tensor& hidden_features, size_t position) {
    OPENVINO_ASSERT(hidden_features.get_size() > 0, "Hidden features tensor is empty");
    const auto& shape = hidden_features.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] > 0, "Expected shape [1, seq_len, hidden_size]");
    OPENVINO_ASSERT(position < shape[1], "Position ", position, " out of bounds for seq_len ", shape[1]);
    auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 1, position, position + 1);
    return ov::Tensor(hidden_features, start_coord, end_coord);
}

/// @brief Copies one hidden-state row (one token) from src tensor at src_position into
///        dst tensor at dst_position.  Both tensors must have shape [1, *, hidden_size].
void copy_hidden_state_row(const ov::Tensor& src, size_t src_position, ov::Tensor& dst, size_t dst_position) {
    const size_t hidden_size = src.get_shape()[2];
    const float* src_ptr = src.data<const float>() + src_position * hidden_size;
    float* dst_ptr = dst.data<float>() + dst_position * hidden_size;
    std::copy_n(src_ptr, hidden_size, dst_ptr);
}

}  // anonymous namespace

namespace ov::genai {

// ---------------------------------------------------------------------------
// Eagle3InputBuilder
// ---------------------------------------------------------------------------

InputTensors Eagle3InputBuilder::build_prefill_inputs(size_t input_token_count) const {
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");
    const auto current_sequence = m_sequence_group->get_running_sequences().at(0);

    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = current_sequence->get_generated_ids();

    const size_t prompt_len = prompt_ids.size();
    const size_t total_len = prompt_len + generated_ids.size();
    const size_t start_pos = total_len - input_token_count;

    OPENVINO_ASSERT(input_token_count > 0 && input_token_count <= total_len,
                    "Invalid input_token_count: ",
                    input_token_count,
                    ", total_len: ",
                    total_len);

    InputTensors result;
    result.input_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    result.position_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    int64_t* ids_ptr = result.input_ids.data<int64_t>();
    int64_t* pos_ptr = result.position_ids.data<int64_t>();

    // Fill input_ids and position_ids from prompt and/or generated tokens.
    if (start_pos < prompt_len) {
        const size_t prompt_count = std::min(input_token_count, prompt_len - start_pos);
        std::copy_n(prompt_ids.data() + start_pos, prompt_count, ids_ptr);
        std::iota(pos_ptr, pos_ptr + prompt_count, static_cast<int64_t>(start_pos));
        if (input_token_count > prompt_count) {
            const size_t generated_count = input_token_count - prompt_count;
            std::copy_n(generated_ids.data(), generated_count, ids_ptr + prompt_count);
            std::iota(pos_ptr + prompt_count,
                      pos_ptr + prompt_count + generated_count,
                      static_cast<int64_t>(prompt_len));
        }
    } else {
        const size_t generated_start = start_pos - prompt_len;
        std::copy_n(generated_ids.data() + generated_start, input_token_count, ids_ptr);
        std::iota(pos_ptr, pos_ptr + input_token_count, static_cast<int64_t>(prompt_len + generated_start));
    }

    // Attention mask: all-ones covering [0, last_position+1).
    const size_t attn_len = static_cast<size_t>(pos_ptr[input_token_count - 1] + 1);
    result.attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
    std::fill_n(result.attention_mask.data<int64_t>(), attn_len, 1);

    // Prefill and DRAFT_INITIAL use a dummy minimal tree mask (no tree attention needed).
    result.eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, 1, 1});
    result.eagle_tree_mask.data<float>()[0] = 0.0f;

    return result;
}

InputTensors Eagle3InputBuilder::build_draft_iteration_inputs(size_t past_accepted_token_count) const {
    // DRAFT_ITERATION: all running sequences (beam paths) are concatenated into a single flat
    // sequence so that one infer call handles all paths simultaneously.
    //
    // Draft generated_ids layout across speculative iterations:
    //   [hist_0, ..., hist_{past_accepted_token_count-2}, root, branch_0, branch_1, ...]
    //
    // past_accepted_token_count equals the number of tokens already in the target's
    // generated_ids at the start of this speculative window.  The draft mirrors the target's
    // accepted history, so the first past_accepted_token_count entries are in the KV cache.
    // Only branch tokens starting at index past_accepted_token_count are new.
    //
    // Example: past_accepted_token_count=2, branching_factor=3, prompt_len=15
    //   generated_ids: [hist_0, root, 271, 106287, 99692]
    //   input_ids    : [271, 106287, 99692]       shape {1, 3}
    //   history_len  : prompt_len + past_accepted_token_count = 17
    //   position_ids : [17, 17, 17]
    //   attention_mask: {1, 20} all-ones
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");
    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const size_t prompt_len = prompt_ids.size();

    const auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t num_seqs = running_sequences.size();
    OPENVINO_ASSERT(num_seqs > 0, "No running sequences");

    const size_t full_path_len = running_sequences[0]->get_generated_ids().size();
    OPENVINO_ASSERT(full_path_len > past_accepted_token_count,
                    "Expected generated_ids length > past_accepted_token_count in DRAFT_ITERATION, got ",
                    full_path_len,
                    " with past_accepted_token_count=",
                    past_accepted_token_count);
    const size_t branch_len = full_path_len - past_accepted_token_count;
    const size_t total_tokens = num_seqs * branch_len;

    // history_len: prompt + past_accepted_token_count (history tokens + root), all in KV cache.
    const size_t history_len = prompt_len + past_accepted_token_count;

    InputTensors result;

    // 1. input_ids and position_ids — concatenate branch tokens of all paths.
    result.input_ids = ov::Tensor(ov::element::i64, {1, total_tokens});
    result.position_ids = ov::Tensor(ov::element::i64, {1, total_tokens});
    int64_t* ids_ptr = result.input_ids.data<int64_t>();
    int64_t* pos_ptr = result.position_ids.data<int64_t>();

    for (size_t s = 0; s < num_seqs; ++s) {
        const auto& gen = running_sequences[s]->get_generated_ids();
        OPENVINO_ASSERT(gen.size() == full_path_len,
                        "Sequence ",
                        s,
                        " has length ",
                        gen.size(),
                        " but expected ",
                        full_path_len);
        for (size_t t = 0; t < branch_len; ++t) {
            ids_ptr[s * branch_len + t] = gen[past_accepted_token_count + t];
            pos_ptr[s * branch_len + t] = static_cast<int64_t>(history_len + t);
        }
    }

    // 2. attention_mask — covers history + all branch tokens, all-ones.
    const size_t attn_len = history_len + total_tokens;
    result.attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
    std::fill_n(result.attention_mask.data<int64_t>(), attn_len, 1);

    // 3. eagle_tree_mask — shape {1, 1, total_tokens, attn_len}.
    //    History columns [0, history_len): 0 (prompt + past accepted + root, all accessible).
    //    Branch columns  [history_len, attn_len): -INF by default; each row opens only
    //    the columns belonging to its own path that are causally at or before it.
    result.eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, total_tokens, attn_len});
    float* mask_ptr = result.eagle_tree_mask.data<float>();
    std::fill_n(mask_ptr, total_tokens * attn_len, -std::numeric_limits<float>::infinity());

    for (size_t s = 0; s < num_seqs; ++s) {
        for (size_t t = 0; t < branch_len; ++t) {
            const size_t row = s * branch_len + t;
            float* row_ptr = mask_ptr + row * attn_len;

            // History region: all rows can attend.
            std::fill_n(row_ptr, history_len, 0.0f);

            // Branch region: open own-path columns up to and including t.
            const size_t path_col_start = history_len + s * branch_len;
            for (size_t t2 = 0; t2 <= t; ++t2) {
                row_ptr[path_col_start + t2] = 0.0f;
            }
        }
    }

    return result;
}

InputTensors Eagle3InputBuilder::build_target_validation_inputs() const {
    // TARGET_VALIDATION: feed all N+1 tree candidate tokens (root + N tree nodes) to the
    // target model using the exact tree attention structure stored in EagleMetaData.
    //
    // The root token is the LAST token output by the previous target step.  Although the
    // target model produced it, its KV has NOT been written to the target KV cache yet.
    // Therefore the root must be re-submitted as part of input_ids together with its children.
    //
    // EagleMetaData layout (set by select_top_k, indexed over all N+1 candidates):
    //   tree_mask[i][j] == 1  -> candidate j IS an ancestor of candidate i (attend allowed)
    //   tree_mask[i][j] == 0  -> not an ancestor (blocked -> -INF)
    //   tree_position_ids[i]  = tree depth of candidate i (root depth = 0)
    //
    // Attention layout:
    //   context_len = prompt_len + K  (K = tokens already in target KV cache)
    //   attn_len    = context_len + num_candidates
    //   eagle_tree_mask: {1, 1, N+1, attn_len}
    //     history  [0, context_len): 0.0
    //     tree     [context_len, attn_len): tree_mask[i][j]==1 -> 0.0, else -INF
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");
    const auto current_sequence = m_sequence_group->get_running_sequences().at(0);

    const size_t prompt_len = m_sequence_group->get_prompt_ids().size();
    const auto& generated_ids = current_sequence->get_generated_ids();
    const auto& metadata = current_sequence->get_eagle_metadata();
    const auto& tree_mask_bin = metadata.tree_mask;         // (N+1) x (N+1)
    const auto& tree_pos_ids = metadata.tree_position_ids;  // size = N+1 (root depth=0)

    OPENVINO_ASSERT(tree_mask_bin.size() >= 2,
                    "tree_mask must cover at least root + one tree node, got ",
                    tree_mask_bin.size());
    const size_t num_candidates = tree_mask_bin.size();

    OPENVINO_ASSERT(tree_pos_ids.size() == num_candidates,
                    "tree_position_ids size (",
                    tree_pos_ids.size(),
                    ") != num_candidates (",
                    num_candidates,
                    ")");
    OPENVINO_ASSERT(generated_ids.size() >= num_candidates,
                    "generated_ids has ",
                    generated_ids.size(),
                    " entries but expected at least ",
                    num_candidates,
                    " tokens (root + N tree nodes)");

    // context_len: prompt + previously accepted tokens already in the KV cache.
    const size_t past_accepted_len = generated_ids.size() - num_candidates;
    const size_t context_len = prompt_len + past_accepted_len;
    const size_t attn_len = context_len + num_candidates;

    InputTensors result;

    // 1. input_ids: all N+1 candidates — root then tree nodes.
    result.input_ids = ov::Tensor(ov::element::i64, {1, num_candidates});
    int64_t* ids_ptr = result.input_ids.data<int64_t>();
    const size_t gen_offset = generated_ids.size() - num_candidates;
    for (size_t i = 0; i < num_candidates; ++i) {
        ids_ptr[i] = generated_ids[gen_offset + i];
    }

    // 2. position_ids: context_len + tree depth of each candidate.
    result.position_ids = ov::Tensor(ov::element::i64, {1, num_candidates});
    int64_t* pos_ptr = result.position_ids.data<int64_t>();
    for (size_t i = 0; i < num_candidates; ++i) {
        pos_ptr[i] = static_cast<int64_t>(context_len) + static_cast<int64_t>(tree_pos_ids[i]);
    }

    // 3. attention_mask: all-ones.
    result.attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
    std::fill_n(result.attention_mask.data<int64_t>(), attn_len, 1);

    // 4. eagle_tree_mask: shape {1, 1, num_candidates, attn_len}.
    result.eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, num_candidates, attn_len});
    float* mask_ptr = result.eagle_tree_mask.data<float>();
    const float neg_inf = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < num_candidates; ++i) {
        float* row = mask_ptr + i * attn_len;
        std::fill_n(row, context_len, 0.0f);
        OPENVINO_ASSERT(tree_mask_bin[i].size() == num_candidates,
                        "tree_mask row ",
                        i,
                        " has ",
                        tree_mask_bin[i].size(),
                        " columns, expected ",
                        num_candidates);
        for (size_t j = 0; j < num_candidates; ++j) {
            row[context_len + j] = (tree_mask_bin[i][j] == 1) ? 0.0f : neg_inf;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Eagle3InferWrapperBase
// ---------------------------------------------------------------------------

Eagle3InferWrapperBase::Eagle3InferWrapperBase(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(model_desc.tokenizer),
      m_sampler(model_desc.tokenizer),
      m_sequence_group(nullptr) {
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);

    m_cache_types = utils::get_cache_types(*model_desc.model);
    OPENVINO_ASSERT(!m_cache_types.has_linear(),
        "Stateful speculative decoding does not support models with linear attention states. "
        "KV cache rollback would reset the entire state instead of trimming.");

    if (m_device == "NPU") {
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_axes_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_request = compiled.create_infer_request();
    } else {
        m_request =
            utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
    }

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
}

void Eagle3InferWrapperBase::append_tokens(const std::vector<int64_t>& tokens) {
    if (tokens.empty())
        return;

    const auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    for (const int64_t token : tokens) {
        current_sequence->append_token(token, 0.0f);
    }
}

void Eagle3InferWrapperBase::truncate_sequence(size_t size) {
    const auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const size_t prompt_len = m_sequence_group->get_prompt_len();
    const size_t current_len = prompt_len + current_sequence->get_generated_len();

    if (size < current_len) {
        OPENVINO_ASSERT(size >= prompt_len, "Cannot truncate prompt tokens");
        const size_t tokens_to_remove = current_len - size;
        current_sequence->remove_last_tokens(tokens_to_remove);
    }
}

void Eagle3InferWrapperBase::trim_kv_cache(size_t tokens_to_remove) {
    const size_t current_len = get_sequence_length();
    if (tokens_to_remove == 0 || current_len == 0) {
        return;
    }

    OPENVINO_ASSERT(tokens_to_remove > 0 && tokens_to_remove < current_len,
                    "Cannot trim ",
                    tokens_to_remove,
                    " tokens from ",
                    current_len,
                    " tokens. Valid range: 0 < tokens_to_remove < current_len");

    if (m_device != "NPU") {
        utils::CacheState state(m_cache_types);
        state.num_tokens_to_trim = tokens_to_remove;
        state.seq_length_axis = m_kv_axes_pos.seq_len;
        state.reset_mem_state = false;
        utils::trim_kv_cache(m_request, state, {});
    }
}

void Eagle3InferWrapperBase::set_npu_sampling_result(size_t num_candidates,
                                                     const std::vector<int64_t>& accepted_indices) {
    const size_t num_accepted = accepted_indices.size();

    constexpr const char* STATE_NAME = "npuw_eagle3_sampling_result";
    auto states = m_request.query_state();
    for (auto& state : states) {
        if (state.get_name() != STATE_NAME) {
            continue;
        }
        auto tensor = state.get_state();
        OPENVINO_ASSERT(tensor.get_element_type() == ov::element::i64,
                        STATE_NAME,
                        " VariableState must be of type int64");
        OPENVINO_ASSERT(tensor.get_size() >= 2 + num_candidates,
                        STATE_NAME,
                        " tensor capacity (",
                        tensor.get_size(),
                        ") is too small for num_candidates=",
                        num_candidates);

        int64_t* data = tensor.data<int64_t>();
        data[0] = static_cast<int64_t>(num_candidates);
        data[1] = static_cast<int64_t>(num_accepted);
        std::fill_n(data + 2, num_candidates, int64_t{0});
        for (const int64_t idx : accepted_indices) {
            OPENVINO_ASSERT(idx >= 0 && static_cast<size_t>(idx) < num_candidates,
                            "accepted_index ",
                            idx,
                            " is out of range [0, ",
                            num_candidates,
                            ")");
            data[2 + idx] = 1;
        }
        state.set_state(tensor);
        return;
    }

    // VariableState not found — expected on non-NPUW devices; silently ignore.
}

void Eagle3InferWrapperBase::reset_state() {
    if (m_sequence_group) {
        m_sampler.clear_request_info(m_sequence_group->get_request_id());
    }
    m_sequence_group = nullptr;

    m_request.reset_state();

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void Eagle3InferWrapperBase::release_memory() {
    m_request.get_compiled_model().release_memory();
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

void Eagle3InferWrapperBase::invoke_sampler(const ov::Tensor& logits,
                                            size_t input_token_count,
                                            size_t sample_count,
                                            bool is_validation) {
    const auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    sequence_group->schedule_tokens(input_token_count);
    sequence_group->set_output_seq_len(sample_count);
    sequence_group->set_num_validated_tokens(is_validation ? sample_count : 0);

    m_sampler.sample({sequence_group}, logits, is_validation);
    restore_running_status();
    sequence_group->finish_iteration();
}

void Eagle3InferWrapperBase::restore_running_status() {
    if (!m_sequence_group) {
        return;
    }
    for (const auto& seq : m_sequence_group->get_sequences()) {
        if (seq->has_finished()) {
            seq->set_status(SequenceStatus::RUNNING);
        }
    }
}

std::vector<int64_t> Eagle3InferWrapperBase::sample_draft_tokens(const ov::Tensor& logits, size_t input_token_count) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape: ", shape);
    OPENVINO_ASSERT(input_token_count > 0, "Invalid input_token_count");

    const auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    // Snapshot running sequences BEFORE sampling.  The sampler may fork a sequence into
    // multiple children; the original Sequence objects remain alive and their generated_ids
    // are only appended to, so the snapshot stays valid after m_sampler.sample().
    const auto running_sequences = sequence_group->get_running_sequences();
    const size_t num_sequences = running_sequences.size();
    OPENVINO_ASSERT(num_sequences > 0, "No running sequences");

    // Logits are expected to be pre-trimmed by the caller (forward()) so that
    // shape[1] == num_sequences.  The sampler reinterprets {1, K, V} as {K, 1, V}.
    OPENVINO_ASSERT(shape[1] == num_sequences,
                    "Logits seq_len (",
                    shape[1],
                    ") != num_sequences (",
                    num_sequences,
                    "). Caller must normalize logits before sampling.");

    invoke_sampler(logits, input_token_count, /*sample_count=*/1, /*is_validation=*/false);

    // Collect the last generated token from each sequence.
    std::vector<int64_t> result_tokens;
    result_tokens.reserve(num_sequences);
    for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        const auto& generated_ids = running_sequences[seq_idx]->get_generated_ids();
        OPENVINO_ASSERT(!generated_ids.empty(), "Sequence ", seq_idx, " has no generated tokens");
        result_tokens.push_back(generated_ids.back());
    }

    record_generated_tokens(num_sequences);
    return result_tokens;
}

std::vector<int64_t> Eagle3InferWrapperBase::sample_and_validate(const ov::Tensor& logits,
                                                                 size_t input_token_count,
                                                                 size_t num_candidates,
                                                                 size_t num_tokens_to_validate) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape: ", shape);
    OPENVINO_ASSERT(input_token_count > 0, "Invalid input_token_count");
    OPENVINO_ASSERT(num_tokens_to_validate > 0, "num_tokens_to_validate must be > 0 in validation mode");

    const auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    const auto running_sequences = sequence_group->get_running_sequences();
    OPENVINO_ASSERT(running_sequences.size() == 1, "Validation mode expects exactly one running sequence");

    // Record the root position before the sampler rewrites generated_ids.
    // generated_ids = [...history..., root, n_1, ..., n_N]  (N = num_tokens_to_validate)
    // After validate_tree_candidates: [...history..., root, acc_1, ..., acc_k, bonus]
    const auto& gen_before = running_sequences[0]->get_generated_ids();
    OPENVINO_ASSERT(gen_before.size() > num_tokens_to_validate,
                    "generated_ids too short before validation: ",
                    gen_before.size(),
                    " <= num_tokens_to_validate: ",
                    num_tokens_to_validate);
    const size_t root_index = gen_before.size() - num_tokens_to_validate - 1;

    // Logits are expected to be pre-trimmed by the caller (forward()) so that
    // shape[1] == num_candidates.
    OPENVINO_ASSERT(shape[1] == num_candidates,
                    "Logits seq_len (",
                    shape[1],
                    ") != num_candidates (",
                    num_candidates,
                    "). Caller must normalize logits before validation.");

    sequence_group->schedule_tokens(input_token_count);
    sequence_group->set_output_seq_len(num_candidates);
    sequence_group->set_num_validated_tokens(num_tokens_to_validate);

    m_sampler.sample({sequence_group}, logits, /*is_validation=*/true);
    restore_running_status();
    sequence_group->finish_iteration();

    // Extract [acc_1, ..., acc_k, bonus_token] = everything after the root.
    const auto& gen_after = running_sequences[0]->get_generated_ids();
    const size_t result_start = root_index + 1;
    OPENVINO_ASSERT(gen_after.size() > result_start,
                    "Validation result sequence too short: size=",
                    gen_after.size(),
                    ", expected > root_index+1=",
                    result_start);

    std::vector<int64_t> result_tokens(gen_after.begin() + static_cast<ptrdiff_t>(result_start), gen_after.end());
    record_generated_tokens(result_tokens.size());
    return result_tokens;
}

ov::Tensor Eagle3InferWrapperBase::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor Eagle3InferWrapperBase::get_hidden_features(size_t actual_seq_len) const {
    const auto hidden_state = m_request.get_tensor("last_hidden_state");
    const auto& shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid hidden state shape: ", shape);

    const size_t output_seq_len = shape[1];
    if (actual_seq_len == 0) {
        actual_seq_len = m_request.get_tensor("input_ids").get_shape()[1];
    }

    if (output_seq_len == actual_seq_len)
        return hidden_state;

    OPENVINO_ASSERT(actual_seq_len <= output_seq_len,
                    "Sequence length mismatch: actual=",
                    actual_seq_len,
                    ", output=",
                    output_seq_len);
    auto [start_coord, end_coord] =
        ov::genai::utils::make_roi(shape, 1, output_seq_len - actual_seq_len, output_seq_len);
    return ov::Tensor(hidden_state, start_coord, end_coord);
}

ov::Tensor Eagle3InferWrapperBase::trim_tensor_tail(const ov::Tensor& tensor, size_t useful_len) {
    const auto& shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1,
                    "trim_tensor_tail expects shape [1, seq_len, dim], got: ",
                    shape);
    const size_t output_len = shape[1];
    if (output_len == useful_len)
        return tensor;

    OPENVINO_ASSERT(useful_len > 0 && useful_len <= output_len,
                    "useful_len (",
                    useful_len,
                    ") out of range [1, ",
                    output_len,
                    "]");
    auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 1, output_len - useful_len, output_len);
    return ov::Tensor(tensor, start_coord, end_coord);
}

uint64_t Eagle3InferWrapperBase::execute_inference() {
    const auto start = std::chrono::steady_clock::now();
    m_request.infer();
    const auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    return static_cast<uint64_t>(duration_us);
}

void Eagle3InferWrapperBase::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
}

void Eagle3InferWrapperBase::record_generated_tokens(size_t actual_generated_count) {
    m_raw_perf_metrics.m_batch_sizes.emplace_back(actual_generated_count);
}

// ---------------------------------------------------------------------------
// Eagle3TargetWrapper
// ---------------------------------------------------------------------------

Eagle3TargetWrapper::Eagle3TargetWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {}

void Eagle3TargetWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto& shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len], got ", shape);

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = shape[1];
    OPENVINO_ASSERT(seq_len > 0, "Empty prompt");

    const TokenIds prompt_ids(ids_data, ids_data + seq_len);
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config, 0);

    OPENVINO_ASSERT(m_sequence_group->num_total_seqs() == 1,
                    "Expected single sequence after initialization, got ",
                    m_sequence_group->num_total_seqs());
}

InferenceOutput Eagle3TargetWrapper::infer(const InputTensors& inputs) {
    const size_t input_len = inputs.input_ids.get_shape()[1];

    if (m_device == "NPU") {
        OPENVINO_ASSERT(input_len <= m_max_prompt_len,
                        "NPU prompt length ",
                        input_len,
                        " exceeds max ",
                        m_max_prompt_len);
    }

    m_request.set_tensor("input_ids", inputs.input_ids);
    m_request.set_tensor("attention_mask", inputs.attention_mask);
    m_request.set_tensor("position_ids", inputs.position_ids);
    m_request.set_tensor("eagle_tree_mask", inputs.eagle_tree_mask);

    const uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    return InferenceOutput{get_logits(), get_hidden_features(input_len)};
}

InferResult Eagle3TargetWrapper::forward(const InferContext& ctx) {
    Eagle3InputBuilder builder(m_sequence_group);

    // Build inputs based on phase.
    const bool is_validation = ctx.num_tokens_to_validate > 0;
    const InputTensors inputs =
        is_validation ? builder.build_target_validation_inputs() : builder.build_prefill_inputs(ctx.input_token_count);

    // Infer.
    auto output = infer(inputs);

    // Normalize output: strip NPU padding from logits.
    // Prefill: only the last position is sampled → trim to 1.
    // Validation: all num_candidates positions are needed → trim to num_candidates.
    const size_t useful_logits_len = is_validation ? inputs.input_ids.get_shape()[1] : 1;
    output.logits = trim_tensor_tail(output.logits, useful_logits_len);

    // Sample.
    // During TARGET_PREFILL the sampler must produce exactly one greedy token.
    // The sequence group's config still has tree_depth > 0 (needed for the subsequent
    // speculative loop), which would make the sampler fork sequences via TreeSearcher.
    // Temporarily zero tree_depth so the sampler takes the greedy path; restore afterwards.
    std::vector<int64_t> sampled;
    if (!is_validation) {
        const auto saved_config = get_sequence_group()->get_sampling_parameters();
        auto prefill_config = saved_config;
        prefill_config.eagle_tree_params.tree_depth = 0;
        get_sequence_group()->set_sampling_parameters(prefill_config);

        sampled = sample_draft_tokens(output.logits, ctx.input_token_count);

        get_sequence_group()->set_sampling_parameters(saved_config);
    } else {
        const size_t num_candidates = inputs.input_ids.get_shape()[1];
        sampled = sample_and_validate(output.logits, ctx.input_token_count, num_candidates, ctx.num_tokens_to_validate);
    }

    // Store hidden states to sequence for the draft model.
    get_current_sequence()->update_hidden_state(output.hidden_features);

    return InferResult{std::move(output), std::move(sampled)};
}

// ---------------------------------------------------------------------------
// Eagle3DraftWrapper
// ---------------------------------------------------------------------------

Eagle3DraftWrapper::Eagle3DraftWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {}

void Eagle3DraftWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto& shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len], got ", shape);

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t total_len = shape[1];
    OPENVINO_ASSERT(total_len >= 2, "Draft model requires at least 2 tokens");

    // Draft model uses tokens[1:] (Eagle3 specific behavior).
    const TokenIds draft_prompt_ids(ids_data + 1, ids_data + total_len);
    m_sequence_group = std::make_shared<SequenceGroup>(1, draft_prompt_ids, config, 0);

    OPENVINO_ASSERT(m_sequence_group->num_total_seqs() == 1,
                    "Expected single sequence after initialization, got ",
                    m_sequence_group->num_total_seqs());
}

void Eagle3DraftWrapper::allocate_buffers(size_t max_sequences,
                                          size_t max_depth,
                                          size_t hidden_size,
                                          size_t vocab_size) {
    m_max_sequences = max_sequences;
    m_max_depth = max_depth;
    m_hidden_size = hidden_size;
    m_vocab_size = vocab_size;

    // Pre-allocate logits gathering buffer: {1, max_sequences, vocab_size}.
    m_logits_gather_buf = ov::Tensor(ov::element::f32, {1, max_sequences, vocab_size});

    // Pre-allocate hidden state concatenation buffer: {1, max_sequences * max_depth, hidden_size}.
    m_hidden_concat_buf = ov::Tensor(ov::element::f32, {1, max_sequences * max_depth, hidden_size});

    m_buffers_allocated = true;
}

InferenceOutput Eagle3DraftWrapper::infer(const InputTensors& inputs, const ov::Tensor& hidden_states) {
    const size_t input_token_count = inputs.input_ids.get_shape()[1];

    m_request.set_tensor("input_ids", inputs.input_ids);
    m_request.set_tensor("attention_mask", inputs.attention_mask);
    m_request.set_tensor("position_ids", inputs.position_ids);
    m_request.set_tensor("eagle_tree_mask", inputs.eagle_tree_mask);

    OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "hidden_states must be provided");
    const auto& hs_shape = hidden_states.get_shape();
    OPENVINO_ASSERT(hs_shape.size() == 3, "Invalid hidden states shape: ", hs_shape);
    m_request.set_tensor("hidden_states", hidden_states);

    const uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    return InferenceOutput{get_logits(), get_hidden_features(input_token_count)};
}

ov::Tensor Eagle3DraftWrapper::prepare_hidden_states(const InferContext& ctx) {
    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: use target model's hidden state for the single root position.
        OPENVINO_ASSERT(ctx.target_sequence, "target_sequence required when use_target_hidden=true");
        const auto hidden = ctx.target_sequence->get_hidden_state();
        OPENVINO_ASSERT(hidden && hidden.get_size() > 0, "Source sequence contains invalid hidden state");
        return hidden;
    }

    // DRAFT_ITERATION: concatenate the full stored hidden-state path of every sequence.
    // After DRAFT_INITIAL each sequence holds {1, 1, hidden_size} (the root).
    // After each subsequent iteration the stored tensor grows by one token per sequence,
    // so at iteration k each sequence holds {1, k, hidden_size}.
    // Concatenating all sequences yields {1, num_seqs * branch_len, hidden_size}, matching
    // the flat input_ids layout produced by build_draft_iteration_inputs.
    const auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t num_sequences = running_sequences.size();
    OPENVINO_ASSERT(num_sequences > 0, "No running sequences");

    // Calculate total sequence length.
    size_t total_seq_len = 0;
    for (size_t i = 0; i < num_sequences; ++i) {
        const auto seq_hidden = running_sequences[i]->get_hidden_state();
        OPENVINO_ASSERT(seq_hidden && seq_hidden.get_size() > 0, "Sequence ", i, " contains invalid hidden state");
        const auto& h_shape = seq_hidden.get_shape();
        OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1,
                        "Expected hidden state shape [1, branch_len, hidden_size], got: ",
                        h_shape);
        total_seq_len += h_shape[1];
    }

    // Use pre-allocated buffer if available and large enough, otherwise allocate.
    const size_t hidden_size = running_sequences[0]->get_hidden_state().get_shape()[2];
    ov::Tensor concat_buf;
    if (m_buffers_allocated && total_seq_len <= m_max_sequences * m_max_depth) {
        m_hidden_concat_buf.set_shape({1, total_seq_len, hidden_size});
        concat_buf = m_hidden_concat_buf;
    } else {
        concat_buf = ov::Tensor(ov::element::f32, {1, total_seq_len, hidden_size});
    }

    // Copy each sequence's hidden states into the contiguous buffer.
    size_t offset = 0;
    for (size_t i = 0; i < num_sequences; ++i) {
        const auto seq_hidden = running_sequences[i]->get_hidden_state();
        const size_t seq_len = seq_hidden.get_shape()[1];
        const size_t copy_elems = seq_len * hidden_size;
        std::memcpy(concat_buf.data<float>() + offset * hidden_size,
                    seq_hidden.data<const float>(),
                    copy_elems * sizeof(float));
        offset += seq_len;
    }

    return concat_buf;
}

void Eagle3DraftWrapper::update_hidden_states(const InferenceOutput& output, const InferContext& ctx) {
    const auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t num_sequences = running_sequences.size();

    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: all beam sequences start from the same root hidden state.
        const auto& h_shape = output.hidden_features.get_shape();
        const auto root_hidden = slice_hidden_state(output.hidden_features, h_shape[1] - 1);
        for (size_t i = 0; i < num_sequences; ++i) {
            running_sequences[i]->update_hidden_state(root_hidden);
        }
        return;
    }

    // DRAFT_ITERATION: output.hidden_features has shape {1, total_tokens, hidden_size}
    // where total_tokens = num_seqs * branch_len (flat layout, same order as input_ids).
    // Hidden features are already trimmed to input_token_count by get_hidden_features()
    // in infer(), so hidden_shape[1] == total_hidden_tokens (no pad offset needed).
    // Append each sequence's LAST branch token hidden state to its accumulated buffer,
    // growing from {1, k, H} to {1, k+1, H}.
    const auto& hidden_shape = output.hidden_features.get_shape();
    OPENVINO_ASSERT(hidden_shape.size() == 3 && hidden_shape[0] == 1, "Invalid hidden features shape: ", hidden_shape);

    const size_t branch_len = running_sequences[0]->get_hidden_state().get_shape()[1];
    const size_t total_hidden_tokens = num_sequences * branch_len;
    OPENVINO_ASSERT(hidden_shape[1] == total_hidden_tokens,
                    "hidden_features seq_len (",
                    hidden_shape[1],
                    ") != num_seqs (",
                    num_sequences,
                    ") * branch_len (",
                    branch_len,
                    ")");
    const size_t hidden_size = hidden_shape[2];
    const size_t new_len = branch_len + 1;

    for (size_t i = 0; i < num_sequences; ++i) {
        // The last branch token for sequence i sits at (i+1)*branch_len - 1.
        const size_t last_tok_pos = (i + 1) * branch_len - 1;

        // Allocate {1, branch_len+1, H} and copy existing + new row.
        ov::Tensor updated(ov::element::f32, {1, new_len, hidden_size});
        const auto existing = running_sequences[i]->get_hidden_state();

        // Copy existing branch_len rows.
        std::memcpy(updated.data(), existing.data(), branch_len * hidden_size * sizeof(float));
        // Copy the new token's hidden state as the last row.
        copy_hidden_state_row(output.hidden_features, last_tok_pos, updated, branch_len);

        running_sequences[i]->update_hidden_state(updated);
    }
}

ov::Tensor Eagle3DraftWrapper::gather_logits_for_sampling(const InferenceOutput& output, const InferContext& ctx) {
    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: logits already trimmed to {1, 1, vocab_size} by forward().
        return output.logits;
    }

    // DRAFT_ITERATION: logits already trimmed to {1, total_tokens, vocab_size} by forward().
    // Gather the last-branch-token logit of each sequence into {1, num_seqs, vocab_size}.
    const auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t num_sequences = running_sequences.size();

    const auto& lshape = output.logits.get_shape();
    OPENVINO_ASSERT(lshape.size() == 3 && lshape[0] == 1, "Invalid logits shape: ", lshape);

    // The stored hidden state was just grown to branch_len+1, so prev_branch_len = shape[1]-1.
    const size_t new_branch_len = running_sequences[0]->get_hidden_state().get_shape()[1];
    const size_t prev_branch_len = new_branch_len - 1;
    OPENVINO_ASSERT(prev_branch_len > 0, "branch_len must be positive in DRAFT_ITERATION");
    const size_t total_tokens = num_sequences * prev_branch_len;

    OPENVINO_ASSERT(lshape[1] == total_tokens,
                    "Logits seq_len (",
                    lshape[1],
                    ") != num_seqs (",
                    num_sequences,
                    ") * branch_len (",
                    prev_branch_len,
                    ")");
    const size_t vocab_size = lshape[2];

    // Use pre-allocated buffer if available.
    ov::Tensor gather_buf;
    if (m_buffers_allocated && num_sequences <= m_max_sequences && vocab_size == m_vocab_size) {
        m_logits_gather_buf.set_shape({1, num_sequences, vocab_size});
        gather_buf = m_logits_gather_buf;
    } else {
        gather_buf = ov::Tensor(ov::element::f32, {1, num_sequences, vocab_size});
    }

    const float* src = output.logits.data<const float>();
    float* dst = gather_buf.data<float>();

    for (size_t s = 0; s < num_sequences; ++s) {
        // Each sequence occupies [s*prev_branch_len, (s+1)*prev_branch_len) in the flat layout;
        // take the last position of each sequence's branch.
        const size_t flat_pos = (s + 1) * prev_branch_len - 1;
        std::copy_n(src + flat_pos * vocab_size, vocab_size, dst + s * vocab_size);
    }

    return gather_buf;
}

InferResult Eagle3DraftWrapper::forward(const InferContext& ctx) {
    Eagle3InputBuilder builder(m_sequence_group);

    // 1. Build input tensors.
    const InputTensors inputs = ctx.use_target_hidden
                                    ? builder.build_prefill_inputs(ctx.input_token_count)
                                    : builder.build_draft_iteration_inputs(ctx.past_accepted_token_count);

    // 2. Prepare hidden states.
    const ov::Tensor hidden_states = prepare_hidden_states(ctx);

    // 3. Infer.
    auto output = infer(inputs, hidden_states);

    // Normalize output: strip NPU padding from logits.
    // DRAFT_INITIAL: only last 1 position is sampled → trim to 1.
    // DRAFT_ITERATION: all total_tokens positions are needed → trim to input_ids length.
    const size_t useful_logits_len = ctx.use_target_hidden ? 1 : inputs.input_ids.get_shape()[1];
    output.logits = trim_tensor_tail(output.logits, useful_logits_len);

    // 4. Update per-sequence hidden states (before sampling, so forks get correct state).
    update_hidden_states(output, ctx);

    // 5. Gather logits and sample.
    const ov::Tensor logits = gather_logits_for_sampling(output, ctx);
    auto sampled = sample_draft_tokens(logits, ctx.input_token_count);

    return InferResult{std::move(output), std::move(sampled)};
}

// ---------------------------------------------------------------------------
// StatefulEagle3LLMPipeline
// ---------------------------------------------------------------------------

StatefulEagle3LLMPipeline::StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config) {
    // eagle_tree_params are read from draft_model_desc.generation_config.
    // m_generation_config is initialised from target_model_desc (eos_token_id, etc.).
    m_generation_config.eagle_tree_params = draft_model_desc.generation_config.eagle_tree_params;

    // Apply compile-time defaults when no eagle_tree_params were provided by the user or
    // the draft model JSON (tree_depth == 0 is the zero-initialised struct default).
    ensure_eagle_tree_params_is_set(m_generation_config);

    OPENVINO_ASSERT(m_generation_config.is_tree_search(), "Eagle3 pipeline requires eagle_tree_params.tree_depth > 0.");

    // Extract hidden_layers_list from draft model properties — used only during model
    // construction to wire the target model's hidden-state extraction.
    OPENVINO_ASSERT(draft_model_desc.properties.find("hidden_layers_list") != draft_model_desc.properties.end(),
                    "hidden_layers_list must be present in draft model properties");

    const auto hidden_layers_to_abstract =
        draft_model_desc.properties.at("hidden_layers_list").as<std::vector<int32_t>>();

    OPENVINO_ASSERT(hidden_layers_to_abstract.size() == 3,
                    "Eagle3 requires exactly three layers for feature extraction, got: " +
                        std::to_string(hidden_layers_to_abstract.size()) +
                        ". Please ensure 'hidden_layers_list' is properly configured in draft model properties.");

    auto target_model = target_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(target_model, "Target model must not be null");
    OPENVINO_ASSERT(draft_model, "Draft model must not be null");

    // --- Model graph transformations ---
    utils::eagle3::share_vocabulary(target_model, draft_model);

    auto d2t_mapping = utils::eagle3::extract_d2t_mapping_table(draft_model);
    OPENVINO_ASSERT(d2t_mapping && d2t_mapping->get_element_type() == ov::element::i64, "Invalid d2t mapping tensor");

    utils::eagle3::apply_eagle3_attention_mask_transform(draft_model);
    utils::eagle3::apply_eagle3_attention_mask_transform(target_model);

    utils::eagle3::transform_hidden_state(target_model, hidden_layers_to_abstract);
    utils::eagle3::move_fc_from_draft_to_main(draft_model, target_model);
    utils::eagle3::transform_hidden_state(draft_model, {-1});

    // --- Compute validation windows ---
    // target_validation_window: number of candidate tokens the target model validates in one
    // step — all N tree nodes (num_speculative_tokens), including the root token.
    const size_t target_validation_window = m_generation_config.eagle_tree_params.num_speculative_tokens;

    // draft_validation_window: maximum number of tokens the draft model processes in a single
    // DRAFT_ITERATION pass.  At each iteration all running sequences each contribute one token
    // so the worst case is at the last pass: (tree_depth - 1) * branching_factor.
    m_draft_iterations = m_generation_config.eagle_tree_params.tree_depth;
    const size_t draft_validation_window =
        (m_draft_iterations - 1) * m_generation_config.eagle_tree_params.branching_factor;

    // --- Configure and create draft model ---
    auto draft_desc = draft_model_desc;
    if (draft_desc.device == "NPU") {
        draft_desc.properties["NPUW_EAGLE"] = "TRUE";
        draft_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = draft_validation_window;
        draft_desc.properties["NPUW_ONLINE_PIPELINE"] = "NONE";
        draft_desc.properties["NPUW_DEVICES"] = "CPU";
    }
    m_draft = std::make_unique<Eagle3DraftWrapper>(draft_desc);
    m_draft->set_draft_target_mapping(d2t_mapping);

    // --- Configure and create target model ---
    auto target_desc = target_model_desc;
    if (target_desc.device == "NPU") {
        target_desc.properties["NPUW_EAGLE"] = "TRUE";
        target_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = target_validation_window;
        target_desc.properties["NPUW_SLICE_OUT"] = "NO";
        target_desc.properties["NPUW_DEVICES"] = "CPU";
    }
    m_target = std::make_unique<Eagle3TargetWrapper>(target_desc);
}

StatefulEagle3LLMPipeline::~StatefulEagle3LLMPipeline() {
    m_target->release_memory();
    m_draft->release_memory();
}

void StatefulEagle3LLMPipeline::ensure_eagle_tree_params_is_set(GenerationConfig& config) {
    if (!config.is_tree_search()) {
        config.eagle_tree_params.branching_factor = DEFAULT_EAGLE_BRANCHING_FACTOR;
        config.eagle_tree_params.tree_depth = DEFAULT_EAGLE_TREE_DEPTH;
        config.eagle_tree_params.num_speculative_tokens = DEFAULT_EAGLE_NUM_SPECULATIVE_TOKENS;
    }
}

GenerationConfig StatefulEagle3LLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);
    ensure_eagle_tree_params_is_set(config);
    return config;
}

EncodedResults StatefulEagle3LLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate(EncodedInputs)");
    generate_timer.start();

    const auto streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    // Extract input tensors.
    ov::Tensor input_ids, attention_mask;
    if (const auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (const auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    OPENVINO_ASSERT(input_ids.get_shape()[0] == 1, "Only batch size 1 supported");
    m_prompt_length = input_ids.get_shape()[1];

    // Reset model states.
    m_target->reset_state();
    m_draft->reset_state();

    // Prepare sampling config with extended max_new_tokens to prevent premature termination
    // during draft generation. Actual length control is in the generation loop.
    auto sampling_config = config;
    sampling_config.max_new_tokens = config.max_new_tokens + m_draft_iterations + 2;

    m_draft->initialize_sequence(input_ids, sampling_config);
    m_target->initialize_sequence(input_ids, sampling_config);

    // --- Phase 1: Initial Prompt Processing (Prefill) ---
    InferContext prefill_ctx;
    prefill_ctx.input_token_count = m_prompt_length;
    const auto prefill_result = m_target->forward(prefill_ctx);
    OPENVINO_ASSERT(prefill_result.sampled_tokens.size() == 1, "Expected single token from prefill");
    const int64_t initial_token = prefill_result.sampled_tokens[0];

    m_draft->append_tokens({initial_token});

    // Try to allocate pre-allocated buffers for draft model now that hidden_size is known.
    if (!m_draft->buffers_allocated()) {
        const auto& h_shape = prefill_result.output.hidden_features.get_shape();
        if (h_shape.size() == 3) {
            const size_t hidden_size = h_shape[2];
            // Determine vocab_size from logits.
            const auto& l_shape = prefill_result.output.logits.get_shape();
            const size_t vocab_size = l_shape.size() == 3 ? l_shape[2] : 0;
            if (vocab_size > 0) {
                m_draft->allocate_buffers(m_generation_config.eagle_tree_params.branching_factor,
                                          m_draft_iterations,
                                          hidden_size,
                                          vocab_size);
            }
        }
    }

    auto streaming_status = stream_generated_tokens(streamer_ptr, {initial_token});

    // --- Phase 2: Speculative Decoding Loop ---
    size_t generated_tokens = 1;
    size_t total_draft_accepted = 0;
    size_t total_draft_generated = 0;
    bool eos_reached = false;

    size_t input_token_count = m_draft->get_sequence_length();

    while (!eos_reached && generated_tokens < config.max_new_tokens &&
           m_target->get_sequence_length() < m_prompt_length + config.max_new_tokens &&
           streaming_status == ov::genai::StreamingStatus::RUNNING) {
        auto result = run_speculative_iteration(input_token_count, static_cast<int64_t>(config.eos_token_id));

        // Truncate validated tokens if they would exceed max_new_tokens.
        const size_t remaining_budget = config.max_new_tokens - generated_tokens;
        if (result.validated_tokens.size() > remaining_budget) {
            result.validated_tokens.resize(remaining_budget);
            result.eos_reached = true;  // Force stop after this iteration.
        }

        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        // Update statistics.
        total_draft_generated += m_draft_iterations;
        total_draft_accepted += result.accepted_tokens_count;
        generated_tokens += result.validated_tokens.size();
        eos_reached = result.eos_reached;

        input_token_count = result.next_window_size;
    }

    // --- Phase 3: Finalization ---
    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr)
        streamer_ptr->end();

    EncodedResults results;
    results.tokens = {m_target->get_generated_tokens()};
    results.scores = {0.0f};

    generate_timer.end();

    // Update performance metrics.
    m_sd_perf_metrics.num_input_tokens = m_prompt_length;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;

    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_target->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft->get_raw_perf_metrics();

    if (total_draft_generated > 0) {
        const float acceptance_rate = static_cast<float>(total_draft_accepted) / total_draft_generated * 100.0f;
        m_sd_metrics.update_acceptance_rate(0, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens);
    }

    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    generate_timer.clear();

    return results;
}

// ---------------------------------------------------------------------------
// Speculative iteration sub-steps
// ---------------------------------------------------------------------------

InferResult StatefulEagle3LLMPipeline::generate_initial_draft(size_t input_token_count,
                                                              size_t past_accepted_token_count) {
    InferContext ctx;
    ctx.input_token_count = input_token_count;
    ctx.use_target_hidden = true;
    ctx.target_sequence = m_target->get_current_sequence();
    ctx.past_accepted_token_count = past_accepted_token_count;
    auto result = m_draft->forward(ctx);
    OPENVINO_ASSERT(!result.sampled_tokens.empty(), "Expected at least one token from initial draft");
    return result;
}

void StatefulEagle3LLMPipeline::expand_draft_tree(size_t past_accepted_token_count) {
    for (size_t i = 1; i < m_draft_iterations; ++i) {
        InferContext ctx;
        ctx.input_token_count = 1;
        ctx.use_target_hidden = false;
        ctx.past_accepted_token_count = past_accepted_token_count;
        const auto result = m_draft->forward(ctx);
        OPENVINO_ASSERT(!result.sampled_tokens.empty(), "Expected tokens from draft iteration ", i);
    }
}

ValidationResult StatefulEagle3LLMPipeline::validate_draft_with_target() {
    // In tree-search mode the draft model generates a tree of N+1 candidates:
    //   draft generated_ids tail = [root, node_1, ..., node_N]
    //   EagleMetaData.tree_mask  = (N+1) x (N+1), index 0 = root.
    // The root token is the LAST token sampled by the target in the previous iteration.
    // It has NOT been fed back to the target KV cache yet, so the target must process
    // all N+1 candidates (root + N tree nodes) during validation.
    const auto& draft_gen_ids = m_draft->get_current_sequence()->get_generated_ids();
    const auto& draft_metadata = m_draft->get_current_sequence()->get_eagle_metadata();
    const size_t num_candidates = draft_metadata.tree_mask.size();
    OPENVINO_ASSERT(num_candidates >= 2,
                    "Expected at least 2 candidates (root + at least 1 tree node), got: ",
                    num_candidates);
    const size_t num_tree_nodes = num_candidates - 1;

    // Sync EagleMetaData and candidate tokens from DRAFT -> TARGET.
    m_target->get_current_sequence()->set_eagle_metadata(draft_metadata);

    OPENVINO_ASSERT(draft_gen_ids.size() >= num_candidates,
                    "draft generated_ids too short: ",
                    draft_gen_ids.size(),
                    " < num_candidates: ",
                    num_candidates);
    // The root is already in the target sequence; append only node_1..node_N.
    const size_t draft_tail_offset = draft_gen_ids.size() - num_candidates;
    for (size_t i = 1; i < num_candidates; ++i) {
        m_target->get_current_sequence()->append_token(draft_gen_ids[draft_tail_offset + i], 0.0f);
    }

    // Run target validation.
    InferContext val_ctx;
    val_ctx.input_token_count = num_candidates;
    val_ctx.sample_count = num_candidates;
    val_ctx.num_tokens_to_validate = num_tree_nodes;
    const auto val_result = m_target->forward(val_ctx);

    ValidationResult result;
    result.validated_tokens = val_result.sampled_tokens;
    result.accepted_count = result.validated_tokens.size() - 1;
    result.num_candidates = num_candidates;
    result.output = val_result.output;

    OPENVINO_ASSERT(result.validated_tokens.size() <= num_candidates,
                    "Sampler returned more tokens (",
                    result.validated_tokens.size(),
                    ") than candidates (",
                    num_candidates,
                    ")");

    return result;
}

void StatefulEagle3LLMPipeline::synchronize_after_validation(const ValidationResult& validation,
                                                             size_t pre_draft_token_len) {
    const size_t total_accepted_tokens = validation.validated_tokens.size();
    const size_t tokens_to_remove = validation.num_candidates - total_accepted_tokens;

    // Sync draft model's sequence.
    m_draft->truncate_sequence(pre_draft_token_len);
    m_draft->append_tokens(validation.validated_tokens);

    // KV cache management after tree-search validation:
    //
    // TARGET (NPU): Accepted candidates form a non-contiguous path through the tree.
    //   Write the acceptance mask into "npuw_eagle3_sampling_result" VariableState.
    //   NPUW reads this before the next infer call and retains matching KV entries.
    //
    // DRAFT (NPU): No explicit KV cache management needed.  In the next iteration, all
    //   accepted tokens are re-submitted as input_ids with correct position_ids.
    //
    // NON-NPU: Trim the tail to remove rejected positions.
    if (m_target->device() == "NPU") {
        const auto& validated_indices = m_target->get_current_sequence()->get_eagle_metadata().validated_indices;
        m_target->set_npu_sampling_result(validation.num_candidates, validated_indices);
    } else if (tokens_to_remove > 0) {
        m_target->trim_kv_cache(tokens_to_remove);
        m_draft->trim_kv_cache(tokens_to_remove);
    }
}

void StatefulEagle3LLMPipeline::gather_accepted_hidden_states(const ValidationResult& validation) {
    // current_hidden has shape {1, num_candidates, H}, where the seq dimension is a flat
    // enumeration of all N+1 tree candidates submitted to the target model.
    //
    // validated_indices contains the tree-position indices of the accepted path,
    // e.g. [0, 2, 5] = root + nodes at indices 2 and 5.  These are NOT contiguous.
    //
    // Gather the hidden-state rows corresponding to accepted positions to form
    // {1, total_accepted_tokens, H} for the next draft-model pass.
    const auto& current_hidden = validation.output.hidden_features;
    OPENVINO_ASSERT(current_hidden && current_hidden.get_size() > 0, "Missing hidden features");

    const auto& h_shape = current_hidden.get_shape();
    OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1, "Invalid hidden state shape: ", h_shape);

    const auto& validated_indices = m_target->get_current_sequence()->get_eagle_metadata().validated_indices;
    const size_t total_accepted_tokens = validation.validated_tokens.size();
    OPENVINO_ASSERT(validated_indices.size() == total_accepted_tokens,
                    "validated_indices size (",
                    validated_indices.size(),
                    ") != total_accepted_tokens (",
                    total_accepted_tokens,
                    ")");

    const size_t hidden_size = h_shape[2];
    ov::Tensor next_hidden(ov::element::f32, {1, total_accepted_tokens, hidden_size});
    const float* src = current_hidden.data<const float>();
    float* dst = next_hidden.data<float>();

    for (size_t i = 0; i < total_accepted_tokens; ++i) {
        const size_t row = static_cast<size_t>(validated_indices[i]);
        OPENVINO_ASSERT(row < h_shape[1],
                        "validated_indices[",
                        i,
                        "] = ",
                        row,
                        " is out of range [0, ",
                        h_shape[1],
                        ")");
        std::copy_n(src + row * hidden_size, hidden_size, dst + i * hidden_size);
    }
    m_target->get_current_sequence()->update_hidden_state(next_hidden);
}

StatefulEagle3LLMPipeline::SpeculativeResult StatefulEagle3LLMPipeline::run_speculative_iteration(
    size_t input_token_count,
    int64_t eos_token_id) {
    OPENVINO_ASSERT(m_target->get_sequence_group() && m_draft->get_sequence_group(),
                    "Eagle3 speculative iteration requires initialized sequence groups");

    const auto target_hidden = m_target->get_current_sequence()->get_hidden_state();
    OPENVINO_ASSERT(target_hidden && target_hidden.get_size() > 0,
                    "Target model contains invalid hidden state for speculation");

    const size_t pre_draft_token_len = m_draft->get_sequence_length();
    const size_t past_accepted_token_count = m_target->get_current_sequence()->get_generated_ids().size();

    // Clear stale EagleMetaData from the previous iteration.
    m_draft->get_current_sequence()->set_eagle_metadata({});

    // Step 1: Generate first draft token using target hidden states (DRAFT_INITIAL).
    generate_initial_draft(input_token_count, past_accepted_token_count);

    // Step 2: Expand the draft tree for (m_draft_iterations - 1) more levels.
    expand_draft_tree(past_accepted_token_count);

    // Step 3: Validate draft candidates with the target model.
    auto validation = validate_draft_with_target();

    // Step 4: Synchronize sequences and KV caches.
    synchronize_after_validation(validation, pre_draft_token_len);

    // Step 5: Gather accepted hidden states for next iteration.
    gather_accepted_hidden_states(validation);

    const int64_t target_predicted_token = validation.validated_tokens.back();

    SpeculativeResult result;
    result.accepted_tokens_count = validation.accepted_count;
    result.next_window_size = validation.accepted_count + 1;
    result.validated_tokens = std::move(validation.validated_tokens);
    result.eos_reached = (target_predicted_token == eos_token_id);
    return result;
}

void StatefulEagle3LLMPipeline::finish_chat() {
    StatefulSpeculativePipelineBase::finish_chat();
}

SpeculativeDecodingMetrics StatefulEagle3LLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

}  // namespace ov::genai
