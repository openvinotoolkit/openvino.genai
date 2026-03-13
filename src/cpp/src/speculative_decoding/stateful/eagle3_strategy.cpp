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

ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

/// Extracts hidden state at a specific position; returns a tensor view [1, 1, hidden_size].
ov::Tensor slice_hidden_state_at_position(const ov::Tensor& hidden_features, size_t position) {
    OPENVINO_ASSERT(hidden_features.get_size() > 0, "Hidden features tensor is empty");
    const auto shape = hidden_features.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] > 0, "Expected shape [1, seq_len, hidden_size]");
    OPENVINO_ASSERT(position < shape[1], "Position ", position, " out of bounds for seq_len ", shape[1]);
    auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 1, position, position + 1);
    return ov::Tensor(hidden_features, start_coord, end_coord);
}

/// Concatenates hidden state tensors along the sequence dimension; returns [1, sum(seq_len), hidden_size].
ov::Tensor concatenate_hidden_states(const std::vector<ov::Tensor>& hidden_states) {
    OPENVINO_ASSERT(!hidden_states.empty(), "Cannot concatenate empty vector of hidden states");
    if (hidden_states.size() == 1) {
        return hidden_states[0];
    }
    const auto& first_shape = hidden_states[0].get_shape();
    const auto element_type = hidden_states[0].get_element_type();
    OPENVINO_ASSERT(first_shape.size() == 3 && first_shape[0] == 1, "Expected shape [1, seq_len, hidden_size]");
    const size_t hidden_size = first_shape[2];
    size_t total_seq_len = 0;
    for (const auto& tensor : hidden_states) {
        const auto& shape = tensor.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[2] == hidden_size,
                        "All tensors must have compatible shapes [1, *, ",
                        hidden_size,
                        "]");
        OPENVINO_ASSERT(tensor.get_element_type() == element_type, "All tensors must have the same element type");
        total_seq_len += shape[1];
    }
    ov::Tensor result(element_type, {1, total_seq_len, hidden_size});
    size_t current_offset = 0;
    for (const auto& tensor : hidden_states) {
        const size_t seq_len = tensor.get_shape()[1];
        const size_t copy_size = seq_len * hidden_size * element_type.size();
        std::memcpy(static_cast<char*>(result.data()) + current_offset * hidden_size * element_type.size(),
                    tensor.data(),
                    copy_size);
        current_offset += seq_len;
    }
    return result;
}

}  // anonymous namespace

namespace ov::genai {

Eagle3InferWrapperBase::Eagle3InferWrapperBase(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(model_desc.tokenizer),
      m_sampler(model_desc.tokenizer) {
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

    // Initialize performance metrics
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};

    m_sequence_group = nullptr;
}

void Eagle3InferWrapperBase::append_tokens(const std::vector<int64_t>& tokens) {
    if (tokens.empty())
        return;

    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    for (auto token : tokens) {
        current_sequence->append_token(token, 0.0f);
    }
}

void Eagle3InferWrapperBase::truncate_sequence(size_t size) {
    auto current_sequence = get_current_sequence();
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
        // Zero-initialize all candidate slots, then mark accepted positions.
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

    // VariableState not found — this is expected on non-NPUW devices; silently ignore.
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

void Eagle3InferWrapperBase::build_inputs_for_prefill(size_t input_token_count,
                                                      ov::Tensor& input_ids,
                                                      ov::Tensor& attention_mask,
                                                      ov::Tensor& position_ids,
                                                      ov::Tensor& eagle_tree_mask) {
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

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

    input_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    position_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    int64_t* ids_ptr = input_ids.data<int64_t>();
    int64_t* pos_ptr = position_ids.data<int64_t>();

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

    const size_t attn_len = static_cast<size_t>(pos_ptr[input_token_count - 1] + 1);
    attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
    std::fill_n(attention_mask.data<int64_t>(), attn_len, 1);

    // Prefill and DRAFT_INITIAL use a dummy minimal tree mask (no tree attention needed).
    eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, 1, 1});
    eagle_tree_mask.data<float>()[0] = 0.0f;
}

void Eagle3InferWrapperBase::build_inputs_for_draft_iteration(size_t past_accepted_token_count,
                                                              ov::Tensor& input_ids,
                                                              ov::Tensor& attention_mask,
                                                              ov::Tensor& position_ids,
                                                              ov::Tensor& eagle_tree_mask) {
    // DRAFT_ITERATION: all running sequences (beam paths) are concatenated into a single flat
    // sequence so that one infer call handles all paths simultaneously.
    //
    // draft generated_ids layout across speculative iterations:
    //   [hist_0, ..., hist_{past_accepted_token_count-2}, root, branch_0, branch_1, ...]
    //
    // past_accepted_token_count equals the number of tokens already in the target's
    // generated_ids at the start of this speculative window (tokens accepted in all prior
    // iterations, INCLUDING the root token produced by DRAFT_INITIAL).
    // The draft mirrors the target's accepted history, so the first past_accepted_token_count
    // entries of the draft's generated_ids are already in the KV cache.
    // Only the branch tokens starting at index past_accepted_token_count are new.
    //
    // Example: past_accepted_token_count=2, branching_factor=3, prompt_len=15
    //   generated_ids: [hist_0, root, 271, 106287, 99692]
    //   input_ids    : [271, 106287, 99692]       shape {1, 3}
    //   history_len  : prompt_len + past_accepted_token_count = 17
    //   position_ids : [17, 17, 17]
    //   attention_mask: {1, 20} all-ones
    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const size_t prompt_len = prompt_ids.size();

    auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t num_seqs = running_sequences.size();
    OPENVINO_ASSERT(num_seqs > 0, "No running sequences");

    // The first past_accepted_token_count entries (history + root) are already in the KV cache.
    // Branch tokens start at index past_accepted_token_count.
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

    // 1. input_ids and position_ids — concatenate branch tokens of all paths.
    input_ids = ov::Tensor(ov::element::i64, {1, total_tokens});
    position_ids = ov::Tensor(ov::element::i64, {1, total_tokens});
    int64_t* ids_ptr = input_ids.data<int64_t>();
    int64_t* pos_ptr = position_ids.data<int64_t>();

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
            // Skip the first past_accepted_token_count tokens (history + root, all in KV cache).
            ids_ptr[s * branch_len + t] = gen[past_accepted_token_count + t];
            // Root sits at position history_len-1; branch token t sits at history_len+t.
            pos_ptr[s * branch_len + t] = static_cast<int64_t>(history_len + t);
        }
    }

    // 2. attention_mask — covers history + all branch tokens, all-ones.
    const size_t attn_len = history_len + total_tokens;
    attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
    std::fill_n(attention_mask.data<int64_t>(), attn_len, 1);

    // 3. eagle_tree_mask — shape {1, 1, total_tokens, attn_len}.
    //    History columns [0, history_len): 0 (prompt + past accepted + root, all accessible).
    //    Branch columns  [history_len, attn_len): -INF by default; each row opens only
    //    the columns belonging to its own path that are causally at or before it.
    eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, total_tokens, attn_len});
    float* mask_ptr = eagle_tree_mask.data<float>();
    std::fill_n(mask_ptr, total_tokens * attn_len, -std::numeric_limits<float>::infinity());

    for (size_t s = 0; s < num_seqs; ++s) {
        for (size_t t = 0; t < branch_len; ++t) {
            const size_t row = s * branch_len + t;
            float* row_ptr = mask_ptr + row * attn_len;

            // History region (prompt + past accepted + root): all rows can attend.
            std::fill_n(row_ptr, history_len, 0.0f);

            // Branch region: open own-path columns up to and including t.
            const size_t path_col_start = history_len + s * branch_len;
            for (size_t t2 = 0; t2 <= t; ++t2) {
                row_ptr[path_col_start + t2] = 0.0f;
            }
        }
    }
}

void Eagle3InferWrapperBase::build_inputs_for_target_validation(ov::Tensor& input_ids,
                                                                ov::Tensor& attention_mask,
                                                                ov::Tensor& position_ids,
                                                                ov::Tensor& eagle_tree_mask) {
    // TARGET_VALIDATION: feed all N+1 tree candidate tokens (root + N tree nodes) to the
    // target model using the exact tree attention structure stored in EagleMetaData.
    //
    // Key insight: the root token is the LAST token output by the previous target step.
    // Although the target model produced it, its KV has NOT been written to the target
    // KV cache yet — the target processed only the prompt during prefill and previously
    // accepted tokens during prior iterations.  Therefore the root must be re-submitted
    // as part of input_ids together with its N children.
    //
    // EagleMetaData layout (set by select_top_k, indexed over all N+1 candidates):
    //   tree_mask[i][j] == 1  → candidate j IS an ancestor of candidate i (attend allowed)
    //   tree_mask[i][j] == 0  → not an ancestor (blocked → -INF)
    //   tree_position_ids[i]  = tree depth of candidate i (root depth = 0, children depth > 0)
    //   Both are size N+1, index 0 = root.
    //
    // generated_ids layout (after syncing draft candidates to target):
    //   [hist_0, ..., hist_{K-1}, root, node_1, ..., node_N]
    //   K = generated_ids.size() - num_candidates  (tokens already in target KV cache)
    //
    // Attention layout:
    //   context_len = prompt_len + K
    //   attn_len    = context_len + num_candidates
    //   input_ids    : {1, N+1}
    //   position_ids : {1, N+1} = context_len + tree_position_ids[i]
    //   attention_mask: {1, attn_len} all-ones
    //   eagle_tree_mask: {1, 1, N+1, attn_len}
    //     history  [0, context_len): 0.0
    //     tree     [context_len, attn_len): tree_mask[i][j]==1 → 0.0, else -INF
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const size_t prompt_len = m_sequence_group->get_prompt_ids().size();
    const auto& generated_ids = current_sequence->get_generated_ids();
    const auto& metadata = current_sequence->get_eagle_metadata();
    const auto& tree_mask_bin = metadata.tree_mask;         // (N+1)×(N+1)
    const auto& tree_pos_ids = metadata.tree_position_ids;  // size = N+1 (root depth=0)

    OPENVINO_ASSERT(tree_mask_bin.size() >= 2,
                    "tree_mask must cover at least root + one tree node, got ",
                    tree_mask_bin.size());
    const size_t num_candidates = tree_mask_bin.size();  // N+1 (root + N nodes)

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
    // The last num_candidates entries of generated_ids are the current candidates (not in KV cache).
    const size_t past_accepted_len = generated_ids.size() - num_candidates;
    const size_t context_len = prompt_len + past_accepted_len;
    const size_t attn_len = context_len + num_candidates;

    // 1. input_ids: all N+1 candidates — root then tree nodes.
    input_ids = ov::Tensor(ov::element::i64, {1, num_candidates});
    int64_t* ids_ptr = input_ids.data<int64_t>();
    const size_t gen_offset = generated_ids.size() - num_candidates;
    for (size_t i = 0; i < num_candidates; ++i) {
        ids_ptr[i] = generated_ids[gen_offset + i];
    }

    // 2. position_ids: context_len + tree depth of each candidate.
    //    tree_pos_ids[0] = 0 → root sits at position context_len.
    position_ids = ov::Tensor(ov::element::i64, {1, num_candidates});
    int64_t* pos_ptr = position_ids.data<int64_t>();
    for (size_t i = 0; i < num_candidates; ++i) {
        pos_ptr[i] = static_cast<int64_t>(context_len) + static_cast<int64_t>(tree_pos_ids[i]);
    }

    // 3. attention_mask: all-ones.
    attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
    std::fill_n(attention_mask.data<int64_t>(), attn_len, 1);

    // 4. eagle_tree_mask: shape {1, 1, num_candidates, attn_len}.
    eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, num_candidates, attn_len});
    float* mask_ptr = eagle_tree_mask.data<float>();
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
}

void Eagle3InferWrapperBase::build_model_inputs(size_t input_token_count,
                                                ov::Tensor& input_ids,
                                                ov::Tensor& attention_mask,
                                                ov::Tensor& position_ids,
                                                ov::Tensor& eagle_tree_mask,
                                                InferencePhase phase,
                                                size_t past_accepted_token_count) {
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");

    switch (phase) {
    case InferencePhase::TARGET_PREFILL:
    case InferencePhase::DRAFT_INITIAL:
        build_inputs_for_prefill(input_token_count, input_ids, attention_mask, position_ids, eagle_tree_mask);
        break;
    case InferencePhase::DRAFT_ITERATION:
        build_inputs_for_draft_iteration(past_accepted_token_count,
                                         input_ids,
                                         attention_mask,
                                         position_ids,
                                         eagle_tree_mask);
        break;
    case InferencePhase::TARGET_VALIDATION:
        build_inputs_for_target_validation(input_ids, attention_mask, position_ids, eagle_tree_mask);
        break;
    }
}

std::vector<int64_t> Eagle3InferWrapperBase::sample_tokens(const ov::Tensor& logits,
                                                           size_t input_token_count,
                                                           size_t sample_count,
                                                           size_t num_tokens_to_validate) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape: ", shape);
    OPENVINO_ASSERT(sample_count > 0 && sample_count <= shape[1],
                    "Invalid sample_count: ",
                    sample_count,
                    ", logits seq_len: ",
                    shape[1]);
    OPENVINO_ASSERT(input_token_count > 0, "Invalid input_token_count");

    const bool is_validation_mode = num_tokens_to_validate > 0;

    auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");


    const size_t logits_seq_len = shape[1];
    const size_t vocab_size = shape[2];

    // Snapshot running sequences BEFORE sampling.  In draft mode the sampler may fork a
    // sequence into multiple children; the original Sequence objects remain alive (owned by the
    // SequenceGroup) and their generated_ids are only appended to, never replaced, so the
    // snapshot is still valid after m_sampler.sample().  In validation mode the sampler calls
    // validate_tree_candidates which rewrites generated_ids in-place on the single running
    // sequence — the Sequence pointer itself is stable, so running_sequences[0] is safe to
    // dereference after the call.
    auto running_sequences = sequence_group->get_running_sequences();
    const size_t num_sequences = running_sequences.size();
    OPENVINO_ASSERT(num_sequences > 0, "No running sequences");

    // In validation mode the sampler's validate_tree_candidates indexes logits from the end
    // using (seq_len - token_idx - 1), so it needs the full [1, num_candidates, vocab] tensor.
    // In non-validation (draft) mode, slice to the last num_sequences positions so each
    // running beam gets the logit from its own output position.
    ov::Tensor sliced_logits = logits;
    if (!is_validation_mode && num_sequences < logits_seq_len) {
        auto [start_coord, end_coord] =
            ov::genai::utils::make_roi(shape, 1, logits_seq_len - num_sequences, logits_seq_len);
        sliced_logits = ov::Tensor(logits, start_coord, end_coord);
    }

    // Configure sequence group for sampling
    sequence_group->schedule_tokens(input_token_count);
    sequence_group->set_output_seq_len(sample_count);
    sequence_group->set_num_validated_tokens(num_tokens_to_validate);

    // In validation mode, record the length of generated_ids just before the sampler runs.
    // At this point generated_ids = [...history..., root, n_1, ..., n_N] where N =
    // num_tokens_to_validate.  After validate_tree_candidates the sequence is rewritten to
    // [...history..., root, acc_1, ..., acc_k, bonus].  Slicing from (pre_sample_len - N)
    // (i.e. the position right after the history, at the root) gives exactly
    // [root, acc_1, ..., acc_k, bonus]; we then drop the root at index 0 to get the result.
    size_t root_index_in_generated = 0;
    if (is_validation_mode) {
        const auto& gen = running_sequences[0]->get_generated_ids();
        OPENVINO_ASSERT(gen.size() > num_tokens_to_validate,
                        "generated_ids too short before validation: ",
                        gen.size(),
                        " <= num_tokens_to_validate: ",
                        num_tokens_to_validate);
        // root sits at generated_ids[gen.size() - num_tokens_to_validate - 1]
        root_index_in_generated = gen.size() - num_tokens_to_validate - 1;
    }

    // Execute sampling
    m_sampler.sample({sequence_group}, sliced_logits, is_validation_mode);
    sequence_group->finish_iteration();

    std::vector<int64_t> result_tokens;

    if (!is_validation_mode) {
        // Non-validation mode: collect last token from each sequence
        for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
            auto seq = running_sequences[seq_idx];
            OPENVINO_ASSERT(seq, "Invalid sequence at index ", seq_idx);

            const auto& generated_ids = seq->get_generated_ids();
            OPENVINO_ASSERT(!generated_ids.empty(), "Sequence ", seq_idx, " has no generated tokens");

            // Add the last token from this sequence
            result_tokens.push_back(generated_ids.back());
        }

        record_generated_tokens(num_sequences);

        return result_tokens;
    } else {
        // Validation mode: the sampler's validate_tree_candidates has already rewritten the
        // sequence to [...history..., root, acc_1, ..., acc_k, bonus_token].
        // root_index_in_generated points to the root position recorded before sampling.
        // Return [acc_1, ..., acc_k, bonus_token], i.e. everything after the root.
        OPENVINO_ASSERT(num_sequences == 1, "Validation mode expects exactly one running sequence");
        auto seq = running_sequences[0];
        OPENVINO_ASSERT(seq, "Invalid sequence in validation mode");

        const auto& generated_ids = seq->get_generated_ids();
        // generated_ids[root_index_in_generated]     = root  (stays in sequence as context)
        // generated_ids[root_index_in_generated + 1..] = acc_1, ..., acc_k, bonus_token
        const size_t result_start = root_index_in_generated + 1;
        OPENVINO_ASSERT(generated_ids.size() > result_start,
                        "Validation result sequence too short: size=",
                        generated_ids.size(),
                        ", expected > root_index+1=",
                        result_start);
        result_tokens.assign(generated_ids.begin() + result_start, generated_ids.end());

        record_generated_tokens(result_tokens.size());

        return result_tokens;
    }
}

ov::Tensor Eagle3InferWrapperBase::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor Eagle3InferWrapperBase::get_hidden_features(size_t actual_seq_len) const {
    auto hidden_state = m_request.get_tensor("last_hidden_state");
    const auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid hidden state shape: ", shape);

    const size_t output_seq_len = shape[1];
    const size_t hidden_size = shape[2];
    // If the caller did not supply actual_seq_len, fall back to querying the input_ids tensor.
    // Callers that already hold input_ids should pass its shape[1] to avoid the extra query.
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

uint64_t Eagle3InferWrapperBase::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    return duration_us;
}

void Eagle3InferWrapperBase::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
}

void Eagle3InferWrapperBase::record_generated_tokens(size_t actual_generated_count) {
    m_raw_perf_metrics.m_batch_sizes.emplace_back(actual_generated_count);
}

Eagle3TargetWrapper::Eagle3TargetWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {}

void Eagle3TargetWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len], got ", shape);

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = shape[1];
    OPENVINO_ASSERT(seq_len > 0, "Empty prompt");

    TokenIds prompt_ids(ids_data, ids_data + seq_len);
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config, 0);

    OPENVINO_ASSERT(m_sequence_group->num_total_seqs() == 1,
                    "Expected single sequence after initialization, got ",
                    m_sequence_group->num_total_seqs());
}

InferenceOutput Eagle3TargetWrapper::infer(const ov::Tensor& input_ids,
                                           const ov::Tensor& attention_mask,
                                           const ov::Tensor& position_ids,
                                           const ov::Tensor& eagle_tree_mask) {
    const size_t prompt_len = input_ids.get_shape()[1];

    if (m_device == "NPU") {
        OPENVINO_ASSERT(prompt_len <= m_max_prompt_len,
                        "NPU prompt length ",
                        prompt_len,
                        " exceeds max ",
                        m_max_prompt_len);
    }

    // Set model inputs
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    m_request.set_tensor("eagle_tree_mask", eagle_tree_mask);

    // Execute inference
    uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    // Collect outputs
    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features(prompt_len);

    return output;
}

InferResult Eagle3TargetWrapper::forward(const InferContext& ctx) {
    // 1. Prepare inputs from sequence state
    ov::Tensor input_ids, attention_mask, position_ids, eagle_tree_mask;
    const InferencePhase phase =
        (ctx.num_tokens_to_validate > 0) ? InferencePhase::TARGET_VALIDATION : InferencePhase::TARGET_PREFILL;
    build_model_inputs(ctx.input_token_count, input_ids, attention_mask, position_ids, eagle_tree_mask, phase);

    // 2. Infer
    auto output = infer(input_ids, attention_mask, position_ids, eagle_tree_mask);

    // 3. Sample
    // During TARGET_PREFILL the sampler must produce exactly one greedy token.
    // The sequence group's config still has tree_depth > 0 (needed for the subsequent
    // speculative loop), which would make the sampler fork sequences via TreeSearcher.
    // Temporarily zero tree_depth so the sampler takes the greedy path; restore afterwards.
    // This is an internal detail of the prefill sampling step — callers are not affected.
    std::vector<int64_t> sampled;
    if (phase == InferencePhase::TARGET_PREFILL) {
        const auto saved_config = get_sequence_group()->get_sampling_parameters();
        auto prefill_config = saved_config;
        prefill_config.eagle_tree_params.tree_depth = 0;
        get_sequence_group()->set_sampling_parameters(prefill_config);

        sampled = sample_tokens(output.logits, ctx.input_token_count, ctx.sample_count, 0);

        // Restore the original tree-search config for the speculative loop.
        get_sequence_group()->set_sampling_parameters(saved_config);
    } else {
        sampled = sample_tokens(output.logits, ctx.input_token_count, ctx.sample_count, ctx.num_tokens_to_validate);
    }

    // 4. Store hidden states to sequence for draft model to use
    get_current_sequence()->update_hidden_state(output.hidden_features);

    return InferResult{std::move(output), std::move(sampled)};
}

Eagle3DraftWrapper::Eagle3DraftWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {}

void Eagle3DraftWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len], got ", shape);

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t total_len = shape[1];
    OPENVINO_ASSERT(total_len >= 2, "Draft model requires at least 2 tokens");

    // Draft model uses tokens[1:] (Eagle3 specific behavior)
    TokenIds draft_prompt_ids(ids_data + 1, ids_data + total_len);
    m_sequence_group = std::make_shared<SequenceGroup>(1, draft_prompt_ids, config, 0);

    OPENVINO_ASSERT(m_sequence_group->num_total_seqs() == 1,
                    "Expected single sequence after initialization, got ",
                    m_sequence_group->num_total_seqs());
}

InferenceOutput Eagle3DraftWrapper::infer(const ov::Tensor& input_ids,
                                          const ov::Tensor& attention_mask,
                                          const ov::Tensor& position_ids,
                                          const ov::Tensor& eagle_tree_mask,
                                          const ov::Tensor& hidden_states) {
    const size_t input_token_count = input_ids.get_shape()[1];

    // Set standard inputs
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    m_request.set_tensor("eagle_tree_mask", eagle_tree_mask);

    // Set hidden states (either from target model or internal)
    OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "hidden_states must be provided");
    auto shape = hidden_states.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Invalid hidden states shape: ", shape);
    m_request.set_tensor("hidden_states", hidden_states);

    // Execute inference
    uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    // Collect outputs
    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features(input_token_count);

    return output;
}

InferResult Eagle3DraftWrapper::forward(const InferContext& ctx) {
    // 1. Prepare inputs
    ov::Tensor input_ids, attention_mask, position_ids, eagle_tree_mask;
    const InferencePhase phase =
        ctx.use_target_hidden ? InferencePhase::DRAFT_INITIAL : InferencePhase::DRAFT_ITERATION;
    build_model_inputs(ctx.input_token_count,
                       input_ids,
                       attention_mask,
                       position_ids,
                       eagle_tree_mask,
                       phase,
                       ctx.past_accepted_token_count);

    // 2. Get hidden states from appropriate source
    ov::Tensor hidden_states;
    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: use target model's hidden state for the single root position.
        OPENVINO_ASSERT(ctx.target_sequence, "target_sequence required when use_target_hidden=true");
        hidden_states = ctx.target_sequence->get_hidden_state();
        OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "Source sequence contains invalid hidden state");
    } else {
        // DRAFT_ITERATION: Concatenate the full stored hidden-state path of every sequence.
        // After the DRAFT_INITIAL pass each sequence holds {1, 1, hidden_size} (the root).
        // After each subsequent iteration the stored tensor grows by one token per sequence,
        // so at iteration k each sequence holds {1, k, hidden_size}.
        // Concatenating all sequences yields {1, num_seqs * branch_len, hidden_size}, which
        // matches the flat input_ids layout produced by build_inputs_for_draft_iteration.
        auto running_sequences = m_sequence_group->get_running_sequences();
        const size_t num_sequences = running_sequences.size();
        OPENVINO_ASSERT(num_sequences > 0, "No running sequences");

        std::vector<ov::Tensor> seq_hidden_states;
        seq_hidden_states.reserve(num_sequences);
        for (size_t i = 0; i < num_sequences; ++i) {
            const auto seq_hidden = running_sequences[i]->get_hidden_state();
            OPENVINO_ASSERT(seq_hidden && seq_hidden.get_size() > 0, "Sequence ", i, " contains invalid hidden state");
            const auto& h_shape = seq_hidden.get_shape();
            OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1,
                            "Expected hidden state shape [1, branch_len, hidden_size], got: ",
                            h_shape);
            seq_hidden_states.push_back(seq_hidden);
        }

        hidden_states = concatenate_hidden_states(seq_hidden_states);
    }

    // 3. Infer
    auto output = infer(input_ids, attention_mask, position_ids, eagle_tree_mask, hidden_states);

    // 4. Store hidden states BEFORE sampling so that each sequence holds its correct path
    //    hidden states regardless of how the sampler may later fork or prune paths.
    auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t num_sequences = running_sequences.size();

    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: The output has a single new position (the root token).
        // All beam sequences start from the same root, so they all receive the same hidden state.
        const auto next_hidden =
            slice_hidden_state_at_position(output.hidden_features, output.hidden_features.get_shape()[1] - 1);
        for (size_t i = 0; i < num_sequences; ++i) {
            running_sequences[i]->update_hidden_state(next_hidden);
        }
    } else {
        // DRAFT_ITERATION: output.hidden_features has shape {1, total_tokens, hidden_size}
        // where total_tokens = num_seqs * branch_len (flat layout, same order as input_ids).
        // For each sequence append the hidden state of its LAST branch token to the
        // sequence's accumulated hidden state, growing it from {1, k, H} to {1, k+1, H}.
        // This is done before sample_tokens so that if the sampler forks sequences the
        // per-sequence hidden state is already correctly assigned.
        const auto& hidden_shape = output.hidden_features.get_shape();
        OPENVINO_ASSERT(hidden_shape.size() == 3 && hidden_shape[0] == 1,
                        "Invalid hidden features shape: ",
                        hidden_shape);

        // Recover branch_len from the stored hidden state of the first sequence (= k tokens).
        // After this iteration it will become k+1.
        const size_t branch_len = running_sequences[0]->get_hidden_state().get_shape()[1];
        const size_t total_hidden_tokens = num_sequences * branch_len;
        // Hardware may pad the sequence dimension at the front; valid tokens are at the tail.
        OPENVINO_ASSERT(hidden_shape[1] >= total_hidden_tokens,
                        "hidden_features seq_len (",
                        hidden_shape[1],
                        ") < num_seqs (",
                        num_sequences,
                        ") * branch_len (",
                        branch_len,
                        ")");
        const size_t hidden_pad_offset = hidden_shape[1] - total_hidden_tokens;

        for (size_t i = 0; i < num_sequences; ++i) {
            // The last branch token for sequence i sits at hidden_pad_offset + (i+1)*branch_len - 1.
            const size_t last_tok_pos = hidden_pad_offset + (i + 1) * branch_len - 1;
            const auto new_tok_hidden = slice_hidden_state_at_position(output.hidden_features, last_tok_pos);

            // Append the new token's hidden state to the existing path: {1,k,H} → {1,k+1,H}.
            // TODO(performance): concatenate_hidden_states allocates a new {1, k+1, H} tensor on
            // every draft iteration.  Consider pre-allocating a {1, max_draft_iters, H} buffer
            // per sequence and writing new tokens in-place to avoid repeated heap allocation.
            const auto existing_hidden = running_sequences[i]->get_hidden_state();
            const auto updated_hidden = concatenate_hidden_states({existing_hidden, new_tok_hidden});
            running_sequences[i]->update_hidden_state(updated_hidden);
        }
    }

    // 5. Sample
    // For DRAFT_ITERATION the model produced logits with shape {1, total_tokens, vocab_size}
    // where total_tokens = num_seqs * branch_len (flat layout).  The sampler expects one logit
    // row per running sequence, i.e. shape {1, num_seqs, vocab_size}.  Gather the
    // last-branch-token logit of each sequence (positions branch_len-1, 2*branch_len-1, …)
    // into a compact tensor before calling sample_tokens.
    //
    // For DRAFT_INITIAL the logits shape is {1, prompt_len, vocab_size} with a single sequence;
    // sample_tokens' existing tail-slice handles that correctly without any pre-processing.
    //
    // TODO(performance): logits_for_sampling is allocated fresh on every DRAFT_ITERATION call
    // with shape {1, num_seqs, vocab_size}.  Since num_seqs and vocab_size are fixed for the
    // lifetime of a speculative window, consider pre-allocating this tensor and reusing it.
    ov::Tensor logits_for_sampling = output.logits;
    if (!ctx.use_target_hidden) {
        // DRAFT_ITERATION path: gather last-position logits for each sequence.
        const auto& lshape = output.logits.get_shape();
        // branch_len was already computed and used when storing hidden states above;
        // recover it from the sequence hidden state which was just grown to branch_len+1.
        // The stored hidden state now has shape {1, branch_len+1, H}, so branch_len = shape[1]-1.
        const size_t new_branch_len = running_sequences[0]->get_hidden_state().get_shape()[1];
        const size_t prev_branch_len = new_branch_len - 1;  // branch_len used during this infer pass
        OPENVINO_ASSERT(prev_branch_len > 0, "branch_len must be positive in DRAFT_ITERATION");
        const size_t total_tokens = num_sequences * prev_branch_len;
        // Hardware may pad the sequence dimension to a multiple of 8 at the front.
        // The valid tokens always occupy the tail: [lshape[1] - total_tokens, lshape[1]).
        OPENVINO_ASSERT(lshape[1] >= total_tokens,
                        "Logits seq_len (",
                        lshape[1],
                        ") < num_seqs (",
                        num_sequences,
                        ") * branch_len (",
                        prev_branch_len,
                        ")");
        const size_t pad_offset = lshape[1] - total_tokens;

        const size_t vocab_size = lshape[2];
        logits_for_sampling = ov::Tensor(ov::element::f32, {1, num_sequences, vocab_size});
        float* dst = logits_for_sampling.data<float>();
        const float* src = output.logits.data<const float>();

        for (size_t s = 0; s < num_sequences; ++s) {
            // Last token of sequence s sits at flat position pad_offset + (s+1)*prev_branch_len - 1.
            const size_t flat_pos = pad_offset + (s + 1) * prev_branch_len - 1;
            std::copy_n(src + flat_pos * vocab_size, vocab_size, dst + s * vocab_size);
        }
    }

    auto sampled = sample_tokens(logits_for_sampling, ctx.input_token_count, 1);

    return InferResult{std::move(output), std::move(sampled)};
}

StatefulEagle3LLMPipeline::StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config) {
    // eagle_tree_params are read from draft_model_desc.generation_config
    // m_generation_config is initialised from target_model_desc (eos_token_id, etc.)
    m_generation_config.eagle_tree_params = draft_model_desc.generation_config.eagle_tree_params;

    // Apply compile-time defaults when no eagle_tree_params were provided by the user or
    // the draft model JSON (tree_depth == 0 is the zero-initialised struct default).
    ensure_eagle_tree_params_is_set(m_generation_config);

    OPENVINO_ASSERT(m_generation_config.is_tree_search(), "Eagle3 pipeline requires eagle_tree_params.tree_depth > 0.");

    // Extract hidden_layers_list from draft model properties — used only during model
    // construction to wire the target model's hidden-state extraction; no need to retain it.
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

    // Model preparation
    utils::eagle3::share_vocabulary(target_model, draft_model);

    auto d2t_mapping = utils::eagle3::extract_d2t_mapping_table(draft_model);
    OPENVINO_ASSERT(d2t_mapping && d2t_mapping->get_element_type() == ov::element::i64, "Invalid d2t mapping tensor");

    utils::eagle3::apply_eagle3_attention_mask_transform(draft_model);
    utils::eagle3::apply_eagle3_attention_mask_transform(target_model);

    utils::eagle3::transform_hidden_state(target_model, hidden_layers_to_abstract);
    utils::eagle3::move_fc_from_draft_to_main(draft_model, target_model);
    utils::eagle3::transform_hidden_state(draft_model, {-1});

    // target_validation_window: number of candidate tokens the target model validates in one
    // step — all N tree nodes (num_speculative_tokens + 1), including the accepted root token.
    const size_t target_validation_window = m_generation_config.eagle_tree_params.num_speculative_tokens;

    // draft_validation_window: maximum number of tokens the draft model processes in a single
    // DRAFT_ITERATION pass.  At each iteration all running sequences each contribute one token
    // so the worst case is at the last pass: (tree_depth - 1) * branching_factor
    m_draft_iterations = m_generation_config.eagle_tree_params.tree_depth;
    const size_t draft_validation_window =
        (m_draft_iterations - 1) * m_generation_config.eagle_tree_params.branching_factor;

    // Configure and create draft model
    auto draft_desc = draft_model_desc;
    if (draft_desc.device == "NPU") {
        draft_desc.properties["NPUW_EAGLE"] = "TRUE";
        draft_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = draft_validation_window;
        draft_desc.properties["NPUW_ONLINE_PIPELINE"] = "NONE";
        draft_desc.properties["NPUW_DEVICES"] = "CPU";
    }
    m_draft = std::make_unique<Eagle3DraftWrapper>(draft_desc);

    m_draft->set_draft_target_mapping(d2t_mapping);

    // Configure and create target model
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
    // Call base class implementation to handle common defaults
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);

    // Apply Eagle3 defaults: if the user did not provide eagle_tree_params
    ensure_eagle_tree_params_is_set(config);

    return config;
}

EncodedResults StatefulEagle3LLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate(EncodedInputs)");
    generate_timer.start();

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    // Extract input tensors
    ov::Tensor input_ids, attention_mask;
    if (auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    OPENVINO_ASSERT(input_ids.get_shape()[0] == 1, "Only batch size 1 supported");
    m_prompt_length = input_ids.get_shape()[1];

    // Initialize position IDs
    ov::Tensor position_ids{ov::element::i64, input_ids.get_shape()};
    utils::initialize_position_ids(position_ids, attention_mask);

    // Reset model states
    m_target->reset_state();
    m_draft->reset_state();

    // Prepare sampling config with extended max_new_tokens to prevent premature termination
    // during draft generation. Actual length control is in the generation loop.
    auto sampling_config = config;
    // Reserve m_draft_iterations (DRAFT_ITERATION passes) + 1 (DRAFT_INITIAL) + 1 (bonus token)
    // extra slots to prevent premature termination during draft generation.
    sampling_config.max_new_tokens = config.max_new_tokens + m_draft_iterations + 2;

    m_draft->initialize_sequence(input_ids, sampling_config);
    m_target->initialize_sequence(input_ids, sampling_config);

    // Phase 1: Initial Prompt Processing (Prefill)
    InferContext prefill_ctx;
    prefill_ctx.input_token_count = m_prompt_length;
    auto prefill_result = m_target->forward(prefill_ctx);
    OPENVINO_ASSERT(prefill_result.sampled_tokens.size() == 1, "Expected single token from prefill");
    const auto initial_token = prefill_result.sampled_tokens[0];

    // Append initial token to draft model.
    m_draft->append_tokens({initial_token});

    auto streaming_status = stream_generated_tokens(streamer_ptr, {initial_token});

    // Phase 2: Speculative Decoding Loop
    size_t generated_tokens = 1;
    size_t total_draft_accepted = 0;
    size_t total_draft_generated = 0;
    bool eos_reached = false;

    size_t input_token_count = m_draft->get_sequence_length();

    while (!eos_reached && generated_tokens < config.max_new_tokens &&
           m_target->get_sequence_length() < m_prompt_length + config.max_new_tokens &&
           streaming_status == ov::genai::StreamingStatus::RUNNING) {
        auto result = run_speculative_iteration(input_token_count, static_cast<int64_t>(config.eos_token_id));

        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        // Update statistics
        total_draft_generated += m_draft_iterations;
        total_draft_accepted += result.accepted_tokens_count;
        generated_tokens += result.validated_tokens.size();
        eos_reached = result.eos_reached;

        // Prepare for next iteration (hidden states are stored in sequence)
        input_token_count = result.next_window_size;
    }

    // Phase 3: Finalization
    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr)
        streamer_ptr->end();

    // Collect results
    EncodedResults results;
    results.tokens = {m_target->get_generated_tokens()};
    results.scores = {0.0f};

    generate_timer.end();

    // Update performance metrics
    m_sd_perf_metrics.num_input_tokens = m_prompt_length;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    // Reset evaluated flags before updating raw_metrics to ensure statistics are recalculated
    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;

    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_target->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft->get_raw_perf_metrics();

    if (total_draft_generated > 0) {
        float acceptance_rate = static_cast<float>(total_draft_accepted) / total_draft_generated * 100.0f;
        m_sd_metrics.update_acceptance_rate(0, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens);
    }

    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    // Reset timer
    generate_timer.clear();

    return results;
}

StatefulEagle3LLMPipeline::SpeculativeResult StatefulEagle3LLMPipeline::run_speculative_iteration(
    size_t input_token_count,
    int64_t eos_token_id,
    size_t current_generated_tokens,
    size_t max_new_tokens) {
    SpeculativeResult result;

    OPENVINO_ASSERT(m_target->get_sequence_group() && m_draft->get_sequence_group(),
                    "Eagle3 speculative iteration requires initialized sequence groups");

    auto target_hidden_states = m_target->get_current_sequence()->get_hidden_state();
    OPENVINO_ASSERT(target_hidden_states && target_hidden_states.get_size() > 0,
                    "Target model contains invalid hidden state for speculation");

    // Record pre-draft sequence length for rollback on mismatch.
    const size_t pre_draft_token_len = m_draft->get_sequence_length();

    // past_accepted_token_count: tokens already accepted by the target before this speculative
    // window (including the initial token from prefill and all previously accepted tokens).
    // DRAFT_ITERATION uses this offset to locate the branch tokens inside the draft's
    // accumulated generated_ids, since the draft mirrors the target's accepted history.
    const size_t past_accepted_token_count = m_target->get_current_sequence()->get_generated_ids().size();

    // Clear stale EagleMetaData left over from the previous iteration so that
    // select_top_k starts from a clean slate.
    m_draft->get_current_sequence()->set_eagle_metadata({});

    // Step 1: Generate first draft token using target hidden states (DRAFT_INITIAL).
    InferContext first_ctx;
    first_ctx.input_token_count = input_token_count;
    first_ctx.use_target_hidden = true;
    first_ctx.target_sequence = m_target->get_current_sequence();
    first_ctx.past_accepted_token_count = past_accepted_token_count;
    auto first_result = m_draft->forward(first_ctx);

    // first_result.sampled_tokens contains one token per running sequence.
    const auto& first_draft_tokens = first_result.sampled_tokens;
    OPENVINO_ASSERT(!first_draft_tokens.empty(), "Expected at least one token from first draft");

    // Step 2: Generate additional draft tokens using internal hidden states (DRAFT_ITERATION).
    for (size_t i = 1; i < m_draft_iterations; ++i) {
        InferContext more_ctx;
        more_ctx.input_token_count = 1;  // updated by build_inputs_for_draft_iteration based on num sequences
        more_ctx.use_target_hidden = false;
        more_ctx.past_accepted_token_count = past_accepted_token_count;
        auto more_result = m_draft->forward(more_ctx);

        const auto& draft_tokens = more_result.sampled_tokens;
        OPENVINO_ASSERT(!draft_tokens.empty(), "Expected tokens from draft iteration ", i);
    }

    // Step 3: Validate draft tokens with target model.
    // In tree-search mode the draft model generates a tree of N+1 candidates:
    //   draft generated_ids tail = [root, node_1, …, node_N]
    //   EagleMetaData.tree_mask  = (N+1)×(N+1), index 0 = root.
    // The root token is the LAST token sampled by the target in the previous iteration.
    // It has NOT been fed back to the target KV cache yet, so the target must process
    // all N+1 candidates (root + N tree nodes) during validation.
    const auto& draft_gen_ids = m_draft->get_current_sequence()->get_generated_ids();
    const auto& draft_metadata = m_draft->get_current_sequence()->get_eagle_metadata();
    const size_t num_candidates = draft_metadata.tree_mask.size();  // N+1 (root + N nodes)
    OPENVINO_ASSERT(num_candidates >= 2,
                    "Expected at least 2 candidates (root + at least 1 tree node), got: ",
                    num_candidates);
    const size_t num_tree_nodes = num_candidates - 1;  // N non-root tokens used for validation

    // Sync EagleMetaData from DRAFT to TARGET so that build_inputs_for_target_validation can read it.
    m_target->get_current_sequence()->set_eagle_metadata(draft_metadata);

    // Sync all N+1 candidate tokens to the target sequence.
    // The root is already present in target generated_ids (appended after prefill/previous iteration).
    // Append only the N non-root tree-node tokens.
    OPENVINO_ASSERT(draft_gen_ids.size() >= num_candidates,
                    "draft generated_ids too short: ",
                    draft_gen_ids.size(),
                    " < num_candidates: ",
                    num_candidates);
    // draft_gen_ids tail layout: [..., root, node_1, ..., node_N]
    // The root is already in the target sequence; append only node_1..node_N.
    const size_t draft_tail_offset = draft_gen_ids.size() - num_candidates;
    for (size_t i = 1; i < num_candidates; ++i) {
        m_target->get_current_sequence()->append_token(draft_gen_ids[draft_tail_offset + i], 0.0f);
    }

    // input_token_count = num_candidates: the target processes all N+1 tokens (root + N nodes).
    // num_tokens_to_validate = num_tree_nodes: the sampler checks the N non-root draft tokens.
    // sample_count = num_candidates: one output position per input token.
    InferContext val_ctx;
    val_ctx.input_token_count = num_candidates;
    val_ctx.sample_count = num_candidates;
    val_ctx.num_tokens_to_validate = num_tree_nodes;
    auto val_result = m_target->forward(val_ctx);

    // Sampler validates draft tokens and returns accepted + new sampled token.
    auto validated_tokens = val_result.sampled_tokens;

    // Result: [accepted_draft_tokens..., new_sampled_token]
    const size_t total_accepted_tokens = validated_tokens.size();
    const size_t accepted_count = total_accepted_tokens - 1;
    const int64_t target_predicted_token = validated_tokens.back();
    OPENVINO_ASSERT(total_accepted_tokens <= num_candidates,
                    "Sampler returned more tokens (",
                    total_accepted_tokens,
                    ") than candidates (",
                    num_candidates,
                    ")");
    // Target KV cache now holds num_candidates new tokens; keep only total_accepted_tokens.
    const size_t tokens_to_remove = num_candidates - total_accepted_tokens;

    // Step 4: Synchronize sequences and KV cache
    // Target model's sequence is already updated by Sampler
    // Sync draft model's sequence
    m_draft->truncate_sequence(pre_draft_token_len);
    m_draft->append_tokens(validated_tokens);

    // KV cache management after tree-search validation:
    //
    // TARGET (NPU): The accepted candidates form a non-contiguous path through the tree, so a
    //   simple tail-trim is incorrect.  Instead, write the acceptance mask into the
    //   "npuw_eagle3_sampling_result" VariableState.  NPUW reads this before the next infer
    //   call and retains only the KV entries whose mask bit is 1.
    //   validated_indices contains the accepted candidate path indices set by validate_tree_candidates
    //   (e.g. [0, 2, 5] = root accepted + candidates 2 and 5 accepted; the rest are rejected).
    //
    // DRAFT (NPU): no explicit KV cache management is needed.  In the next iteration, all
    //   accepted tokens are re-submitted as input_ids with their correct position_ids.  NPUW
    //   uses the position_ids to overwrite the relevant KV slots, so stale entries are
    //   implicitly replaced without any explicit trim or mask.
    //
    // NON-NPU: the KV cache is a plain tensor; trim the tail to remove the rejected positions.
    if (m_target->device() == "NPU") {
        const auto& target_validated_indices = m_target->get_current_sequence()->get_eagle_metadata().validated_indices;
        m_target->set_npu_sampling_result(num_candidates, target_validated_indices);
    } else if (tokens_to_remove > 0) {
        m_target->trim_kv_cache(tokens_to_remove);
        m_draft->trim_kv_cache(tokens_to_remove);
    }

    // Step 5: Update hidden states for next iteration.
    //
    // current_hidden has shape {1, num_candidates, H}, where the seq dimension is a flat
    // enumeration of all N+1 tree candidates submitted to the target model (root at index 0,
    // then tree nodes 1..N in tree order).
    //
    // validated_indices (= validate_path from validate_tree_candidates) contains the
    // tree-position indices of the accepted path, e.g. [0, 2, 5] means the root (index 0),
    // node at tree position 2, and node at tree position 5 were accepted.  These indices are
    // used directly as row selectors into current_hidden — they are NOT contiguous, so a
    // simple head/tail slice is wrong.
    //
    // Gather the hidden-state rows corresponding to the accepted positions to form
    // {1, total_accepted_tokens, H} for the next draft-model pass.
    auto current_hidden = val_result.output.hidden_features;
    OPENVINO_ASSERT(current_hidden && current_hidden.get_size() > 0, "Missing hidden features");

    const auto h_shape = current_hidden.get_shape();
    OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1, "Invalid hidden state shape: ", h_shape);

    const auto& target_validated_indices = m_target->get_current_sequence()->get_eagle_metadata().validated_indices;
    OPENVINO_ASSERT(target_validated_indices.size() == total_accepted_tokens,
                    "validated_indices size (",
                    target_validated_indices.size(),
                    ") != total_accepted_tokens (",
                    total_accepted_tokens,
                    ")");

    const size_t hidden_size = h_shape[2];
    ov::Tensor next_hidden(ov::element::f32, {1, total_accepted_tokens, hidden_size});
    const float* src = current_hidden.data<const float>();
    float* dst = next_hidden.data<float>();
    for (size_t i = 0; i < total_accepted_tokens; ++i) {
        const size_t row = static_cast<size_t>(target_validated_indices[i]);
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

    result.accepted_tokens_count = total_accepted_tokens - 1;
    result.next_window_size = total_accepted_tokens;
    result.validated_tokens = std::move(validated_tokens);
    result.eos_reached = (target_predicted_token == eos_token_id);

    return result;
}

void StatefulEagle3LLMPipeline::finish_chat() {
    // Eagle3 uses base class implementation directly (no model state reset needed)
    StatefulSpeculativePipelineBase::finish_chat();
}

SpeculativeDecodingMetrics StatefulEagle3LLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

}  // namespace ov::genai
