// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

#include "continuous_batching/timer.hpp"
#include "eagle3_utils.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "speculative_decoding/eagle3_debug_utils.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "utils.hpp"

namespace ov::genai {
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
}  // namespace ov::genai

namespace {

ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

}  // anonymous namespace

namespace ov::genai {

Eagle3InferWrapperBase::Eagle3InferWrapperBase(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(model_desc.tokenizer),
      m_sampler(model_desc.tokenizer) {
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);

    if (m_device == "NPU") {
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_axes_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_request = compiled.create_infer_request();

        eagle3::log_debug(eagle3::PipelineStep::INIT,
                          "NPU compiled: max_prompt=" + std::to_string(m_max_prompt_len),
                          m_verbose);
    } else {
        m_request =
            utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
        eagle3::log_debug(eagle3::PipelineStep::INIT, m_device + " model compiled successfully", m_verbose);
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

    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Appended " + std::to_string(tokens.size()) + " tokens: " + eagle3::format_tokens(tokens) +
                          ", new seq_len=" + std::to_string(get_sequence_length()),
                      m_verbose);
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

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Truncated sequence: " + std::to_string(current_len) + " -> " + std::to_string(size) +
                              " (removed " + std::to_string(tokens_to_remove) + " tokens)",
                          m_verbose);
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
        utils::KVCacheState state;
        state.num_tokens_to_trim = tokens_to_remove;
        state.seq_length_axis = m_kv_axes_pos.seq_len;
        state.reset_mem_state = false;
        utils::trim_kv_cache(m_request, state, {});
    }

    eagle3::log_debug(eagle3::PipelineStep::KV_CACHE,
                      "KV cache trimmed: " + std::to_string(current_len) + " -> " +
                          std::to_string(current_len - tokens_to_remove) + " (removed " +
                          std::to_string(tokens_to_remove) + ")",
                      m_verbose);
}

void Eagle3InferWrapperBase::set_npu_sampling_result(size_t num_candidates,
                                                     const std::vector<int64_t>& accepted_indices) {
    // Build acceptance mask: mask[i] = true iff candidate i is in accepted_indices.
    std::vector<bool> mask(num_candidates, false);
    for (const int64_t idx : accepted_indices) {
        OPENVINO_ASSERT(idx >= 0 && static_cast<size_t>(idx) < num_candidates,
                        "accepted_index ",
                        idx,
                        " is out of range [0, ",
                        num_candidates,
                        ")");
        mask[static_cast<size_t>(idx)] = true;
    }
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
        for (size_t i = 0; i < num_candidates; ++i) {
            data[2 + i] = mask[i] ? 1 : 0;
        }
        state.set_state(tensor);

        eagle3::log_debug(eagle3::PipelineStep::KV_CACHE,
                          "NPU sampling result set: num_candidates=" + std::to_string(num_candidates) +
                              ", num_accepted=" + std::to_string(num_accepted),
                          m_verbose);
        return;
    }

    // VariableState not found — this is expected on non-NPUW devices; silently ignore.
    eagle3::log_debug(eagle3::PipelineStep::KV_CACHE,
                      "'" + std::string(STATE_NAME) + "' VariableState not found; skipping NPU KV cache update",
                      m_verbose);
}

void Eagle3InferWrapperBase::reset_state() {
    m_sequence_group = nullptr;

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();

    eagle3::log_debug(eagle3::PipelineStep::INIT, "State reset", m_verbose);
}

void Eagle3InferWrapperBase::release_memory() {
    m_request.get_compiled_model().release_memory();
}

void Eagle3InferWrapperBase::build_model_inputs(const size_t input_token_count,
                                                ov::Tensor& input_ids,
                                                ov::Tensor& attention_mask,
                                                ov::Tensor& position_ids,
                                                ov::Tensor& eagle_tree_mask,
                                                InferencePhase phase,
                                                size_t iteration_id,
                                                std::shared_ptr<std::vector<int64_t>> iteration_history,
                                                size_t past_generate_len) {
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");

    // DRAFT_ITERATION: all running sequences (beam paths) are concatenated into a single flat
    // sequence so that one infer call handles all paths simultaneously.
    //
    // draft generated_ids layout across speculative iterations:
    //   [hist_0, ..., hist_{past_generate_len-2}, root, branch_0, branch_1, ...]
    //
    // past_generate_len equals the number of tokens already in the target's generated_ids at
    // the start of this speculative window (i.e. tokens accepted in all prior iterations,
    // INCLUDING the root token produced by DRAFT_INITIAL).
    // The draft mirrors the target's accepted history, so the first past_generate_len entries
    // of the draft's generated_ids are those same history+root tokens, all already in the KV
    // cache.  Only the branch tokens starting at index past_generate_len are new and
    // must be submitted to the draft model.
    //
    // Example: past_generate_len=2 (hist_0=initial_token, root=second_token), branching_factor=3, prompt_len=15
    //   generated_ids: [hist_0, root, 271, 106287, 99692]
    //   input_ids    : [271, 106287, 99692]        shape {1, 3}
    //   history_len  : prompt_len + past_generate_len = 17
    //   position_ids : [17, 17, 17]
    //   attention_mask: {1, 20} all-ones           (15 prompt + 2 history+root in KV + 3 new)
    if (phase == InferencePhase::DRAFT_ITERATION) {
        const auto& prompt_ids = m_sequence_group->get_prompt_ids();
        const size_t prompt_len = prompt_ids.size();

        auto running_sequences = m_sequence_group->get_running_sequences();
        const size_t num_seqs = running_sequences.size();
        OPENVINO_ASSERT(num_seqs > 0, "No running sequences");

        // generated_ids: [...past_generate_len tokens (including root)..., branch_0, ...]
        // The first past_generate_len entries (history + root) are already in the KV cache.
        // Branch tokens start at index past_generate_len.
        const size_t full_path_len = running_sequences[0]->get_generated_ids().size();
        OPENVINO_ASSERT(full_path_len > past_generate_len,
                        "Expected generated_ids length > past_generate_len in DRAFT_ITERATION, got ",
                        full_path_len,
                        " with past_generate_len=",
                        past_generate_len);
        // branch tokens per sequence (excluding the past_generate_len history+root tokens)
        const size_t branch_len = full_path_len - past_generate_len;
        const size_t total_tokens = num_seqs * branch_len;

        // history_len: prompt + past_generate_len (history tokens + root), all already in KV cache.
        const size_t history_len = prompt_len + past_generate_len;

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
                // Skip the first past_generate_len tokens (history + root, all in KV cache).
                ids_ptr[s * branch_len + t] = gen[past_generate_len + t];
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
        float* mask = eagle_tree_mask.data<float>();
        std::fill_n(mask, total_tokens * attn_len, -std::numeric_limits<float>::infinity());

        for (size_t s = 0; s < num_seqs; ++s) {
            for (size_t t = 0; t < branch_len; ++t) {
                const size_t row = s * branch_len + t;
                const size_t row_offset = row * attn_len;

                // History region (prompt + past accepted + root): all rows can attend.
                std::fill_n(mask + row_offset, history_len, 0.0f);

                // Branch region: open own-path columns up to and including t.
                const size_t path_col_start = history_len + s * branch_len;
                for (size_t t2 = 0; t2 <= t; ++t2) {
                    mask[row_offset + path_col_start + t2] = 0.0f;
                }
            }
        }

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Built model inputs (DRAFT_ITERATION): num_seqs=" + std::to_string(num_seqs) +
                              ", past_generate_len=" + std::to_string(past_generate_len) + ", branch_len=" +
                              std::to_string(branch_len) + ", total_tokens=" + std::to_string(total_tokens) +
                              ", history_len=" + std::to_string(history_len) + ", attn_len=" + std::to_string(attn_len),
                          m_verbose);
        return;
    }

    // TARGET_VALIDATION: feed all N+1 tree candidate tokens (root + N tree nodes) to the
    // target model using the exact tree attention structure stored in EagleMetaData.
    //
    // Key insight: the root token is the LAST token output by the previous target step.
    // Although the target model produced it, its KV has NOT been written to the target
    // KV cache yet — the target processed only the prompt during prefill and the previously
    // accepted tokens during prior iterations.  Therefore the root must be re-submitted
    // as part of input_ids together with its N children.
    //
    // EagleMetaData layout (set by select_top_k, indexed over all N+1 candidates):
    //   tree_mask[i][j] == 1  → candidate j IS an ancestor of candidate i (attend allowed)
    //   tree_mask[i][j] == 0  → not an ancestor (blocked → -INF)
    //   tree_position_ids[i]  = tree depth of candidate i (root depth = 0, children depth > 0)
    //   Both are size N+1, index 0 = root.
    //
    // generated_ids layout (at this point, after syncing draft candidates to target):
    //   [hist_0, ..., hist_{K-1}, root, node_1, ..., node_N]
    //   where K = generated_ids.size() - num_candidates  (tokens already in target KV cache)
    //   and the last num_candidates entries are the current candidates.
    //
    // Attention layout for the target forward pass:
    //   context_len = prompt_len + K   (all tokens already in target KV cache)
    //   attn_len    = context_len + num_candidates
    //
    //   input_ids    : {1, N+1} = all N+1 candidates (root first, then tree nodes)
    //   position_ids : {1, N+1} = context_len + tree_position_ids[i]  for i in 0..N
    //   attention_mask: {1, attn_len} all-ones
    //   eagle_tree_mask: {1, 1, N+1, attn_len}
    //     history columns [0, context_len): 0.0  (all candidates attend to prompt + past accepted)
    //     tree columns [context_len, attn_len): tree_mask[i][j]==1 → 0.0, else -INF
    if (phase == InferencePhase::TARGET_VALIDATION) {
        auto current_sequence = get_current_sequence();
        OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

        const auto& prompt_ids = m_sequence_group->get_prompt_ids();
        const size_t prompt_len = prompt_ids.size();
        const auto& generated_ids = current_sequence->get_generated_ids();
        const auto& metadata = current_sequence->get_eagle_metadata();
        const auto& tree_mask_bin = metadata.tree_mask;         // (N+1)×(N+1)
        const auto& tree_pos_ids = metadata.tree_position_ids;  // size = N+1 (root depth=0)

        // tree_mask_bin covers all N+1 candidates (root + N tree nodes).
        OPENVINO_ASSERT(tree_mask_bin.size() >= 2,
                        "tree_mask must cover at least root + one tree node, got ",
                        tree_mask_bin.size());
        const size_t num_candidates = tree_mask_bin.size();  // N+1 (root + N nodes)
        const size_t num_tree_nodes = num_candidates - 1;    // N  (non-root nodes)

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

        // context_len = prompt + all previously accepted generated tokens (already in KV cache).
        // The last num_candidates entries of generated_ids are the current candidates (root +
        // N tree nodes), which are NOT yet in the KV cache and must be submitted as input_ids.
        const size_t past_accepted_len = generated_ids.size() - num_candidates;
        const size_t context_len = prompt_len + past_accepted_len;
        const size_t attn_len = context_len + num_candidates;

        // 1. input_ids: all N+1 candidates — root then tree nodes
        //    (last num_candidates entries of generated_ids).
        input_ids = ov::Tensor(ov::element::i64, {1, num_candidates});
        int64_t* ids_ptr = input_ids.data<int64_t>();
        const size_t gen_offset = generated_ids.size() - num_candidates;
        for (size_t i = 0; i < num_candidates; ++i) {
            ids_ptr[i] = generated_ids[gen_offset + i];
        }

        // 2. position_ids: context_len + tree depth of each candidate.
        //    tree_pos_ids[0] = 0 → root sits at position context_len (right after KV cache).
        //    tree_pos_ids[i] = depth → node sits at context_len + depth.
        position_ids = ov::Tensor(ov::element::i64, {1, num_candidates});
        int64_t* pos_ptr = position_ids.data<int64_t>();
        for (size_t i = 0; i < num_candidates; ++i) {
            pos_ptr[i] = static_cast<int64_t>(context_len) + static_cast<int64_t>(tree_pos_ids[i]);
        }

        // 3. attention_mask: all-ones over the full attention span.
        attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
        std::fill_n(attention_mask.data<int64_t>(), attn_len, 1);

        // 4. eagle_tree_mask: shape {1, 1, num_candidates, attn_len}.
        //    history columns [0, context_len): 0.0 (all candidates attend to prompt + past accepted).
        //    tree columns [context_len, attn_len): tree_mask_bin[i][j]==1 → 0.0, else -INF.
        eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, num_candidates, attn_len});
        float* mask = eagle_tree_mask.data<float>();
        const float neg_inf = -std::numeric_limits<float>::infinity();

        for (size_t i = 0; i < num_candidates; ++i) {
            float* row = mask + i * attn_len;
            // History region (prompt + past accepted tokens): fully accessible.
            std::fill_n(row, context_len, 0.0f);
            // Tree region: use tree_mask_bin[i][j] directly (includes root at index 0).
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

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Built model inputs (TARGET_VALIDATION): num_candidates=" + std::to_string(num_candidates) +
                              ", num_tree_nodes=" + std::to_string(num_tree_nodes) + ", prompt_len=" +
                              std::to_string(prompt_len) + ", past_accepted_len=" + std::to_string(past_accepted_len) +
                              ", context_len=" + std::to_string(context_len) + ", attn_len=" + std::to_string(attn_len),
                          m_verbose);
        return;
    }

    // Standard logic for other phases
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = current_sequence->get_generated_ids();

    const size_t prompt_len = prompt_ids.size();
    const size_t generated_len = generated_ids.size();
    const size_t total_len = prompt_len + generated_len;
    const size_t start_pos = total_len - input_token_count;

    OPENVINO_ASSERT(input_token_count > 0 && input_token_count <= total_len,
                    "Invalid input_token_count: ",
                    input_token_count,
                    ", total_len: ",
                    total_len);

    // Allocate tensors
    input_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    position_ids = ov::Tensor(ov::element::i64, {1, input_token_count});

    int64_t* input_ids_ptr = input_ids.data<int64_t>();
    int64_t* position_ids_ptr = position_ids.data<int64_t>();

    // Fill input_ids and position_ids from sequence
    if (start_pos < prompt_len) {
        // Part from prompt
        const size_t prompt_count = std::min(input_token_count, prompt_len - start_pos);
        std::copy_n(prompt_ids.data() + start_pos, prompt_count, input_ids_ptr);
        std::iota(position_ids_ptr, position_ids_ptr + prompt_count, static_cast<int64_t>(start_pos));

        // Part from generated (if any)
        if (input_token_count > prompt_count) {
            const size_t generated_count = input_token_count - prompt_count;
            std::copy_n(generated_ids.data(), generated_count, input_ids_ptr + prompt_count);
            std::iota(position_ids_ptr + prompt_count,
                      position_ids_ptr + prompt_count + generated_count,
                      static_cast<int64_t>(prompt_len));
        }
    } else {
        // All from generated
        const size_t generated_start = start_pos - prompt_len;
        std::copy_n(generated_ids.data() + generated_start, input_token_count, input_ids_ptr);
        std::iota(position_ids_ptr,
                  position_ids_ptr + input_token_count,
                  static_cast<int64_t>(prompt_len + generated_start));
    }

    // Build attention mask (always all 1s)
    const size_t attention_mask_len = static_cast<size_t>(position_ids_ptr[input_token_count - 1] + 1);
    attention_mask = ov::Tensor(ov::element::i64, {1, attention_mask_len});
    std::fill_n(attention_mask.data<int64_t>(), attention_mask_len, 1);

    // Build eagle_tree_mask based on inference phase
    switch (phase) {
    case InferencePhase::TARGET_PREFILL:
    case InferencePhase::DRAFT_INITIAL:
        // During prefill/initial phase: eagle_tree_mask is all zeros
        // Minimal shape: {1, 1, 1, 1}
        eagle_tree_mask = ov::Tensor(ov::element::f32, {1, 1, 1, 1});
        std::fill_n(eagle_tree_mask.data<float>(), 1, 0.0f);
        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Eagle tree mask (TARGET_PREFILL/DRAFT_INITIAL): {1, 1, 1, 1}, all zeros",
                          m_verbose);
        break;

    case InferencePhase::TARGET_VALIDATION:
    case InferencePhase::DRAFT_ITERATION:
        // Both cases are handled in the early-return paths above.
        OPENVINO_ASSERT(false, "TARGET_VALIDATION and DRAFT_ITERATION should be handled in the early return paths");
        break;
    }

    // Log input preparation details
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Built model inputs: input_token_count=" + std::to_string(input_token_count) + ", start_pos=" +
                          std::to_string(start_pos) + ", attn_mask_len=" + std::to_string(attention_mask_len),
                      m_verbose);
}

std::vector<int64_t> Eagle3InferWrapperBase::sample_tokens(const ov::Tensor& logits,
                                                           size_t input_token_count,
                                                           size_t sample_count,
                                                           size_t num_tokens_to_validate) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape: ", eagle3::format_shape(shape));
    OPENVINO_ASSERT(sample_count > 0 && sample_count <= shape[1],
                    "Invalid sample_count: ",
                    sample_count,
                    ", logits seq_len: ",
                    shape[1]);
    OPENVINO_ASSERT(input_token_count > 0, "Invalid input_token_count");

    const bool is_validation_mode = num_tokens_to_validate > 0;

    eagle3::log_debug(eagle3::PipelineStep::SAMPLE,
                      "sample_tokens: input_tokens=" + std::to_string(input_token_count) + ", sample_count=" +
                          std::to_string(sample_count) + ", validate=" + std::to_string(num_tokens_to_validate) +
                          ", logits_shape=" + eagle3::format_shape(shape),
                      m_verbose);

    auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    const size_t logits_seq_len = shape[1];
    const size_t vocab_size = shape[2];

    // Get num_sequences BEFORE slicing, as we need it for slicing logic
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

    eagle3::log_debug(eagle3::PipelineStep::SAMPLE,
                      "sliced_logits shape: " + eagle3::format_shape(sliced_logits.get_shape()),
                      m_verbose);

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

    eagle3::log_debug(eagle3::PipelineStep::SAMPLE,
                      "Processing " + std::to_string(num_sequences) + " sequence(s) after sampling",
                      m_verbose);

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

        eagle3::log_debug(eagle3::PipelineStep::SAMPLE,
                          "Sampled " + std::to_string(num_sequences) + " token(s) from " +
                              std::to_string(num_sequences) + " sequence(s): " + eagle3::format_tokens(result_tokens),
                          m_verbose);

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

        eagle3::log_debug(eagle3::PipelineStep::VALID,
                          "Validation result: collected " + std::to_string(result_tokens.size()) + " token(s) " +
                              "(accepted=" + std::to_string(result_tokens.size() - 1) + " + bonus=1)" +
                              ", tokens=" + eagle3::format_tokens(result_tokens),
                          m_verbose);

        return result_tokens;
    }
}

ov::Tensor Eagle3InferWrapperBase::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor Eagle3InferWrapperBase::get_hidden_features() const {
    auto hidden_state = m_request.get_tensor("last_hidden_state");
    const auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid hidden state shape: ", eagle3::format_shape(shape));

    const size_t output_seq_len = shape[1];
    const size_t hidden_size = shape[2];
    const size_t actual_seq_len = m_request.get_tensor("input_ids").get_shape()[1];

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

Eagle3TargetWrapper::Eagle3TargetWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {
    eagle3::log_debug(eagle3::PipelineStep::INIT, "Target model wrapper initialized", m_verbose);
}

void Eagle3TargetWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1,
                    "Expected input_ids shape [1, seq_len], got ",
                    eagle3::format_shape(shape));

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = shape[1];
    OPENVINO_ASSERT(seq_len > 0, "Empty prompt");

    TokenIds prompt_ids(ids_data, ids_data + seq_len);
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config, 0);

    OPENVINO_ASSERT(get_running_sequence_count() == 1,
                    "Expected single sequence after initialization, got ",
                    get_running_sequence_count());

    eagle3::log_debug(eagle3::PipelineStep::INIT,
                      "Target sequence initialized: prompt_len=" + std::to_string(seq_len),
                      m_verbose);
}

InferenceOutput Eagle3TargetWrapper::infer(const ov::Tensor& input_ids,
                                           const ov::Tensor& attention_mask,
                                           const ov::Tensor& position_ids,
                                           const ov::Tensor& eagle_tree_mask) {
    const size_t prompt_len = input_ids.get_shape()[1];

    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Target inference: " + std::to_string(prompt_len) + " tokens",
                      m_verbose);
    eagle3::log_model_inputs(input_ids, attention_mask, position_ids, eagle_tree_mask, m_verbose);

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
    output.hidden_features = get_hidden_features();

    eagle3::log_model_outputs(output.logits, output.hidden_features, m_verbose);
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Target inference done: " + std::to_string(time_us / 1000.0) + "ms",
                      m_verbose);

    return output;
}

InferResult Eagle3TargetWrapper::forward(const InferContext& ctx) {
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Target forward: input_token_count=" + std::to_string(ctx.input_token_count) + ", sample_count=" +
                          std::to_string(ctx.sample_count) + ", validate=" + std::to_string(ctx.num_tokens_to_validate),
                      m_verbose);
    // 1. Prepare inputs from sequence state
    ov::Tensor input_ids, attention_mask, position_ids, eagle_tree_mask;
    // Determine phase based on context
    InferencePhase phase =
        (ctx.num_tokens_to_validate > 0) ? InferencePhase::TARGET_VALIDATION : InferencePhase::TARGET_PREFILL;
    build_model_inputs(ctx.input_token_count,
                       input_ids,
                       attention_mask,
                       position_ids,
                       eagle_tree_mask,
                       phase,
                       ctx.iteration_id,
                       ctx.iteration_history);

    // 2. Infer
    auto output = infer(input_ids, attention_mask, position_ids, eagle_tree_mask);

    // 3. Sample (use sample_count for number of positions to sample from)
    auto sampled = sample_tokens(output.logits, ctx.input_token_count, ctx.sample_count, ctx.num_tokens_to_validate);

    // 4. Store hidden states to sequence for draft model to use
    get_current_sequence()->update_hidden_state(output.hidden_features);

    return InferResult{std::move(output), std::move(sampled)};
}

Eagle3DraftWrapper::Eagle3DraftWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {
    eagle3::log_debug(eagle3::PipelineStep::INIT, "Draft model wrapper initialized", m_verbose);
}

void Eagle3DraftWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1,
                    "Expected input_ids shape [1, seq_len], got ",
                    eagle3::format_shape(shape));

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t total_len = shape[1];
    OPENVINO_ASSERT(total_len >= 2, "Draft model requires at least 2 tokens");

    // Draft model uses tokens[1:] (Eagle3 specific behavior)
    TokenIds draft_prompt_ids(ids_data + 1, ids_data + total_len);
    m_sequence_group = std::make_shared<SequenceGroup>(1, draft_prompt_ids, config, 0);

    OPENVINO_ASSERT(get_running_sequence_count() == 1,
                    "Expected single sequence after initialization, got ",
                    get_running_sequence_count());

    eagle3::log_debug(eagle3::PipelineStep::INIT,
                      "Draft sequence initialized: prompt_len=" + std::to_string(draft_prompt_ids.size()) +
                          " (from original " + std::to_string(total_len) + ")",
                      m_verbose);
}

InferenceOutput Eagle3DraftWrapper::infer(const ov::Tensor& input_ids,
                                          const ov::Tensor& attention_mask,
                                          const ov::Tensor& position_ids,
                                          const ov::Tensor& eagle_tree_mask,
                                          const ov::Tensor& hidden_states) {
    const size_t input_token_count = input_ids.get_shape()[1];

    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Draft inference: " + std::to_string(input_token_count) + " tokens",
                      m_verbose);
    eagle3::log_model_inputs(input_ids, attention_mask, position_ids, eagle_tree_mask, hidden_states, m_verbose);

    // Set standard inputs
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);
    m_request.set_tensor("eagle_tree_mask", eagle_tree_mask);

    // Set hidden states (either from target model or internal)
    OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "hidden_states must be provided");
    auto shape = hidden_states.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Invalid hidden states shape: ", eagle3::format_shape(shape));

    eagle3::log_debug(eagle3::PipelineStep::ITER, "Using hidden states: " + eagle3::format_shape(shape), m_verbose);
    m_request.set_tensor("hidden_states", hidden_states);

    // Execute inference
    uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    // Collect outputs
    InferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features();

    eagle3::log_model_outputs(output.logits, output.hidden_features, m_verbose);
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Draft inference done: " + std::to_string(time_us / 1000.0) + "ms",
                      m_verbose);

    return output;
}

InferResult Eagle3DraftWrapper::forward(const InferContext& ctx) {
    eagle3::log_debug(eagle3::PipelineStep::ITER,
                      "Draft forward: input_token_count=" + std::to_string(ctx.input_token_count) +
                          ", use_target_hidden=" + std::to_string(ctx.use_target_hidden) +
                          ", iteration_id=" + std::to_string(ctx.iteration_id),
                      m_verbose);
    // 1. Prepare inputs
    ov::Tensor input_ids, attention_mask, position_ids, eagle_tree_mask;
    InferencePhase phase = ctx.use_target_hidden ? InferencePhase::DRAFT_INITIAL : InferencePhase::DRAFT_ITERATION;
    build_model_inputs(ctx.input_token_count,
                       input_ids,
                       attention_mask,
                       position_ids,
                       eagle_tree_mask,
                       phase,
                       ctx.iteration_id,
                       ctx.iteration_history,
                       ctx.past_generate_len);

    // 2. Get hidden states from appropriate source
    ov::Tensor hidden_states;
    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: Use target model's hidden state (single position, will be used for all sequences)
        OPENVINO_ASSERT(ctx.target_sequence, "target_sequence required when use_target_hidden=true");
        hidden_states = ctx.target_sequence->get_hidden_state();
        OPENVINO_ASSERT(hidden_states && hidden_states.get_size() > 0, "Source sequence contains invalid hidden state");
    } else {
        // DRAFT_ITERATION: Concatenate the full stored hidden-state path of every sequence.
        // After the DRAFT_INITIAL pass each sequence holds {1, 1, hidden_size} (the root).
        // After each subsequent iteration the stored tensor grows by one token per sequence,
        // so at iteration k each sequence holds {1, k, hidden_size}.
        // Concatenating all sequences yields {1, num_seqs * branch_len, hidden_size}, which
        // matches the flat input_ids layout produced by build_model_inputs (DRAFT_ITERATION).
        auto running_sequences = m_sequence_group->get_running_sequences();
        const size_t sequence_numb = running_sequences.size();
        OPENVINO_ASSERT(sequence_numb > 0, "No running sequences");

        std::vector<ov::Tensor> seq_hidden_states;
        seq_hidden_states.reserve(sequence_numb);
        for (size_t i = 0; i < sequence_numb; ++i) {
            auto seq_hidden = running_sequences[i]->get_hidden_state();
            OPENVINO_ASSERT(seq_hidden && seq_hidden.get_size() > 0, "Sequence ", i, " contains invalid hidden state");
            const auto& h_shape = seq_hidden.get_shape();
            OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1,
                            "Expected hidden state shape [1, branch_len, hidden_size], got: ",
                            eagle3::format_shape(h_shape));
            seq_hidden_states.push_back(seq_hidden);
        }

        hidden_states = utils::eagle3::concatenate_hidden_states(seq_hidden_states);

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Concatenated hidden states: " + eagle3::format_shape(hidden_states.get_shape()) + " from " +
                              std::to_string(sequence_numb) + " sequence(s)",
                          m_verbose);
    }

    // 3. Infer
    auto output = infer(input_ids, attention_mask, position_ids, eagle_tree_mask, hidden_states);

    // 4. Store hidden states BEFORE sampling so that each sequence holds its correct path
    //    hidden states regardless of how the sampler may later fork or prune paths.
    auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t sequence_numb = running_sequences.size();

    if (ctx.use_target_hidden) {
        // DRAFT_INITIAL: The output has a single new position (the root token).
        // All beam sequences start from the same root, so they all receive the same hidden state.
        auto next_hidden = utils::eagle3::slice_hidden_state_at_position(output.hidden_features,
                                                                         output.hidden_features.get_shape()[1] - 1);
        for (size_t i = 0; i < sequence_numb; ++i) {
            running_sequences[i]->update_hidden_state(next_hidden);
        }

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Stored root hidden state for " + std::to_string(sequence_numb) + " sequence(s)",
                          m_verbose);
    } else {
        // DRAFT_ITERATION: output.hidden_features has shape {1, total_tokens, hidden_size}
        // where total_tokens = num_seqs * branch_len (flat layout, same order as input_ids).
        // For each sequence we append the hidden state of its LAST branch token to the
        // sequence's accumulated hidden state, growing it from {1, k, H} to {1, k+1, H}.
        // This is done before sample_tokens so that if the sampler forks sequences the
        // per-sequence hidden state is already correctly assigned.
        const auto& hidden_shape = output.hidden_features.get_shape();
        OPENVINO_ASSERT(hidden_shape.size() == 3 && hidden_shape[0] == 1,
                        "Invalid hidden features shape: ",
                        eagle3::format_shape(hidden_shape));

        // Recover branch_len from the stored hidden state of the first sequence (= k tokens).
        // After this iteration it will become k+1.
        const size_t branch_len = running_sequences[0]->get_hidden_state().get_shape()[1];
        const size_t total_hidden_tokens = sequence_numb * branch_len;
        // Hardware may pad the sequence dimension at the front; valid tokens are at the tail.
        OPENVINO_ASSERT(hidden_shape[1] >= total_hidden_tokens,
                        "hidden_features seq_len (",
                        hidden_shape[1],
                        ") < num_seqs (",
                        sequence_numb,
                        ") * branch_len (",
                        branch_len,
                        ")");
        const size_t hidden_pad_offset = hidden_shape[1] - total_hidden_tokens;

        for (size_t i = 0; i < sequence_numb; ++i) {
            // The last branch token for sequence i sits at hidden_pad_offset + (i+1)*branch_len - 1.
            const size_t last_tok_pos = hidden_pad_offset + (i + 1) * branch_len - 1;
            auto new_tok_hidden = utils::eagle3::slice_hidden_state_at_position(output.hidden_features, last_tok_pos);

            // Append the new token's hidden state to the existing path: {1,k,H} → {1,k+1,H}.
            const auto existing_hidden = running_sequences[i]->get_hidden_state();
            auto updated_hidden = utils::eagle3::concatenate_hidden_states({existing_hidden, new_tok_hidden});
            running_sequences[i]->update_hidden_state(updated_hidden);

            eagle3::log_debug(eagle3::PipelineStep::ITER,
                              "Sequence " + std::to_string(i) + ": appended hidden state at flat pos " +
                                  std::to_string(last_tok_pos) + ", new stored shape " +
                                  eagle3::format_shape(updated_hidden.get_shape()),
                              m_verbose);
        }

        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Updated hidden states for " + std::to_string(sequence_numb) + " sequence(s)",
                          m_verbose);
    }

    // 5. Sample
    // For DRAFT_ITERATION the model produced logits with shape {1, total_tokens, vocab_size}
    // where total_tokens = num_seqs * branch_len (flat layout).  The sampler expects one logit
    // row per running sequence, i.e. shape {1, num_seqs, vocab_size}.  We must gather the
    // last-branch-token logit of each sequence (positions branch_len-1, 2*branch_len-1, …)
    // into a compact tensor before calling sample_tokens.
    //
    // For DRAFT_INITIAL the logits shape is {1, prompt_len, vocab_size} with a single sequence;
    // sample_tokens' existing tail-slice handles that correctly without any pre-processing.
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
        const size_t total_tokens = sequence_numb * prev_branch_len;
        // Hardware may pad the sequence dimension to a multiple of 8 at the front.
        // The valid tokens always occupy the tail: [lshape[1] - total_tokens, lshape[1]).
        OPENVINO_ASSERT(lshape[1] >= total_tokens,
                        "Logits seq_len (",
                        lshape[1],
                        ") < num_seqs (",
                        sequence_numb,
                        ") * branch_len (",
                        prev_branch_len,
                        ")");
        const size_t pad_offset = lshape[1] - total_tokens;

        const size_t vocab_size = lshape[2];
        logits_for_sampling = ov::Tensor(ov::element::f32, {1, sequence_numb, vocab_size});
        float* dst = logits_for_sampling.data<float>();
        const float* src = output.logits.data<const float>();

        for (size_t s = 0; s < sequence_numb; ++s) {
            // Last token of sequence s sits at flat position pad_offset + (s+1)*prev_branch_len - 1.
            const size_t flat_pos = pad_offset + (s + 1) * prev_branch_len - 1;
            std::copy_n(src + flat_pos * vocab_size, vocab_size, dst + s * vocab_size);
        }

        eagle3::log_debug(eagle3::PipelineStep::SAMPLE,
                          "Gathered logits for sampling: " + eagle3::format_shape(logits_for_sampling.get_shape()),
                          m_verbose);
    }

    auto sampled = sample_tokens(logits_for_sampling, ctx.input_token_count, 1);

    return InferResult{std::move(output), std::move(sampled)};
}

StatefulEagle3LLMPipeline::StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config) {
    // Initialize draft iterations from generation config
    ensure_num_assistant_tokens_is_set(m_generation_config);
    m_draft_iterations = m_generation_config.num_assistant_tokens;

    // Extract hidden_layers_list from draft model properties
    OPENVINO_ASSERT(draft_model_desc.properties.find("hidden_layers_list") != draft_model_desc.properties.end(),
                    "hidden_layers_list must be present in draft model properties");

    m_hidden_layers_to_abstract = draft_model_desc.properties.at("hidden_layers_list").as<std::vector<int32_t>>();

    OPENVINO_ASSERT(m_hidden_layers_to_abstract.size() == 3,
                    "Eagle3 requires exactly three layers for feature extraction, got: " +
                        std::to_string(m_hidden_layers_to_abstract.size()) +
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

    utils::eagle3::transform_hidden_state(target_model, m_hidden_layers_to_abstract);
    utils::eagle3::move_fc_from_draft_to_main(draft_model, target_model);
    utils::eagle3::transform_hidden_state(draft_model, {-1});

    const size_t validation_window = m_draft_iterations + 1;

    // Configure and create draft model
    auto draft_desc = draft_model_desc;
    if (draft_desc.device == "NPU") {
        draft_desc.properties["NPUW_EAGLE"] = "TRUE";
        draft_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = validation_window;
        draft_desc.properties["NPUW_ONLINE_PIPELINE"] = "NONE";
        draft_desc.properties["NPUW_DEVICES"] = "CPU";
    }
    m_draft = std::make_unique<Eagle3DraftWrapper>(draft_desc);

    m_draft->set_draft_target_mapping(d2t_mapping);

    // Configure and create target model
    auto target_desc = target_model_desc;
    if (target_desc.device == "NPU") {
        target_desc.properties["NPUW_EAGLE"] = "TRUE";
        target_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = validation_window;
        target_desc.properties["NPUW_SLICE_OUT"] = "NO";
        target_desc.properties["NPUW_DEVICES"] = "CPU";
    }
    m_target = std::make_unique<Eagle3TargetWrapper>(target_desc);

    eagle3::log_info("Pipeline initialized: draft_iterations=" + std::to_string(m_draft_iterations) +
                     ", validation_window=" + std::to_string(validation_window));
}

StatefulEagle3LLMPipeline::~StatefulEagle3LLMPipeline() {
    m_target->release_memory();
    m_draft->release_memory();
}

GenerationConfig StatefulEagle3LLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    // Call base class implementation to handle common defaults
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);

    // Apply Eagle3 specific validations
    const size_t prev_draft_iterations = m_draft_iterations;
    ensure_num_assistant_tokens_is_set(config);
    m_draft_iterations = config.num_assistant_tokens;

    // Log configuration changes
    if (m_draft_iterations != prev_draft_iterations) {
        if (m_draft_iterations == 0) {
            eagle3::log_info("Speculative decoding DISABLED (num_assistant_tokens=0)");
        } else {
            eagle3::log_debug("Draft iterations: " + std::to_string(prev_draft_iterations) + " -> " +
                                  std::to_string(m_draft_iterations),
                              is_verbose());
        }
    }

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

    eagle3::log_debug("=== GENERATION START ===", is_verbose());
    eagle3::log_debug("Prompt length: " + std::to_string(m_prompt_length) +
                          ", max_new_tokens: " + std::to_string(config.max_new_tokens) +
                          ", draft_iterations: " + std::to_string(m_draft_iterations),
                      is_verbose());

    // Reset model states
    m_target->reset_state();
    m_draft->reset_state();

    // Prepare sampling config with extended max_new_tokens to prevent premature termination
    // during draft generation. Actual length control is in the generation loop.
    auto sampling_config = config;
    sampling_config.max_new_tokens = config.max_new_tokens + m_draft_iterations + 1;

    m_draft->initialize_sequence(input_ids, sampling_config);

    // For tree search mode, temporarily disable tree expansion during initial token generation
    // We need to modify the GenerationConfig inside the SequenceGroup (not the local sampling_config)
    const bool is_tree_search_mode = sampling_config.is_tree_search();
    size_t original_tree_depth = 0;
    if (is_tree_search_mode) {
        original_tree_depth = sampling_config.eagle_tree_params.tree_depth;
        sampling_config.eagle_tree_params.tree_depth = 0;
        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Prefill: temporarily set tree_depth to 0 (original: " + std::to_string(original_tree_depth) +
                              ") for initial token generation",
                          is_verbose());
    }

    // Initialize sequences with sampling config
    m_target->initialize_sequence(input_ids, sampling_config);

    // Phase 1: Initial Prompt Processing (Prefill)
    eagle3::log_generation_step("PREFILL", 0, is_verbose());

    // Prefill: process all prompt tokens from sequence
    InferContext prefill_ctx;
    prefill_ctx.input_token_count = m_prompt_length;
    auto prefill_result = m_target->forward(prefill_ctx);
    OPENVINO_ASSERT(prefill_result.sampled_tokens.size() == 1, "Expected single token from prefill");
    auto initial_token = prefill_result.sampled_tokens[0];

    // Append initial token to draft model
    m_draft->append_tokens({initial_token});

    // Restore original tree_depth after initial token generation
    if (is_tree_search_mode) {
        auto& target_params =
            const_cast<ov::genai::GenerationConfig&>(m_target->get_sequence_group()->get_sampling_parameters());
        target_params.eagle_tree_params.tree_depth = original_tree_depth;
        eagle3::log_debug(eagle3::PipelineStep::ITER,
                          "Prefill: restored tree_depth to " + std::to_string(original_tree_depth),
                          is_verbose());
    }

    eagle3::log_debug("Initial token: " + std::to_string(initial_token), is_verbose());
    eagle3::log_sequence_state("after prefill",
                               m_prompt_length,
                               m_target->get_sequence_length(),
                               m_draft->get_sequence_length(),
                               m_target->get_generated_tokens(),
                               m_draft->get_generated_tokens(),
                               is_verbose());

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
        eagle3::log_generation_step("SPECULATIVE ITERATION", generated_tokens, is_verbose());

        auto result = run_speculative_iteration(input_token_count, static_cast<int64_t>(config.eos_token_id));

        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        // Update statistics
        total_draft_generated += m_draft_iterations;
        total_draft_accepted += result.accepted_tokens_count;
        eos_reached = result.eos_reached;
        generated_tokens++;

        // Prepare for next iteration (hidden states are stored in sequence)
        input_token_count = result.next_window_size;

        eagle3::log_debug("Iteration complete: accepted=" + std::to_string(result.accepted_tokens_count) + "/" +
                              std::to_string(m_draft_iterations) + ", eos=" + std::to_string(result.eos_reached),
                          is_verbose());
        eagle3::log_sequence_state("after iteration " + std::to_string(generated_tokens),
                                   m_prompt_length,
                                   m_target->get_sequence_length(),
                                   m_draft->get_sequence_length(),
                                   m_target->get_generated_tokens(),
                                   m_draft->get_generated_tokens(),
                                   is_verbose());
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

    eagle3::log_debug("=== GENERATION END ===", is_verbose());
    return results;
}

StatefulEagle3LLMPipeline::SpeculativeResult StatefulEagle3LLMPipeline::run_speculative_iteration(
    size_t input_token_count,
    int64_t eos_token_id) {
    SpeculativeResult result;

    OPENVINO_ASSERT(m_target->get_running_sequence_count() == 1 && m_draft->get_running_sequence_count() == 1,
                    "Eagle3 speculative iteration requires single sequence per model");

    auto target_hidden_states = m_target->get_current_sequence()->get_hidden_state();
    OPENVINO_ASSERT(target_hidden_states && target_hidden_states.get_size() > 0,
                    "Target model contains invalid hidden state for speculation");

    eagle3::log_debug("--- Draft Phase Start ---", is_verbose());
    eagle3::log_debug("Input: input_token_count=" + std::to_string(input_token_count) +
                          ", hidden_shape=" + eagle3::format_shape(target_hidden_states.get_shape()),
                      is_verbose());

    // Record pre-draft sequence lengths for potential rollback
    const size_t pre_draft_token_len = m_draft->get_sequence_length();

    // past_generate_len = number of tokens already in the target model's generated_ids before
    // this speculative window begins.  This equals the number of tokens the target has accepted
    // across all previous speculative iterations (including the initial_token from prefill).
    // DRAFT_ITERATION uses this offset to locate the current root inside the draft's accumulated
    // generated_ids, since the draft mirrors the target's accepted history token-for-token.
    const size_t past_generate_len = m_target->get_current_sequence()->get_generated_ids().size();

    // Clear stale EagleMetaData left over from the previous iteration so that
    // select_top_k starts from a clean slate.
    m_draft->get_current_sequence()->set_eagle_metadata({});

    // Create iteration history to track all generated tokens across iterations
    auto iteration_history = std::make_shared<std::vector<int64_t>>();
    iteration_history->reserve(m_draft_iterations * 10);  // Reserve space for efficiency

    // Step 1: Generate first draft token using target hidden states
    InferContext first_ctx;
    first_ctx.input_token_count = input_token_count;
    first_ctx.use_target_hidden = true;
    first_ctx.target_sequence = m_target->get_current_sequence();
    first_ctx.iteration_id = 0;
    first_ctx.iteration_history = iteration_history;
    first_ctx.past_generate_len = past_generate_len;
    auto first_result = m_draft->forward(first_ctx);

    // first_result.sampled_tokens contains tokens from all sequences (one token per sequence)
    const auto& first_draft_tokens = first_result.sampled_tokens;
    OPENVINO_ASSERT(!first_draft_tokens.empty(), "Expected at least one token from first draft");

    eagle3::log_debug("First draft tokens (" + std::to_string(first_draft_tokens.size()) +
                          " sequence(s)): " + eagle3::format_tokens(first_draft_tokens),
                      is_verbose());

    // Collect draft candidates - store all tokens from first iteration
    std::vector<int64_t> draft_candidates;
    draft_candidates.reserve(m_draft_iterations * first_draft_tokens.size());
    draft_candidates.insert(draft_candidates.end(), first_draft_tokens.begin(), first_draft_tokens.end());

    // Record first iteration tokens in history
    iteration_history->insert(iteration_history->end(), first_draft_tokens.begin(), first_draft_tokens.end());

    // Step 2: Generate additional draft tokens using internal hidden states
    for (size_t i = 1; i < m_draft_iterations; ++i) {
        InferContext more_ctx;
        more_ctx.input_token_count = 1;  // This will be updated by build_model_inputs based on num sequences
        more_ctx.use_target_hidden = false;
        more_ctx.iteration_id = i;                       // Pass the current iteration index
        more_ctx.iteration_history = iteration_history;  // Share the history
        more_ctx.past_generate_len = past_generate_len;
        auto more_result = m_draft->forward(more_ctx);

        const auto& draft_tokens = more_result.sampled_tokens;
        OPENVINO_ASSERT(!draft_tokens.empty(), "Expected tokens from draft iteration ", i);

        eagle3::log_debug("Draft iteration " + std::to_string(i) + " tokens (" + std::to_string(draft_tokens.size()) +
                              " sequence(s)): " + eagle3::format_tokens(draft_tokens),
                          is_verbose());

        // Collect all tokens from this iteration
        draft_candidates.insert(draft_candidates.end(), draft_tokens.begin(), draft_tokens.end());

        // Record this iteration's tokens in history
        iteration_history->insert(iteration_history->end(), draft_tokens.begin(), draft_tokens.end());
    }

    eagle3::log_debug("Draft candidates: " + eagle3::format_tokens(draft_candidates), is_verbose());
    eagle3::log_debug("--- Draft Phase End ---", is_verbose());

    // Step 3: Validate draft tokens with target model
    eagle3::log_debug("--- Validation Phase Start ---", is_verbose());

    // In tree-search mode the draft model generates a tree of N+1 candidates:
    //   draft generated_ids tail = [root, node_1, …, node_N]
    //   EagleMetaData.tree_mask  = (N+1)×(N+1), index 0 = root.
    // The root token is the LAST token sampled by the target in the previous iteration.
    // It has NOT been fed back to the target KV cache yet, so the target must process
    // all N+1 candidates (root + N tree nodes) during validation.
    const auto& draft_gen_ids = m_draft->get_current_sequence()->get_generated_ids();
    const auto& draft_metadata = m_draft->get_current_sequence()->get_eagle_metadata();
    const size_t num_candidates = draft_metadata.tree_mask.size();  // N+1 (root + N nodes)
    OPENVINO_ASSERT(num_candidates >= 2, "Expected at least root + one tree node");
    const size_t num_tree_nodes = num_candidates - 1;  // N non-root tokens (used for validation count)

    // The EagleMetaData is written into the DRAFT sequence by select_top_k.
    // Sync it to the TARGET sequence so that build_model_inputs can read it.
    m_target->get_current_sequence()->set_eagle_metadata(draft_metadata);

    // Sync all N+1 candidate tokens to target sequence.
    // Target's generated_ids already contains the root (appended by the sampler after prefill/
    // previous iteration).  Append the N non-root tree-node tokens (draft generated_ids[1..N]).
    OPENVINO_ASSERT(draft_gen_ids.size() >= num_candidates,
                    "draft generated_ids too short: ",
                    draft_gen_ids.size(),
                    " < num_candidates: ",
                    num_candidates);
    // draft_gen_ids tail layout: [..., root, node_1, ..., node_N]
    // The last num_candidates entries are: [root, node_1, ..., node_N].
    // The root is already present in the target sequence; append only the N non-root nodes.
    const size_t draft_tail_offset = draft_gen_ids.size() - num_candidates;
    for (size_t i = 1; i < num_candidates; ++i) {
        m_target->get_current_sequence()->append_token(draft_gen_ids[draft_tail_offset + i], 0.0f);
    }

    eagle3::log_debug("Synced draft metadata and " + std::to_string(num_tree_nodes) +
                          " tree-node tokens to target sequence (total candidates: " + std::to_string(num_candidates) +
                          ")",
                      is_verbose());

    // 打印出目前 draft 和 target group sequence 里面的 sequence 存储的generated ids 内容
    eagle3::log_debug("Draft generated_ids: " + eagle3::format_tokens(draft_gen_ids), is_verbose());
    eagle3::log_debug(
        "Target generated_ids: " + eagle3::format_tokens(m_target->get_current_sequence()->get_generated_ids()),
        is_verbose());
    eagle3::log_debug("past_generate_len: " + std::to_string(past_generate_len), is_verbose());

    // input_token_count = num_candidates: the target processes all N+1 tokens (root + N nodes).
    // num_tokens_to_validate = num_tree_nodes: the sampler checks the N non-root draft tokens.
    // sample_count = num_candidates: one output position per input token.
    InferContext val_ctx;
    val_ctx.input_token_count = num_candidates;
    val_ctx.sample_count = num_candidates;
    val_ctx.num_tokens_to_validate = num_tree_nodes;
    auto val_result = m_target->forward(val_ctx);

    // Sampler validates draft tokens and returns accepted + new sampled token
    auto validated_tokens = val_result.sampled_tokens;

    // Result: [accepted_draft_tokens..., new_sampled_token]
    const size_t accepted_count = validated_tokens.size() - 1;
    const int64_t target_predicted_token = validated_tokens.back();
    const size_t total_accepted_tokens = validated_tokens.size();
    // Target KV cache now holds num_candidates new tokens; keep only total_accepted_tokens.
    const size_t tokens_to_remove = num_candidates - total_accepted_tokens;

    eagle3::log_debug("Validation result: accepted=" + std::to_string(accepted_count) + "/" +
                          std::to_string(num_tree_nodes) + ", new_token=" + std::to_string(target_predicted_token),
                      is_verbose());
    eagle3::log_debug("Validated tokens: " + eagle3::format_tokens(validated_tokens), is_verbose());

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
        eagle3::log_debug("KV cache trimmed: removed " + std::to_string(tokens_to_remove) + " rejected tokens",
                          is_verbose());
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
    OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1,
                    "Invalid hidden state shape: ",
                    eagle3::format_shape(h_shape));

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

    result.accepted_tokens_count = accepted_count;
    result.next_window_size = accepted_count + 1;
    result.validated_tokens = std::move(validated_tokens);
    result.eos_reached = (target_predicted_token == eos_token_id);

    eagle3::log_debug("--- Validation Phase End ---", is_verbose());
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
