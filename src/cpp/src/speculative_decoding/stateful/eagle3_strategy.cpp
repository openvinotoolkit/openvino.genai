// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>

#include "continuous_batching/timer.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "sampling/sampler.hpp"
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

/// @brief Copies one hidden-state row (one token) from src[src_position] into dst[dst_position].
/// Both tensors must have shape [1, *, hidden_size] and identical element type.
void copy_hidden_state_row(const ov::Tensor& src, size_t src_position, ov::Tensor& dst, size_t dst_position) {
    const auto& src_shape = src.get_shape();
    const auto& dst_shape = dst.get_shape();
    OPENVINO_ASSERT(src_shape.size() == 3, "copy_hidden_state_row: src must be 3D");
    OPENVINO_ASSERT(dst_shape.size() == 3, "copy_hidden_state_row: dst must be 3D");
    OPENVINO_ASSERT(src_position < src_shape[1], "copy_hidden_state_row: src_position out of range");
    OPENVINO_ASSERT(dst_position < dst_shape[1], "copy_hidden_state_row: dst_position out of range");
    OPENVINO_ASSERT(src_shape[2] == dst_shape[2], "copy_hidden_state_row: hidden_size mismatch");
    OPENVINO_ASSERT(src.get_element_type() == dst.get_element_type(), "copy_hidden_state_row: element type mismatch");
    const size_t hidden_size = src_shape[2];
    const size_t elem_size = src.get_element_type().size();
    const size_t row_bytes = hidden_size * elem_size;
    const auto* src_ptr = static_cast<const uint8_t*>(src.data()) + src_position * row_bytes;
    auto* dst_ptr = static_cast<uint8_t*>(dst.data()) + dst_position * row_bytes;
    std::memcpy(dst_ptr, src_ptr, row_bytes);
}

/// @brief Fills an eagle_tree_mask tensor with a single 0.0 scalar.
void set_eagle_tree_mask_scalar_zero(ov::Tensor& mask) {
    std::memset(mask.data(), 0, mask.get_byte_size());
}

template <typename T>
void fill_causal_square_mask_impl(ov::Tensor& mask, size_t seq_len) {
    const T neg_inf = -std::numeric_limits<T>::infinity();
    T* mask_ptr = mask.data<T>();
    std::fill_n(mask_ptr, seq_len * seq_len, neg_inf);
    for (size_t i = 0; i < seq_len; ++i) {
        std::fill_n(mask_ptr + i * seq_len, i + 1, T{});
    }
}

/// @brief Writes a lower-triangular causal mask {1, 1, seq_len, seq_len}:
/// on/below diagonal = 0.0, above = -inf.  Dispatches on mask type (f32 or f16).
void fill_causal_square_mask(ov::Tensor& mask, size_t seq_len) {
    const auto type = mask.get_element_type();
    if (type == ov::element::f32) {
        fill_causal_square_mask_impl<float>(mask, seq_len);
    } else if (type == ov::element::f16) {
        fill_causal_square_mask_impl<ov::float16>(mask, seq_len);
    } else {
        OPENVINO_ASSERT(false, "Unsupported eagle_tree_mask element type: ", type);
    }
}

template <typename T>
void fill_block_causal_mask_impl(ov::Tensor& mask, size_t num_seqs, size_t branch_len) {
    const size_t total_tokens = num_seqs * branch_len;
    const T neg_inf = -std::numeric_limits<T>::infinity();
    T* mask_ptr = mask.data<T>();
    std::fill_n(mask_ptr, total_tokens * total_tokens, neg_inf);

    for (size_t s = 0; s < num_seqs; ++s) {
        const size_t path_col_start = s * branch_len;
        for (size_t t = 0; t < branch_len; ++t) {
            T* row_ptr = mask_ptr + (s * branch_len + t) * total_tokens;
            std::fill_n(row_ptr + path_col_start, t + 1, T{});
        }
    }
}

/// @brief Writes a block-causal mask {1, 1, num_seqs*branch_len, num_seqs*branch_len} for
/// DRAFT_ITERATION: within-sequence causal, cross-sequence blocked.  Dispatches on mask type.
void fill_block_causal_mask(ov::Tensor& mask, size_t num_seqs, size_t branch_len) {
    const auto type = mask.get_element_type();
    if (type == ov::element::f32) {
        fill_block_causal_mask_impl<float>(mask, num_seqs, branch_len);
    } else if (type == ov::element::f16) {
        fill_block_causal_mask_impl<ov::float16>(mask, num_seqs, branch_len);
    } else {
        OPENVINO_ASSERT(false, "Unsupported eagle_tree_mask element type: ", type);
    }
}

template <typename T>
void fill_tree_mask_from_bin_impl(ov::Tensor& mask, const std::vector<std::vector<uint8_t>>& tree_mask_bin) {
    const size_t num_candidates = tree_mask_bin.size();
    const T neg_inf = -std::numeric_limits<T>::infinity();
    T* mask_ptr = mask.data<T>();
    for (size_t i = 0; i < num_candidates; ++i) {
        T* row = mask_ptr + i * num_candidates;
        for (size_t j = 0; j < num_candidates; ++j) {
            row[j] = (tree_mask_bin[i][j] == 1) ? T{} : neg_inf;
        }
    }
}

/// @brief Writes a tree-ancestor mask {1, 1, N, N} from a binary matrix:
/// tree_mask_bin[i][j]==1 -> 0.0 (open), else -inf.  Dispatches on mask type.
void fill_tree_mask_from_bin(ov::Tensor& mask, const std::vector<std::vector<uint8_t>>& tree_mask_bin) {
    const auto type = mask.get_element_type();
    if (type == ov::element::f32) {
        fill_tree_mask_from_bin_impl<float>(mask, tree_mask_bin);
    } else if (type == ov::element::f16) {
        fill_tree_mask_from_bin_impl<ov::float16>(mask, tree_mask_bin);
    } else {
        OPENVINO_ASSERT(false, "Unsupported eagle_tree_mask element type: ", type);
    }
}

}  // anonymous namespace

namespace ov::genai {

// ---------------------------------------------------------------------------
// Input builders
// ---------------------------------------------------------------------------

InputTensors Eagle3TargetWrapper::build_prefill_inputs() const {
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");
    const auto current_sequence = m_sequence_group->get_running_sequences().at(0);

    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = current_sequence->get_generated_ids();
    const size_t prompt_len = prompt_ids.size();

    OPENVINO_ASSERT(prompt_len > 0, "Empty prompt");
    OPENVINO_ASSERT(generated_ids.empty(), "TARGET_PREFILL expects no generated tokens, got ", generated_ids.size());

    InputTensors result;
    result.input_ids = ov::Tensor(ov::element::i64, {1, prompt_len});
    result.position_ids = ov::Tensor(ov::element::i64, {1, prompt_len});
    int64_t* ids_ptr = result.input_ids.data<int64_t>();
    int64_t* pos_ptr = result.position_ids.data<int64_t>();

    std::copy_n(prompt_ids.data(), prompt_len, ids_ptr);
    std::iota(pos_ptr, pos_ptr + prompt_len, int64_t{0});

    result.attention_mask = ov::Tensor(ov::element::i64, {1, prompt_len});
    std::fill_n(result.attention_mask.data<int64_t>(), prompt_len, 1);

    // TARGET_PREFILL processes tokens causally; no tree attention mask is needed.
    result.eagle_tree_mask = ov::Tensor(m_eagle_tree_mask_type, {1, 1, 1, 1});
    set_eagle_tree_mask_scalar_zero(result.eagle_tree_mask);

    return result;
}

InputTensors Eagle3DraftWrapper::build_initial_inputs(size_t input_token_count) const {
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");
    const auto current_sequence = m_sequence_group->get_running_sequences().at(0);

    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = current_sequence->get_generated_ids();
    const size_t prompt_len = prompt_ids.size();

    OPENVINO_ASSERT(prompt_len > 0, "Empty draft prompt");
    OPENVINO_ASSERT(!generated_ids.empty(), "DRAFT_INITIAL requires at least one generated token");

    const size_t total_len = prompt_len + generated_ids.size();
    OPENVINO_ASSERT(input_token_count > 0 && input_token_count <= total_len,
                    "Invalid input_token_count: ",
                    input_token_count,
                    ", total_len: ",
                    total_len);

    const size_t start_pos = total_len - input_token_count;

    InputTensors result;
    result.input_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    result.position_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    int64_t* ids_ptr = result.input_ids.data<int64_t>();
    int64_t* pos_ptr = result.position_ids.data<int64_t>();

    // Treat prompt_ids and generated_ids as one contiguous logical sequence via index lookup.
    const auto get_token = [&](size_t idx) -> int64_t {
        return idx < prompt_len ? prompt_ids[idx] : generated_ids[idx - prompt_len];
    };
    for (size_t i = 0; i < input_token_count; ++i) {
        ids_ptr[i] = get_token(start_pos + i);
    }

    // Position IDs are contiguous absolute offsets starting at start_pos.
    std::iota(pos_ptr, pos_ptr + input_token_count, static_cast<int64_t>(start_pos));

    result.attention_mask = ov::Tensor(ov::element::i64, {1, total_len});
    std::fill_n(result.attention_mask.data<int64_t>(), total_len, 1);

    // eagle_tree_mask shape depends on whether an intra-input causal pattern is needed:
    //   - start_pos == 0: full prefill; the model applies its built-in causal mask, so
    //     a scalar mask {1,1,1,1} suffices.
    //   - input_token_count == 1: single token; no intra-input relationship to encode.
    //   - otherwise: {1, 1, input_token_count, input_token_count} causal mask to prevent
    //     earlier tokens from attending to later ones.
    if (start_pos == 0 || input_token_count == 1) {
        result.eagle_tree_mask = ov::Tensor(m_eagle_tree_mask_type, {1, 1, 1, 1});
        set_eagle_tree_mask_scalar_zero(result.eagle_tree_mask);
    } else {
        result.eagle_tree_mask = ov::Tensor(m_eagle_tree_mask_type, {1, 1, input_token_count, input_token_count});
        fill_causal_square_mask(result.eagle_tree_mask, input_token_count);
    }

    return result;
}

InputTensors Eagle3DraftWrapper::build_iteration_inputs(size_t past_accepted_token_count) const {
    // All running sequences (beam paths) are concatenated into one flat batch so a single
    // infer call handles all paths.  The first past_accepted_token_count entries of each
    // sequence's generated_ids are already in the draft KV cache; only the tail (branch_len
    // tokens) needs to be re-submitted.
    //
    // Example: num_seqs=2, branch_len=2, past_accepted_token_count=1, prompt_len=5
    //   seq0 generated_ids: [root, 42, 107]   seq1: [root, 99, 12]
    //   input_ids    : [42, 107, 99, 12]       shape {1, 4}
    //   history_len  : 5 + 1 = 6
    //   position_ids : [6, 7, 6, 7]            (history_len + t per token, reset per seq)
    //   tree_mask    : {1, 1, 4, 4}            (square, history columns implied)
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");
    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const size_t prompt_len = prompt_ids.size();

    const auto running_sequences = m_sequence_group->get_running_sequences();
    const size_t num_seqs = running_sequences.size();
    OPENVINO_ASSERT(num_seqs > 0, "No running sequences");

    const size_t full_path_len = running_sequences.at(0)->get_generated_ids().size();
    OPENVINO_ASSERT(full_path_len > past_accepted_token_count,
                    "DRAFT_ITERATION: generated_ids length ",
                    full_path_len,
                    " <= past_accepted_token_count ",
                    past_accepted_token_count);
    const size_t branch_len = full_path_len - past_accepted_token_count;
    const size_t total_tokens = num_seqs * branch_len;

    // history_len: tokens already in the KV cache (prompt + accepted history).
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
        const size_t offset = s * branch_len;
        std::copy_n(gen.data() + past_accepted_token_count, branch_len, ids_ptr + offset);
        std::iota(pos_ptr + offset, pos_ptr + offset + branch_len, static_cast<int64_t>(history_len));
    }

    // 2. attention_mask
    const size_t attn_len = history_len + total_tokens;
    result.attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
    std::fill_n(result.attention_mask.data<int64_t>(), attn_len, 1);

    // 3. eagle_tree_mask — square shape {1, 1, total_tokens, total_tokens}.
    //    Only encodes inter-token relationships among current input tokens.
    //    Each row opens its own path's columns causally; cross-sequence is blocked.
    //
    //    Example (num_seqs=2, branch_len=2):
    //      cols:           0    1  |   2    3     ← seq0=[0,1], seq1=[2,3]
    //      row0 (s0,t=0):  0  -inf | -inf -inf
    //      row1 (s0,t=1):  0    0  | -inf -inf
    //      row2 (s1,t=0): -inf -inf |   0  -inf
    //      row3 (s1,t=1): -inf -inf |   0    0
    result.eagle_tree_mask = ov::Tensor(m_eagle_tree_mask_type, {1, 1, total_tokens, total_tokens});
    fill_block_causal_mask(result.eagle_tree_mask, num_seqs, branch_len);

    return result;
}

InputTensors Eagle3TargetWrapper::build_validation_inputs() const {
    // Submits all candidates (root + N tree nodes) to the target model in one pass.
    // The root is the last token the target produced; its KV was not written back yet,
    // so it must be re-submitted together with the draft tree nodes for validation.
    //
    // tree_mask[i][j] (from TreeMetaData, set by select_top_k):
    //   1 → candidate j is an ancestor of i (attend allowed → 0.0)
    //   0 → not an ancestor (blocked → -INF)
    // tree_position_ids[i]: tree depth of candidate i (root = 0).
    //
    // eagle_tree_mask layout — square {num_candidates x num_candidates}:
    //   tree_mask (4×4):  1 0 0 0      row i attends to col j iff tree_mask[i][j]==1
    //                     1 1 0 0
    //                     1 0 1 0
    //                     1 1 0 1
    //   cols:   root  n1    n2    n3
    //   row0:    0   -inf  -inf  -inf
    //   row1:    0     0   -inf  -inf
    //   row2:    0   -inf    0   -inf
    //   row3:    0     0   -inf    0
    OPENVINO_ASSERT(m_sequence_group, "SequenceGroup not initialized");
    const auto current_sequence = m_sequence_group->get_running_sequences().at(0);

    const size_t prompt_len = m_sequence_group->get_prompt_ids().size();
    const auto& generated_ids = current_sequence->get_generated_ids();
    const auto& metadata = current_sequence->get_tree_metadata();
    const auto& tree_mask_bin = metadata.tree_mask;
    const auto& tree_pos_ids = metadata.tree_position_ids;

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
                    " entries, need >= ",
                    num_candidates,
                    " (root + N tree nodes)");

    // context_len: prompt + previously accepted tokens already in the KV cache.
    const size_t past_accepted_len = generated_ids.size() - num_candidates;
    const size_t context_len = prompt_len + past_accepted_len;
    const size_t attn_len = context_len + num_candidates;

    InputTensors result;

    // 1. input_ids: all N+1 candidates — root then tree nodes.
    result.input_ids = ov::Tensor(ov::element::i64, {1, num_candidates});
    std::copy_n(generated_ids.data() + past_accepted_len, num_candidates, result.input_ids.data<int64_t>());

    // 2. position_ids: context_len + tree depth of each candidate.
    result.position_ids = ov::Tensor(ov::element::i64, {1, num_candidates});
    int64_t* pos_ptr = result.position_ids.data<int64_t>();
    for (size_t i = 0; i < num_candidates; ++i) {
        pos_ptr[i] = static_cast<int64_t>(context_len) + static_cast<int64_t>(tree_pos_ids[i]);
    }

    // 3. attention_mask
    result.attention_mask = ov::Tensor(ov::element::i64, {1, attn_len});
    std::fill_n(result.attention_mask.data<int64_t>(), attn_len, 1);

    // 4. eagle_tree_mask: square shape {1, 1, num_candidates, num_candidates}.
    //    Only encodes tree ancestor relationships among current candidates.
    result.eagle_tree_mask = ov::Tensor(m_eagle_tree_mask_type, {1, 1, num_candidates, num_candidates});
    fill_tree_mask_from_bin(result.eagle_tree_mask, tree_mask_bin);

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

    // Cache Eagle3 port types; precision follows the IR.
    const auto compiled_model = m_request.get_compiled_model();
    m_eagle_tree_mask_type = compiled_model.input("eagle_tree_mask").get_element_type();
    m_hidden_output_type = compiled_model.output("last_hidden_state").get_element_type();
    m_logits_type = compiled_model.output("logits").get_element_type();
    OPENVINO_ASSERT(m_eagle_tree_mask_type == ov::element::f32 || m_eagle_tree_mask_type == ov::element::f16,
                    "eagle_tree_mask element type must be f32 or f16, got ",
                    m_eagle_tree_mask_type);
    OPENVINO_ASSERT(m_hidden_output_type == ov::element::f32 || m_hidden_output_type == ov::element::f16,
                    "last_hidden_state element type must be f32 or f16, got ",
                    m_hidden_output_type);
    // The sampler reads logits as `logits.data<const float>()`, so the model must output f32.
    OPENVINO_ASSERT(m_logits_type == ov::element::f32, "logits element type must be f32, got ", m_logits_type);

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
                    ". Valid range: (0, current_len)");

    if (m_device != "NPU") {
        utils::CacheState state(m_cache_types);
        state.num_tokens_to_trim = tokens_to_remove;
        state.seq_length_axis = m_kv_axes_pos.seq_len;
        state.reset_mem_state = false;
        utils::trim_kv_cache(m_request, state, {});
    }
}

void Eagle3InferWrapperBase::set_npu_sampling_result(size_t num_candidates,
                                                     const std::vector<size_t>& accepted_indices) {
    const size_t num_accepted = accepted_indices.size();

    constexpr const char* STATE_NAME = "npuw_eagle3_sampling_result";
    auto states = m_request.query_state();
    for (auto& state : states) {
        if (state.get_name() != STATE_NAME) {
            continue;
        }
        // Allocate a fresh tensor rather than reusing get_state(). get_state() returns
        // a trimmed view whose size depends on the previous write, making it unreliable
        // for variable-length candidate sets. set_state() copies into a fixed 514-element
        // internal buffer, so any tensor up to that size is safe.
        static constexpr size_t kSamplingStateHeaderSize = 2;  // [num_candidates, num_accepted]
        static constexpr size_t kSamplingStateCapacity = 514;
        const size_t tensor_size = kSamplingStateHeaderSize + num_candidates;
        OPENVINO_ASSERT(tensor_size <= kSamplingStateCapacity,
                        "npuw_eagle3_sampling_result requires tensor_size <= ",
                        kSamplingStateCapacity,
                        ", but got ",
                        tensor_size);
        ov::Tensor tensor(ov::element::i64, ov::Shape{tensor_size});

        int64_t* data = tensor.data<int64_t>();
        data[0] = static_cast<int64_t>(num_candidates);
        data[1] = static_cast<int64_t>(num_accepted);
        std::fill_n(data + kSamplingStateHeaderSize, num_candidates, int64_t{0});
        for (const size_t idx : accepted_indices) {
            OPENVINO_ASSERT(idx < num_candidates, "accepted_index ", idx, " is out of range [0, ", num_candidates, ")");
            data[kSamplingStateHeaderSize + idx] = 1;
        }
        state.set_state(tensor);
        return;
    }

    OPENVINO_ASSERT(false,
                    std::string("VariableState '") + STATE_NAME +
                        "' not found. Requires NPUW-compiled model with eagle3 sampling support.");
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
                                            size_t num_validated_tokens,
                                            bool is_validation) {
    const auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    sequence_group->schedule_tokens(input_token_count);
    sequence_group->set_output_seq_len(sample_count);
    sequence_group->set_num_validated_tokens(num_validated_tokens);

    m_sampler.sample({sequence_group}, logits, is_validation);
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

std::vector<int64_t> Eagle3InferWrapperBase::sample_next_tokens(const ov::Tensor& logits, size_t input_token_count) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape: ", shape);
    OPENVINO_ASSERT(input_token_count > 0, "Invalid input_token_count");

    const auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    const auto running_sequences = sequence_group->get_running_sequences();
    const size_t num_sequences = running_sequences.size();
    OPENVINO_ASSERT(num_sequences > 0, "No running sequences");

    // Logits are expected to be pre-trimmed by the caller (forward()) so that
    // shape[1] == num_sequences.
    OPENVINO_ASSERT(shape[1] == num_sequences,
                    "Logits seq_len (",
                    shape[1],
                    ") != num_sequences (",
                    num_sequences,
                    "). Caller must normalize logits before sampling.");

    invoke_sampler(logits, input_token_count, /*sample_count=*/1, /*num_validated_tokens=*/0, /*is_validation=*/false);
    // Draft EOS must not terminate the sequence: the target model may reject it, and the
    // next iteration still needs every sequence reachable.
    restore_running_status();

    // Re-fetch running sequences after sampler — advance_draft_layer may have forked sequences.
    const auto updated_sequences = sequence_group->get_running_sequences();
    const size_t updated_count = updated_sequences.size();

    // Collect the last generated token from each sequence.
    std::vector<int64_t> result_tokens;
    result_tokens.reserve(updated_count);
    for (size_t seq_idx = 0; seq_idx < updated_count; ++seq_idx) {
        const auto& generated_ids = updated_sequences[seq_idx]->get_generated_ids();
        OPENVINO_ASSERT(!generated_ids.empty(), "Sequence ", seq_idx, " has no generated tokens");
        result_tokens.push_back(generated_ids.back());
    }

    record_generated_tokens(updated_count);
    return result_tokens;
}

std::vector<int64_t> Eagle3InferWrapperBase::sample_and_validate(const ov::Tensor& logits,
                                                                 size_t input_token_count,
                                                                 size_t num_candidates,
                                                                 size_t num_tokens_to_validate) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape: ", shape);
    OPENVINO_ASSERT(input_token_count > 0, "Invalid input_token_count");

    const auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    const auto running_sequences = sequence_group->get_running_sequences();
    OPENVINO_ASSERT(running_sequences.size() == 1, "Target-model pass expects exactly one running sequence");

    // Record how many tokens are generated before the sampler runs so the
    // newly accepted tokens can be sliced out afterwards.
    // Prefill (N=0):  generated_ids = []                           → result_start = 0
    // Validation (N>0): generated_ids = [...history, root, n_1..n_N] → result_start = history_len + 1
    const auto& gen_before = running_sequences.at(0)->get_generated_ids();
    OPENVINO_ASSERT(gen_before.size() >= num_tokens_to_validate,
                    "generated_ids too short before sampling: ",
                    gen_before.size(),
                    " < num_tokens_to_validate: ",
                    num_tokens_to_validate);
    const size_t result_start = gen_before.size() - num_tokens_to_validate;

    // Logits are expected to be pre-trimmed by the caller (forward()) so that
    // shape[1] == num_candidates.
    OPENVINO_ASSERT(shape[1] == num_candidates,
                    "Logits seq_len (",
                    shape[1],
                    ") != num_candidates (",
                    num_candidates,
                    "). Caller must normalize logits before validation.");

    // Target-model sampling always uses is_validation=true (even for prefill, num_validated_tokens=0);
    // running with is_validation=false forks target sequences and breaks subsequent validation passes.
    invoke_sampler(logits, input_token_count, num_candidates, num_tokens_to_validate, /*is_validation=*/true);

    // Extract all tokens appended by the sampler (new_token for prefill; acc_1..acc_k + bonus for validation).
    const auto& gen_after = running_sequences.at(0)->get_generated_ids();
    OPENVINO_ASSERT(gen_after.size() > result_start,
                    "Sampler produced no new tokens: gen_after.size()=",
                    gen_after.size(),
                    ", result_start=",
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
    if (actual_seq_len == 0) {
        actual_seq_len = m_request.get_tensor("input_ids").get_shape().at(1);
    }
    return trim_tensor_tail(hidden_state, actual_seq_len);
}

ov::Tensor Eagle3InferWrapperBase::trim_tensor_tail(const ov::Tensor& tensor, size_t useful_len) {
    const auto& shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1,
                    "trim_tensor_tail expects shape [1, seq_len, dim], got: ",
                    shape);
    const size_t output_len = shape[1];
    if (output_len == useful_len) {
        return tensor;
    }

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
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config);

    OPENVINO_ASSERT(m_sequence_group->num_total_seqs() == 1,
                    "Expected single sequence after initialization, got ",
                    m_sequence_group->num_total_seqs());
}

InferenceOutput Eagle3TargetWrapper::infer(const InputTensors& inputs) {
    const auto& ids_shape = inputs.input_ids.get_shape();
    OPENVINO_ASSERT(ids_shape.size() == 2 && ids_shape[0] == 1,
                    "Expected input_ids shape [1, seq_len], got ",
                    ids_shape);
    const size_t input_len = ids_shape[1];

    m_request.set_tensor("input_ids", inputs.input_ids);
    m_request.set_tensor("attention_mask", inputs.attention_mask);
    m_request.set_tensor("position_ids", inputs.position_ids);
    m_request.set_tensor("eagle_tree_mask", inputs.eagle_tree_mask);

    const uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    auto result = InferenceOutput{get_logits(), get_hidden_features(input_len)};

    return result;
}

// Runs one target model pass: prefill (initial prompt) or validation (draft tree candidates).
// Dispatched by ctx.num_tokens_to_validate: 0 = prefill, >0 = tree validation.
// Hidden states are stored to the target sequence for the next DRAFT_INITIAL pass.
InferResult Eagle3TargetWrapper::forward(const InferContext& ctx) {
    const bool is_validation = ctx.num_tokens_to_validate > 0;
    const InputTensors inputs = is_validation ? build_validation_inputs() : build_prefill_inputs();

    auto output = infer(inputs);

    // Trim padded outputs: prefill uses last 1 position, validation uses all candidates.
    const size_t num_candidates = is_validation ? inputs.input_ids.get_shape().at(1) : 1;
    output.logits = trim_tensor_tail(output.logits, num_candidates);

    const std::vector<int64_t> sampled =
        sample_and_validate(output.logits, ctx.input_token_count, num_candidates, ctx.num_tokens_to_validate);

    get_current_sequence()->update_hidden_state(output.hidden_features);

    return InferResult{std::move(output), std::move(sampled)};
}

// ---------------------------------------------------------------------------
// Eagle3DraftWrapper
// ---------------------------------------------------------------------------

Eagle3DraftWrapper::Eagle3DraftWrapper(const ov::genai::ModelDesc& model_desc) : Eagle3InferWrapperBase(model_desc) {
    // Draft `hidden_states` input receives the target's `last_hidden_state` output verbatim,
    // so the two precisions must match.
    m_hidden_input_type = m_request.get_compiled_model().input("hidden_states").get_element_type();
    OPENVINO_ASSERT(m_hidden_input_type == m_hidden_output_type,
                    "Draft `hidden_states` input type (",
                    m_hidden_input_type,
                    ") must match target `last_hidden_state` output type (",
                    m_hidden_output_type,
                    ")");
}

void Eagle3DraftWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto& shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len], got ", shape);

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t total_len = shape[1];
    OPENVINO_ASSERT(total_len >= 2, "Draft model requires at least 2 tokens");

    // Draft model uses tokens[1:] — Eagle draft pairs each hidden state h_i (from the
    // target's layer for position i) with the token at position i+1, so the input
    // sequence is advanced by one timestep relative to the prompt.
    const TokenIds draft_prompt_ids(ids_data + 1, ids_data + total_len);
    m_sequence_group = std::make_shared<SequenceGroup>(1, draft_prompt_ids, config);

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

    // `m_logits_gather_buf` holds raw logits; the hidden buffers feed the draft's
    // `hidden_states` input directly — allocate with the matching model precisions.
    m_logits_gather_buf = ov::Tensor(m_logits_type, {1, max_sequences, vocab_size});
    m_hidden_concat_buf = ov::Tensor(m_hidden_input_type, {1, max_sequences * max_depth, hidden_size});

    m_per_seq_hidden_bufs.resize(max_sequences);
    for (size_t i = 0; i < max_sequences; ++i) {
        m_per_seq_hidden_bufs[i] = ov::Tensor(m_hidden_input_type, {1, max_depth, hidden_size});
    }

    m_buffers_allocated = true;
}

InferenceOutput Eagle3DraftWrapper::infer(const InputTensors& inputs, const ov::Tensor& hidden_states) {
    const auto& ids_shape = inputs.input_ids.get_shape();
    OPENVINO_ASSERT(ids_shape.size() == 2 && ids_shape[0] == 1,
                    "Expected input_ids shape [1, seq_len], got ",
                    ids_shape);
    const size_t input_token_count = ids_shape[1];

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

    auto result = InferenceOutput{get_logits(), get_hidden_features(input_token_count)};

    return result;
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
    const size_t hidden_size = running_sequences.at(0)->get_hidden_state().get_shape().at(2);
    ov::Tensor concat_buf;
    if (m_buffers_allocated && total_seq_len <= m_max_sequences * m_max_depth) {
        m_hidden_concat_buf.set_shape({1, total_seq_len, hidden_size});
        concat_buf = m_hidden_concat_buf;
    } else {
        concat_buf = ov::Tensor(m_hidden_input_type, {1, total_seq_len, hidden_size});
    }

    // Concatenate per-sequence hidden states into a contiguous tensor matching
    // the flat input_ids layout expected by the draft model.  Byte-based copy so
    // f32 / f16 both work; src/dst types match by construction.
    const size_t elem_size = concat_buf.get_element_type().size();
    const size_t row_bytes = hidden_size * elem_size;
    size_t offset = 0;
    for (size_t i = 0; i < num_sequences; ++i) {
        const auto seq_hidden = running_sequences[i]->get_hidden_state();
        const size_t seq_len = seq_hidden.get_shape().at(1);
        std::memcpy(static_cast<uint8_t*>(concat_buf.data()) + offset * row_bytes,
                    seq_hidden.data(),
                    seq_len * row_bytes);
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
        const size_t hidden_size = h_shape[2];
        const size_t row_bytes = hidden_size * root_hidden.get_element_type().size();

        for (size_t i = 0; i < num_sequences; ++i) {
            if (m_buffers_allocated && i < m_per_seq_hidden_bufs.size()) {
                // Use pre-allocated buffer: set shape to {1, 1, H} and copy root.
                m_per_seq_hidden_bufs[i].set_shape({1, 1, hidden_size});
                std::memcpy(m_per_seq_hidden_bufs[i].data(), root_hidden.data(), row_bytes);
                running_sequences[i]->update_hidden_state(m_per_seq_hidden_bufs[i]);
            } else {
                running_sequences[i]->update_hidden_state(root_hidden);
            }
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

    const size_t branch_len = running_sequences.at(0)->get_hidden_state().get_shape().at(1);
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
    const size_t row_bytes = hidden_size * output.hidden_features.get_element_type().size();

    for (size_t i = 0; i < num_sequences; ++i) {
        // The last branch token for sequence i sits at (i+1)*branch_len - 1.
        const size_t last_tok_pos = (i + 1) * branch_len - 1;

        if (m_buffers_allocated && i < m_per_seq_hidden_bufs.size() && new_len <= m_max_depth) {
            // Sequence::fork() shallow-copies m_hidden_state (ov::Tensor shares data).
            // A forked sequence's hidden may point to another buffer, so copy first.
            const auto existing = running_sequences[i]->get_hidden_state();
            m_per_seq_hidden_bufs[i].set_shape({1, new_len, hidden_size});
            if (existing.data() != m_per_seq_hidden_bufs[i].data()) {
                std::memcpy(m_per_seq_hidden_bufs[i].data(), existing.data(), branch_len * row_bytes);
            }
            copy_hidden_state_row(output.hidden_features, last_tok_pos, m_per_seq_hidden_bufs[i], branch_len);
            running_sequences[i]->update_hidden_state(m_per_seq_hidden_bufs[i]);
        } else {
            // Fallback: allocate new tensor.
            ov::Tensor updated(output.hidden_features.get_element_type(), {1, new_len, hidden_size});
            const auto existing = running_sequences[i]->get_hidden_state();
            std::memcpy(updated.data(), existing.data(), branch_len * row_bytes);
            copy_hidden_state_row(output.hidden_features, last_tok_pos, updated, branch_len);
            running_sequences[i]->update_hidden_state(updated);
        }
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
    const size_t new_branch_len = running_sequences.at(0)->get_hidden_state().get_shape().at(1);
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
        gather_buf = ov::Tensor(output.logits.get_element_type(), {1, num_sequences, vocab_size});
    }

    OPENVINO_ASSERT(gather_buf.get_element_type() == output.logits.get_element_type(),
                    "Logits gather buffer type mismatch");

    const size_t elem_size = gather_buf.get_element_type().size();
    const size_t row_bytes = vocab_size * elem_size;
    const auto* src = static_cast<const uint8_t*>(output.logits.data());
    auto* dst = static_cast<uint8_t*>(gather_buf.data());

    for (size_t s = 0; s < num_sequences; ++s) {
        // Each sequence occupies [s*prev_branch_len, (s+1)*prev_branch_len) in the flat layout;
        // take the last position of each sequence's branch.
        const size_t flat_pos = (s + 1) * prev_branch_len - 1;
        std::memcpy(dst + s * row_bytes, src + flat_pos * row_bytes, row_bytes);
    }

    return gather_buf;
}

// Runs one draft model pass: builds inputs, infers, updates hidden states, and samples.
//
// Two modes controlled by ctx.use_target_hidden:
//   DRAFT_INITIAL:   input_token_count tokens (= accepted_count + 1 from previous iteration),
//                    hidden state sourced from the target model.
//   DRAFT_ITERATION: flat batch of all beam paths (1 token per sequence),
//                    hidden state from accumulated draft buffers.
//
// Hidden states are updated BEFORE sampling so that sequence forks (from tree search)
// inherit the correct state.
InferResult Eagle3DraftWrapper::forward(const InferContext& ctx) {
    const InputTensors inputs = ctx.use_target_hidden ? build_initial_inputs(ctx.input_token_count)
                                                      : build_iteration_inputs(ctx.past_accepted_token_count);

    const ov::Tensor hidden_states = prepare_hidden_states(ctx);
    auto output = infer(inputs, hidden_states);

    // Trim padded output to useful positions.
    const size_t useful_logits_len = ctx.use_target_hidden ? 1 : inputs.input_ids.get_shape().at(1);
    output.logits = trim_tensor_tail(output.logits, useful_logits_len);

    update_hidden_states(output, ctx);

    const ov::Tensor logits = gather_logits_for_sampling(output, ctx);
    auto sampled = sample_next_tokens(logits, ctx.input_token_count);

    return InferResult{std::move(output), std::move(sampled)};
}

// ---------------------------------------------------------------------------
// StatefulEagle3LLMPipeline
// ---------------------------------------------------------------------------

StatefulEagle3LLMPipeline::StatefulEagle3LLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config) {
    // m_generation_config is initialised from target_model_desc (eos_token_id, etc.).
    // Tree search params are filled with defaults here; they may be overridden at
    // generate() time via the user-provided GenerationConfig.
    ensure_tree_params_is_set(m_generation_config);

    m_compile_config = build_compile_config(draft_model_desc);

    validate_construction_params(target_model_desc, draft_model_desc);
    apply_graph_transforms(target_model_desc, draft_model_desc);
    configure_and_create_models(target_model_desc, draft_model_desc);
}

Eagle3CompileConfig StatefulEagle3LLMPipeline::build_compile_config(const ModelDesc& draft_model_desc) {
    // Priority: explicit property > m_generation_config (already resolved by ensure_tree_params_is_set).
    const auto& props = draft_model_desc.properties;

    auto get_or = [&](const char* prop_key, size_t resolved_val) -> size_t {
        auto it = props.find(prop_key);
        if (it != props.end()) {
            return it->second.as<size_t>();
        }
        return resolved_val;
    };

    Eagle3CompileConfig cfg;
    cfg.max_tree_depth = get_or("MAX_TREE_DEPTH", m_generation_config.tree_depth);
    cfg.max_branching_factor = get_or("MAX_BRANCHING_FACTOR", m_generation_config.branching_factor);
    cfg.max_assistant_tokens = get_or("MAX_ASSISTANT_TOKENS", m_generation_config.num_assistant_tokens);
    return cfg;
}

void StatefulEagle3LLMPipeline::validate_construction_params(const ModelDesc& target_model_desc,
                                                             const ModelDesc& draft_model_desc) {
    OPENVINO_ASSERT(m_generation_config.is_tree_search(), "Eagle3 pipeline requires tree_depth > 0.");

    // Eagle3 stateful pipeline is designed for NPU devices. Non-NPU KV cache management
    // paths exist in the base class but are not validated for correctness in this pipeline.
    OPENVINO_ASSERT(target_model_desc.device == "NPU" && draft_model_desc.device == "NPU",
                    "Eagle3 pipeline supports NPU only. Target: " + target_model_desc.device +
                        ", Draft: " + draft_model_desc.device);

    OPENVINO_ASSERT(draft_model_desc.properties.find("hidden_layers_list") != draft_model_desc.properties.end(),
                    "hidden_layers_list must be present in draft model properties");
}

void StatefulEagle3LLMPipeline::apply_graph_transforms(const ModelDesc& target_model_desc,
                                                       const ModelDesc& draft_model_desc) {
    const auto& hidden_layers_to_abstract =
        draft_model_desc.properties.at("hidden_layers_list").as<std::vector<int32_t>>();
    OPENVINO_ASSERT(hidden_layers_to_abstract.size() == 3,
                    "hidden_layers_list must contain exactly 3 layers, got ",
                    hidden_layers_to_abstract.size());

    auto target_model = target_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(target_model, "Target model must not be null");
    OPENVINO_ASSERT(draft_model, "Draft model must not be null");

    utils::eagle3::share_vocabulary(target_model, draft_model);

    m_d2t_mapping = utils::eagle3::extract_d2t_mapping_table(draft_model);
    OPENVINO_ASSERT(m_d2t_mapping && m_d2t_mapping->get_element_type() == ov::element::i64,
                    "Invalid d2t mapping tensor");

    utils::eagle3::apply_eagle3_attention_mask_transform(draft_model);
    utils::eagle3::apply_eagle3_attention_mask_transform(target_model);

    utils::eagle3::transform_hidden_state(target_model, hidden_layers_to_abstract);
    utils::eagle3::move_fc_from_draft_to_main(draft_model, target_model);
    utils::eagle3::transform_hidden_state(draft_model, {-1});
}

// Configures NPU properties using m_compile_config shape limits and creates model wrappers.
//
// NPUW requires the maximum generation token length at compile time:
//   - Target: max_assistant_tokens + 1 (all tree candidates including root).
//   - Draft:  max_tree_depth * max_branching_factor (worst-case flat batch at deepest level).
void StatefulEagle3LLMPipeline::configure_and_create_models(const ModelDesc& target_model_desc,
                                                            const ModelDesc& draft_model_desc) {
    const size_t target_validation_window = m_compile_config.target_max_gen_tokens();
    const size_t draft_validation_window = m_compile_config.draft_max_gen_tokens();

    // --- Configure and create draft model ---
    auto draft_desc = draft_model_desc;
    if (draft_desc.device == "NPU") {
        draft_desc.properties["NPUW_EAGLE"] = "TRUE";
        draft_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = draft_validation_window;
        draft_desc.properties["NPUW_ONLINE_PIPELINE"] = "NONE";
    }
    m_draft = std::make_unique<Eagle3DraftWrapper>(draft_desc);
    m_draft->set_draft_target_mapping(m_d2t_mapping);

    // --- Configure and create target model ---
    auto target_desc = target_model_desc;
    if (target_desc.device == "NPU") {
        target_desc.properties["NPUW_EAGLE"] = "TRUE";
        target_desc.properties["NPUW_LLM_MAX_GENERATION_TOKEN_LEN"] = target_validation_window;
        target_desc.properties["NPUW_SLICE_OUT"] = "NO";
    }
    m_target = std::make_unique<Eagle3TargetWrapper>(target_desc);
}

StatefulEagle3LLMPipeline::~StatefulEagle3LLMPipeline() {
    m_target->release_memory();
    m_draft->release_memory();
}

void StatefulEagle3LLMPipeline::ensure_tree_params_is_set(GenerationConfig& config) {
    if (config.is_tree_search()) {
        OPENVINO_ASSERT(config.branching_factor > 0, "branching_factor must be > 0 for Eagle3 tree search");
        OPENVINO_ASSERT(config.num_assistant_tokens > 0, "num_assistant_tokens must be > 0 for Eagle3 tree search");
        return;
    }

    config.branching_factor = DEFAULT_EAGLE_BRANCHING_FACTOR;
    config.tree_depth = DEFAULT_EAGLE_TREE_DEPTH;
    config.num_assistant_tokens = DEFAULT_EAGLE_NUM_ASSISTANT_TOKENS;
}

GenerationConfig StatefulEagle3LLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(std::move(generation_config));
    OPENVINO_ASSERT(!config.do_sample, "Eagle3 speculative decoding requires greedy sampling (do_sample=false)");
    ensure_tree_params_is_set(config);

    // Validate runtime params do not exceed NPU compile-time shape limits.
    OPENVINO_ASSERT(config.tree_depth <= m_compile_config.max_tree_depth,
                    "tree_depth (",
                    config.tree_depth,
                    ") exceeds compile-time limit (",
                    m_compile_config.max_tree_depth,
                    "). Set MAX_TREE_DEPTH in draft_model properties to increase.");
    OPENVINO_ASSERT(config.branching_factor <= m_compile_config.max_branching_factor,
                    "branching_factor (",
                    config.branching_factor,
                    ") exceeds compile-time limit (",
                    m_compile_config.max_branching_factor,
                    "). Set MAX_BRANCHING_FACTOR in draft_model properties to increase.");
    OPENVINO_ASSERT(config.num_assistant_tokens <= m_compile_config.max_assistant_tokens,
                    "num_assistant_tokens (",
                    config.num_assistant_tokens,
                    ") exceeds compile-time limit (",
                    m_compile_config.max_assistant_tokens,
                    "). Set MAX_ASSISTANT_TOKENS in draft_model properties to increase.");

    return config;
}

int64_t StatefulEagle3LLMPipeline::run_prefill() {
    InferContext prefill_ctx;
    prefill_ctx.input_token_count = m_prompt_length;
    const auto prefill_result = m_target->forward(prefill_ctx);
    OPENVINO_ASSERT(prefill_result.sampled_tokens.size() == 1, "Expected single token from prefill");
    const int64_t initial_token = prefill_result.sampled_tokens[0];

    m_draft->append_tokens({initial_token});

    // Allocate draft reuse buffers once, deferred to here because hidden_size and vocab_size
    // are only known after the first target inference.
    if (!m_draft->buffers_allocated()) {
        const auto& h_shape = prefill_result.output.hidden_features.get_shape();
        OPENVINO_ASSERT(h_shape.size() == 3, "hidden_features must be rank-3, got rank ", h_shape.size());
        const size_t hidden_size = h_shape[2];

        const auto& l_shape = prefill_result.output.logits.get_shape();
        OPENVINO_ASSERT(l_shape.size() == 3, "logits must be rank-3, got rank ", l_shape.size());
        const size_t vocab_size = l_shape[2];

        m_draft->allocate_buffers(m_compile_config.max_branching_factor,
                                  m_compile_config.max_tree_depth,
                                  hidden_size,
                                  vocab_size);

        // Pre-allocate accepted hidden state buffer for gather_accepted_hidden_states().
        // Sized to compile-time max; precision follows the target's `last_hidden_state` output.
        const size_t max_accepted = m_compile_config.max_assistant_tokens + 1;
        m_accepted_hidden_buf = ov::Tensor(m_target->hidden_output_type(), {1, max_accepted, hidden_size});
    }

    return initial_token;
}

EncodedResults StatefulEagle3LLMPipeline::build_results(ManualTimer& generate_timer,
                                                        size_t generated_tokens,
                                                        size_t total_draft_accepted,
                                                        size_t total_draft_generated) {
    EncodedResults results;
    auto tokens = m_target->get_generated_tokens();
    if (tokens.size() > generated_tokens) {
        tokens.resize(generated_tokens);
    }
    results.tokens = {std::move(tokens)};
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

    // Normalize batch_sizes so their sum matches the actual (capped) output length.
    auto target_raw = m_target->get_raw_perf_metrics();
    if (!target_raw.m_batch_sizes.empty()) {
        const size_t recorded =
            std::accumulate(target_raw.m_batch_sizes.begin(), target_raw.m_batch_sizes.end(), size_t{0});
        if (recorded > generated_tokens) {
            target_raw.m_batch_sizes.back() -= recorded - generated_tokens;
        }
    }
    m_sd_perf_metrics.main_model_metrics.raw_metrics = std::move(target_raw);
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

EncodedResults StatefulEagle3LLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulEagle3LLMPipeline::generate(EncodedInputs)");
    generate_timer.start();

    const auto streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    // Extract input tensors.
    ov::Tensor input_ids;
    if (const auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
    } else if (const auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
    }

    OPENVINO_ASSERT(input_ids.get_shape().at(0) == 1, "Only batch size 1 supported");
    m_prompt_length = input_ids.get_shape().at(1);

    // Reset model states.
    m_target->reset_state();
    m_draft->reset_state();

    m_draft->initialize_sequence(input_ids, config);
    m_target->initialize_sequence(input_ids, config);

    // --- Phase 1: Initial Prompt Processing (Prefill) ---
    const int64_t initial_token = run_prefill();
    auto streaming_status = stream_generated_tokens(streamer_ptr, {initial_token});

    // --- Phase 2: Speculative Decoding Loop ---
    // Use the runtime config's tree_depth for draft iterations to stay in sync with
    // the TreeSearcher inside the sampler (which reads tree_depth from the SequenceGroup's config).
    // m_max_draft_depth (from constructor) is only used for NPU buffer sizing.
    const size_t draft_iterations = config.tree_depth;
    size_t generated_tokens = 1;
    size_t total_draft_accepted = 0;
    size_t total_draft_generated = 0;
    bool eos_reached = false;

    size_t input_token_count = m_draft->get_sequence_length();

    while (!eos_reached && generated_tokens < config.max_new_tokens &&
           streaming_status == ov::genai::StreamingStatus::RUNNING) {
        auto result = run_speculative_iteration(input_token_count, config.stop_token_ids, draft_iterations);

        // Truncate validated tokens if they would exceed max_new_tokens.
        const size_t remaining_budget = config.max_new_tokens - generated_tokens;
        if (result.validated_tokens.size() > remaining_budget) {
            result.validated_tokens.resize(remaining_budget);
            result.accepted_tokens_count = remaining_budget - 1;
            result.eos_reached = true;  // Force stop after this iteration.
        }

        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        // Update statistics.
        total_draft_generated += result.num_draft_tokens;
        total_draft_accepted += result.accepted_tokens_count;
        generated_tokens += result.validated_tokens.size();
        eos_reached = result.eos_reached;

        input_token_count = result.next_window_size;
    }

    // --- Phase 3: Finalization ---
    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr)
        streamer_ptr->end();

    return build_results(generate_timer, generated_tokens, total_draft_accepted, total_draft_generated);
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

void StatefulEagle3LLMPipeline::expand_draft_tree(size_t past_accepted_token_count, size_t draft_iterations) {
    for (size_t i = 1; i < draft_iterations; ++i) {
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
    //   TreeMetaData.tree_mask  = (N+1) x (N+1), index 0 = root.
    // The root token is the LAST token sampled by the target in the previous iteration.
    // It has NOT been fed back to the target KV cache yet, so the target must process
    // all N+1 candidates (root + N tree nodes) during validation.
    const auto& draft_gen_ids = m_draft->get_current_sequence()->get_generated_ids();
    const auto& draft_metadata = m_draft->get_current_sequence()->get_tree_metadata();
    const size_t num_candidates = draft_metadata.tree_mask.size();
    OPENVINO_ASSERT(num_candidates >= 2,
                    "Expected at least 2 candidates (root + at least 1 tree node), got: ",
                    num_candidates);
    const size_t num_tree_nodes = num_candidates - 1;

    // Sync TreeMetaData and candidate tokens from DRAFT -> TARGET.
    m_target->get_current_sequence()->set_tree_metadata(draft_metadata);

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
    // Note: currently unreachable — constructor asserts NPU-only (validate_construction_params).
    // Kept for future multi-device support.
    if (m_target->device() == "NPU") {
        const auto& validated_indices = m_target->get_current_sequence()->get_tree_metadata().validated_indices;
        m_target->set_npu_sampling_result(validation.num_candidates, validated_indices);
    } else if (tokens_to_remove > 0) {
        m_target->trim_kv_cache(tokens_to_remove);
        m_draft->trim_kv_cache(tokens_to_remove);
    }
}

// Gathers hidden states for the accepted tree path from the target model output.
//
// The target output contains hidden states for ALL candidates {1, N+1, H},
// but only a non-contiguous subset was accepted (e.g. indices [0, 2, 5]).
// Extract those rows into {1, accepted_count, H} for the next DRAFT_INITIAL pass.
void StatefulEagle3LLMPipeline::gather_accepted_hidden_states(const ValidationResult& validation) {
    const auto& current_hidden = validation.output.hidden_features;
    OPENVINO_ASSERT(current_hidden && current_hidden.get_size() > 0, "Missing hidden features");

    const auto& h_shape = current_hidden.get_shape();
    OPENVINO_ASSERT(h_shape.size() == 3 && h_shape[0] == 1, "Invalid hidden state shape: ", h_shape);

    const auto& validated_indices = m_target->get_current_sequence()->get_tree_metadata().validated_indices;
    const size_t total_accepted_tokens = validation.validated_tokens.size();
    OPENVINO_ASSERT(validated_indices.size() == total_accepted_tokens,
                    "validated_indices size (",
                    validated_indices.size(),
                    ") != total_accepted_tokens (",
                    total_accepted_tokens,
                    ")");

    const size_t hidden_size = h_shape[2];

    // Reuse pre-allocated buffer to avoid per-iteration heap allocation.
    // Maximum tokens is bounded by num_assistant_tokens + 1.
    OPENVINO_ASSERT(m_accepted_hidden_buf.get_element_type() == current_hidden.get_element_type(),
                    "Accepted hidden buffer type mismatch");
    m_accepted_hidden_buf.set_shape({1, total_accepted_tokens, hidden_size});
    const size_t elem_size = current_hidden.get_element_type().size();
    const size_t row_bytes = hidden_size * elem_size;
    const auto* src = static_cast<const uint8_t*>(current_hidden.data());
    auto* dst = static_cast<uint8_t*>(m_accepted_hidden_buf.data());

    for (size_t i = 0; i < total_accepted_tokens; ++i) {
        const size_t row = validated_indices[i];
        OPENVINO_ASSERT(row < h_shape[1],
                        "validated_indices[",
                        i,
                        "] = ",
                        row,
                        " is out of range [0, ",
                        h_shape[1],
                        ")");
        std::memcpy(dst + i * row_bytes, src + row * row_bytes, row_bytes);
    }
    m_target->get_current_sequence()->update_hidden_state(m_accepted_hidden_buf);
}

SpeculativeResult StatefulEagle3LLMPipeline::run_speculative_iteration(size_t input_token_count,
                                                                       const std::set<int64_t>& stop_token_ids,
                                                                       size_t draft_iterations) {
    OPENVINO_ASSERT(m_target->get_sequence_group() && m_draft->get_sequence_group(),
                    "Eagle3 speculative iteration requires initialized sequence groups");

    const auto target_hidden = m_target->get_current_sequence()->get_hidden_state();
    OPENVINO_ASSERT(target_hidden && target_hidden.get_size() > 0,
                    "Target model contains invalid hidden state for speculation");

    const size_t pre_draft_token_len = m_draft->get_sequence_length();
    const size_t past_accepted_token_count = m_target->get_current_sequence()->get_generated_ids().size();

    // Clear stale TreeMetaData from the previous iteration.
    m_draft->get_current_sequence()->set_tree_metadata({});

    // Step 1: Generate first draft token using target hidden states (DRAFT_INITIAL).
    generate_initial_draft(input_token_count, past_accepted_token_count);

    // Step 2: Expand the draft tree for (draft_iterations - 1) more levels.
    expand_draft_tree(past_accepted_token_count, draft_iterations);

    // Step 3: Validate draft candidates with the target model.
    auto validation = validate_draft_with_target();

    // Step 4: Synchronize sequences and KV caches.
    synchronize_after_validation(validation, pre_draft_token_len);

    // Step 5: Gather accepted hidden states for next iteration.
    gather_accepted_hidden_states(validation);

    const int64_t target_predicted_token = validation.validated_tokens.back();

    SpeculativeResult result;
    result.accepted_tokens_count = validation.accepted_count;
    result.num_draft_tokens = validation.num_candidates - 1;  // tree nodes excluding root
    result.next_window_size = validation.accepted_count + 1;
    result.validated_tokens = std::move(validation.validated_tokens);
    result.eos_reached = is_stop_token_id_hit(target_predicted_token, stop_token_ids);

    return result;
}

SpeculativeDecodingMetrics StatefulEagle3LLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

}  // namespace ov::genai
