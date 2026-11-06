// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/runtime/tensor.hpp>

#include "openvino/genai/generation_config.hpp"

namespace ov::genai::dflash_cb {

inline constexpr size_t DEFAULT_NUM_ASSISTANT_TOKENS = 5;

inline void add_draft_assistant_token_headroom(GenerationConfig& draft_config, const GenerationConfig& config) {
    const size_t assistant_tokens = config.num_assistant_tokens.value();
    if (config.max_new_tokens != SIZE_MAX) {
        if (config.max_new_tokens < SIZE_MAX - assistant_tokens) {
            draft_config.max_new_tokens = config.max_new_tokens + assistant_tokens;
        }
    } else if (config.max_length != SIZE_MAX && config.max_length < SIZE_MAX - assistant_tokens) {
        draft_config.max_length = config.max_length + assistant_tokens;
    }
}

inline void copy_tensor_bytes(const ov::Tensor& src, ov::Tensor& dst) {
    OPENVINO_ASSERT(src.get_element_type() == dst.get_element_type(),
                    "DFlash hidden state copy requires matching tensor element types.");
    OPENVINO_ASSERT(src.get_byte_size() == dst.get_byte_size(),
                    "DFlash hidden state copy requires matching tensor byte sizes.");
    src.copy_to(dst);
}

inline ov::Tensor truncate_normalized_hidden_state_from_end(const ov::Tensor& hidden_state, size_t tokens_to_remove) {
    if (!hidden_state || hidden_state.get_size() == 0 || tokens_to_remove == 0) {
        return hidden_state;
    }

    auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 1,
                    "DFlash hidden_states delta must have shape [seq_len, 1, hidden].");
    const size_t current_seq_len = shape[0];
    if (tokens_to_remove >= current_seq_len) {
        shape[0] = 0;
        return ov::Tensor(hidden_state.get_element_type(), shape);
    }

    ov::Coordinate start_coord(shape.size(), 0);
    ov::Coordinate end_coord(shape.begin(), shape.end());
    end_coord[0] = current_seq_len - tokens_to_remove;
    return ov::Tensor(hidden_state, start_coord, end_coord);
}

class HiddenDeltaBuffer {
public:
    static constexpr size_t INITIAL_CHUNK_CAPACITY = 10;

    HiddenDeltaBuffer() {
        m_chunks.reserve(INITIAL_CHUNK_CAPACITY);
    }

    void append(const ov::Tensor& hidden_delta, bool copy_data = false) {
        if (!hidden_delta || hidden_delta.get_size() == 0) {
            return;
        }
        const auto shape = hidden_delta.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 1,
                        "DFlash hidden delta buffer expects [seq_len, 1, hidden] chunks.");
        const size_t token_count = shape[0];
        if (token_count == 0) {
            return;
        }

        if (copy_data) {
            ov::Tensor owned(hidden_delta.get_element_type(), shape);
            copy_tensor_bytes(hidden_delta, owned);
            m_chunks.push_back(owned);
        } else {
            m_chunks.push_back(hidden_delta);
        }
        m_token_count += token_count;
    }

    bool empty() const {
        return m_token_count == 0;
    }

    size_t token_count() const {
        return m_token_count;
    }

    ov::Tensor materialize() const {
        OPENVINO_ASSERT(m_token_count > 0, "Cannot materialize empty DFlash hidden deltas.");
        OPENVINO_ASSERT(!m_chunks.empty(), "DFlash hidden delta chunks are empty.");

        if (m_chunks.size() == 1) {
            const auto& chunk = m_chunks.front();
            OPENVINO_ASSERT(chunk && chunk.get_size() > 0,
                            "DFlash single hidden delta chunk is empty.");
            return chunk;
        }

        auto merged_shape = m_chunks.front().get_shape();
        OPENVINO_ASSERT(merged_shape.size() == 3 && merged_shape[1] == 1,
                        "DFlash hidden delta buffer expects [seq_len, 1, hidden] chunks.");
        merged_shape[0] = m_token_count;
        ov::Tensor merged(m_chunks.front().get_element_type(), merged_shape);
        size_t offset = 0;
        for (const auto& chunk : m_chunks) {
            const auto chunk_shape = chunk.get_shape();
            OPENVINO_ASSERT(chunk_shape.size() == 3 && chunk_shape[1] == 1 && chunk_shape[2] == merged_shape[2],
                            "Cannot merge DFlash hidden deltas with incompatible shape.");
            const size_t chunk_tokens = chunk_shape[0];
            ov::Tensor dst(merged,
                           ov::Coordinate{offset, 0, 0},
                           ov::Coordinate{offset + chunk_tokens, 1, merged_shape[2]});
            copy_tensor_bytes(chunk, dst);
            offset += chunk_tokens;
        }
        OPENVINO_ASSERT(offset == m_token_count, "DFlash hidden delta token count mismatch.");
        return merged;
    }

    void clear() {
        m_chunks.clear();
        m_token_count = 0;
    }

private:
    std::vector<ov::Tensor> m_chunks;
    size_t m_token_count = 0;
};

inline void ensure_num_assistant_tokens_is_set(GenerationConfig& config) {
    OPENVINO_ASSERT(config.assistant_confidence_threshold == 0.f,
                    "DFlash CB/PA only supports num_assistant_tokens; assistant_confidence_threshold must be 0.f.");
    OPENVINO_ASSERT(config.max_ngram_size == 0,
                    "DFlash CB/PA does not support prompt lookup decoding; max_ngram_size must be 0.");
    if (!config.num_assistant_tokens.has_value() || config.num_assistant_tokens.value() == 0) {
        config.num_assistant_tokens = DEFAULT_NUM_ASSISTANT_TOKENS;
    }
}

inline void ensure_vlm_generation_config(const GenerationConfig& config) {
    // Keep generic DFlash decoding behavior unchanged. These checks cover only
    // VLM representation/output modes that cannot be expressed faithfully with
    // row-aligned multimodal prompt IDs.
    OPENVINO_ASSERT(config.pruning_ratio == 0,
                    "DFlash VLM does not support visual token pruning.");
    OPENVINO_ASSERT(!config.echo && config.logprobs == 0,
                    "DFlash VLM does not support echo or prompt logprobs for row-aligned multimodal prompt IDs.");
    OPENVINO_ASSERT(!config.return_omni_outputs,
                    "DFlash VLM does not support Omni intermediate outputs.");
}

inline std::vector<int64_t> build_placeholder_prompt_ids(size_t prompt_length, int64_t placeholder_id) {
    OPENVINO_ASSERT(prompt_length > 0, "DFlash VLM logical prompt length must be greater than 0.");
    OPENVINO_ASSERT(placeholder_id >= 0, "DFlash VLM requires a valid placeholder token ID.");
    return std::vector<int64_t>(prompt_length, placeholder_id);
}

inline size_t linear_attention_checkpoint_block_count(size_t num_assistant_tokens) {
    OPENVINO_ASSERT(num_assistant_tokens <= std::numeric_limits<size_t>::max() - 2,
                    "DFlash num_assistant_tokens is too large for linear attention checkpoint block count.");
    return num_assistant_tokens + 2;
}

inline size_t adjusted_linear_attention_block_count(size_t current_block_count,
                                                    size_t num_assistant_tokens,
                                                    bool target_has_linear_attention) {
    if (!target_has_linear_attention) {
        return current_block_count;
    }
    return std::max(current_block_count, linear_attention_checkpoint_block_count(num_assistant_tokens));
}

inline ov::Tensor build_draft_input_ids(int64_t seed_token,
                                        int64_t mask_token_id,
                                        size_t candidate_count,
                                        size_t candidate_position_offset = 1) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    const size_t draft_input_length = candidate_count + candidate_position_offset;
    ov::Tensor input_ids(ov::element::i64, {1, draft_input_length});
    auto* data = input_ids.data<int64_t>();
    data[0] = seed_token;
    std::fill(data + 1, data + draft_input_length, mask_token_id);
    return input_ids;
}

inline ov::Tensor build_draft_position_ids(size_t committed_context_length,
                                           size_t hidden_delta_length,
                                           size_t candidate_count,
                                           size_t candidate_position_offset = 1) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    ov::Tensor position_ids(ov::element::i64,
                            {1, hidden_delta_length + candidate_count + candidate_position_offset});
    auto* data = position_ids.data<int64_t>();
    std::iota(data, data + position_ids.get_size(), static_cast<int64_t>(committed_context_length));
    return position_ids;
}

inline ov::Tensor build_draft_attention_mask(size_t committed_context_length,
                                             size_t hidden_delta_length,
                                             size_t candidate_count,
                                             size_t candidate_position_offset = 1) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    const size_t attention_mask_length =
        committed_context_length + hidden_delta_length + candidate_count + candidate_position_offset;
    ov::Tensor attention_mask(ov::element::i64, {1, attention_mask_length});
    std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), 1);
    return attention_mask;
}

inline size_t draft_candidate_count(size_t num_assistant_tokens, size_t generated_len, size_t max_new_tokens) {
    OPENVINO_ASSERT(num_assistant_tokens > 0, "DFlash num_assistant_tokens must be greater than 0.");
    if (generated_len >= max_new_tokens) {
        return 0;
    }
    return num_assistant_tokens;
}

inline size_t validation_candidate_count(size_t draft_count, size_t generated_len, size_t max_new_tokens) {
    if (generated_len >= max_new_tokens) {
        return 0;
    }
    const size_t remaining = max_new_tokens - generated_len;
    if (remaining <= 1) {
        return 0;
    }
    return std::min(draft_count, remaining - 1);
}

// The candidate scorer takes the rows of num_blocks draft blocks packed back to back: block n spans rows
// subsequence_begins[n] to subsequence_begins[n + 1]. The graph marks the first row of every block and gives it the
// block's anchor as predecessor, so an empty block would mark a row twice and shift the anchors of later blocks.
inline void validate_subsequence_begins(const ov::Tensor& subsequence_begins, size_t num_blocks, size_t num_rows) {
    OPENVINO_ASSERT(num_blocks > 0, "The candidate scorer needs at least one draft block.");
    OPENVINO_ASSERT(subsequence_begins && subsequence_begins.get_element_type() == ov::element::i64 &&
                        subsequence_begins.get_shape() == ov::Shape{num_blocks + 1},
                    "The candidate scorer subsequence_begins must be an I64 tensor of shape [",
                    num_blocks + 1,
                    "].");
    const auto* begins = subsequence_begins.data<const int64_t>();
    OPENVINO_ASSERT(begins[0] == 0 && begins[num_blocks] == static_cast<int64_t>(num_rows),
                    "The candidate scorer subsequence_begins must start at 0 and end at the number of draft rows ",
                    num_rows,
                    ".");
    for (size_t block = 0; block < num_blocks; ++block) {
        OPENVINO_ASSERT(begins[block] < begins[block + 1],
                        "The candidate scorer subsequence_begins must be strictly increasing: draft block ",
                        block,
                        " is empty.");
    }
}

// The candidate scorer computes a Top-K lattice for one draft block of S positions:
//   candidate_ids [S, K]: the backbone's K best tokens at every position;
//   edge_scores [S, K, K]: edge_scores[s, i, j] scores candidate j at position s after candidate i at position s - 1.
// All rows at position 0 follow the anchor token and are identical.
inline void validate_candidate_lattice(const ov::Tensor& edge_scores,
                                       const ov::Tensor& candidate_ids,
                                       size_t num_positions,
                                       size_t vocab_size) {
    OPENVINO_ASSERT(candidate_ids && candidate_ids.get_element_type() == ov::element::i64 &&
                        candidate_ids.get_shape().size() == 2 && candidate_ids.get_shape()[0] == num_positions &&
                        candidate_ids.get_shape()[1] > 0,
                    "Candidate scorer candidate_ids must be an I64 tensor of shape [",
                    num_positions,
                    ", K].");
    const size_t top_k = candidate_ids.get_shape()[1];
    OPENVINO_ASSERT(edge_scores && edge_scores.get_element_type() == ov::element::f32 &&
                        edge_scores.get_shape() == ov::Shape({num_positions, top_k, top_k}),
                    "Candidate scorer edge_scores must be an FP32 tensor of shape [",
                    num_positions,
                    ", ",
                    top_k,
                    ", ",
                    top_k,
                    "].");
    const auto* ids = candidate_ids.data<const int64_t>();
    OPENVINO_ASSERT(std::all_of(ids,
                                ids + candidate_ids.get_size(),
                                [&](int64_t id) {
                                    return id >= 0 && static_cast<size_t>(id) < vocab_size;
                                }),
                    "Candidate scorer candidate_ids must be draft vocabulary ids.");
}

// True when the logit processors of a draft sampler with this config cannot change which token has the highest logit,
// so a greedy draft can take the best candidate of every row without the sampler. Follows the transforms that
// LogitProcessor adds for greedy decoding; the log-prob transforms keep the argmax.
inline bool logit_processors_keep_argmax(const GenerationConfig& config) {
    return config.is_greedy_decoding() && config.repetition_penalty == 1.0f && config.presence_penalty == 0.0f &&
           config.frequency_penalty == 0.0f && config.min_new_tokens == 0 && !config.is_structured_output_generation();
}

// Walks a validated lattice greedily without a sampler: at every position, the best candidate in the row of the
// candidate drafted before it. Ties go to the smaller token id, as in the greedy sampler's scan over the vocabulary,
// so the walk drafts the same tokens as walk_candidate_lattice() with a greedy sampler and no logit processors.
inline std::vector<int64_t> greedy_candidate_path(const ov::Tensor& edge_scores, const ov::Tensor& candidate_ids) {
    const size_t num_positions = candidate_ids.get_shape()[0];
    const size_t top_k = candidate_ids.get_shape()[1];
    const auto* scores = edge_scores.data<const float>();
    std::vector<int64_t> path;
    path.reserve(num_positions);
    size_t previous_index = 0;
    for (size_t position = 0; position < num_positions; ++position) {
        const auto* row = scores + (position * top_k + previous_index) * top_k;
        const auto* ids = candidate_ids.data<const int64_t>() + position * top_k;
        size_t best = 0;
        for (size_t j = 1; j < top_k; ++j) {
            if (row[j] > row[best] || (row[j] == row[best] && ids[j] < ids[best])) {
                best = j;
            }
        }
        previous_index = best;
        path.push_back(ids[best]);
    }
    return path;
}

// Tokens outside the candidates are shifted at least this far below the worst candidate.
inline constexpr float CANDIDATE_ROW_GAP = 1.0e4f;

// Spreads the lattice row that follows candidate previous_index at position - 1 over the vocabulary, so the request's
// logit processors and sampler pick the token as they do from a full logits row. Candidate j gets
// edge_scores[position, previous_index, j]. Every other token keeps its backbone logit, shifted below the candidates
// by at least CANDIDATE_ROW_GAP: a candidate still wins after the usual penalties unless the processors mask all K of
// them (e.g. structured output), and then the token comes from the remaining backbone logits. A repetition_penalty
// far below 1 can lift a non-candidate above the candidates, as it would in the token-by-token Markov walk, and the
// walk then ends the block at that token.
inline void fill_candidate_row(const float* backbone_logits,
                               size_t vocab_size,
                               const ov::Tensor& edge_scores,
                               const ov::Tensor& candidate_ids,
                               size_t position,
                               size_t previous_index,
                               float* row) {
    const size_t top_k = candidate_ids.get_shape()[1];
    const auto* ids = candidate_ids.data<const int64_t>() + position * top_k;
    const auto* scores = edge_scores.data<const float>() + (position * top_k + previous_index) * top_k;
    // The candidates are the backbone's Top-K at this position, so its best logit is among them.
    float best_logit = -std::numeric_limits<float>::infinity();
    float worst_score = std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < top_k; ++j) {
        best_logit = std::max(best_logit, backbone_logits[ids[j]]);
        worst_score = std::min(worst_score, scores[j]);
    }
    const float shift = worst_score - CANDIDATE_ROW_GAP - best_logit;
    std::transform(backbone_logits, backbone_logits + vocab_size, row, [shift](float logit) {
        return logit + shift;
    });
    for (size_t j = 0; j < top_k; ++j) {
        row[ids[j]] = scores[j];
    }
}

inline std::optional<size_t> find_candidate(const ov::Tensor& candidate_ids, size_t position, int64_t token_id) {
    const size_t top_k = candidate_ids.get_shape()[1];
    const auto* ids = candidate_ids.data<const int64_t>() + position * top_k;
    const auto* found = std::find(ids, ids + top_k, token_id);
    if (found == ids + top_k) {
        return std::nullopt;
    }
    return static_cast<size_t>(std::distance(ids, found));
}

// Walks a validated lattice through a sampler. At every position, sample_token(row) picks a token from the lattice row
// of the candidate drafted at the previous position, spread over the vocabulary in the [1, 1, vocab] row tensor by
// fill_candidate_row(). It returns the token, or nullopt to end the block. A token outside the position's candidates
// ends the block as well, since the lattice has no row for it.
// TODO: DFlash drafts are greedy-only until speculative sampling in the target sampler is fixed. With sampling, the
// sampler draws every position from this row, and the target also needs that distribution as the draft proposal of
// the token.
template <typename SampleToken>
void walk_candidate_lattice(const ov::Tensor& backbone_logits,
                            const ov::Tensor& edge_scores,
                            const ov::Tensor& candidate_ids,
                            size_t num_positions,
                            ov::Tensor& row,
                            SampleToken&& sample_token) {
    OPENVINO_ASSERT(backbone_logits && backbone_logits.get_element_type() == ov::element::f32 &&
                        backbone_logits.is_continuous(),
                    "DFlash draft logits must be a contiguous FP32 tensor.");
    const auto logits_shape = backbone_logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3 && logits_shape[0] == 1 && num_positions <= logits_shape[1],
                    "Candidate scorer positions exceed the DFlash draft logits [1, S, vocab].");
    const size_t vocab_size = logits_shape[2];
    OPENVINO_ASSERT(
        row && row.get_element_type() == ov::element::f32 && row.get_shape() == ov::Shape({1, 1, vocab_size}),
        "The candidate row must be an FP32 tensor of shape [1, 1, vocab].");
    const auto* logits = backbone_logits.data<const float>();
    size_t previous_index = 0;
    for (size_t position = 0; position < num_positions; ++position) {
        fill_candidate_row(logits + position * vocab_size,
                           vocab_size,
                           edge_scores,
                           candidate_ids,
                           position,
                           previous_index,
                           row.data<float>());
        const std::optional<int64_t> token = sample_token(row);
        if (!token) {
            return;
        }
        const auto index = find_candidate(candidate_ids, position, *token);
        if (!index) {
            return;
        }
        previous_index = *index;
    }
}

struct ValidationAccounting {
    size_t accepted = 0;
    size_t rejected = 0;
    bool target_extended = false;
};

inline ValidationAccounting validation_accounting(size_t draft_generated,
                                                  size_t generated_before_draft,
                                                  size_t target_generated_len) {
    if (draft_generated == 0 || target_generated_len <= generated_before_draft) {
        return {};
    }

    const size_t produced_by_target = target_generated_len - generated_before_draft;
    const size_t accepted = produced_by_target > 0 ? std::min(draft_generated, produced_by_target - 1) : 0;
    return {accepted, draft_generated - accepted, true};
}

}  // namespace ov::genai::dflash_cb
