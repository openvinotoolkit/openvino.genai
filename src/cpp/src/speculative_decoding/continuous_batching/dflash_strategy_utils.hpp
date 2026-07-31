// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <openvino/core/except.hpp>
#include <openvino/runtime/tensor.hpp>

#include "openvino/genai/generation_config.hpp"

namespace ov::genai::dflash_cb {

inline constexpr size_t DEFAULT_NUM_ASSISTANT_TOKENS = 5;

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
    if (config.num_assistant_tokens == 0) {
        config.num_assistant_tokens = DEFAULT_NUM_ASSISTANT_TOKENS;
    }
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

inline ov::Tensor build_draft_input_ids(int64_t seed_token, int64_t mask_token_id, size_t candidate_count) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    const size_t draft_input_length = candidate_count + 1;
    ov::Tensor input_ids(ov::element::i64, {1, draft_input_length});
    auto* data = input_ids.data<int64_t>();
    data[0] = seed_token;
    std::fill(data + 1, data + draft_input_length, mask_token_id);
    return input_ids;
}

inline ov::Tensor build_draft_position_ids(size_t committed_context_length,
                                           size_t hidden_delta_length,
                                           size_t candidate_count) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    ov::Tensor position_ids(ov::element::i64, {1, hidden_delta_length + candidate_count + 1});
    auto* data = position_ids.data<int64_t>();
    std::iota(data, data + position_ids.get_size(), static_cast<int64_t>(committed_context_length));
    return position_ids;
}

inline ov::Tensor build_draft_attention_mask(size_t committed_context_length,
                                             size_t hidden_delta_length,
                                             size_t candidate_count) {
    OPENVINO_ASSERT(candidate_count > 0, "DFlash candidate_count must be greater than 0.");
    const size_t attention_mask_length = committed_context_length + hidden_delta_length + candidate_count + 1;
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

inline size_t linear_attention_checkpoint_slot_for_validation(const ValidationAccounting& accounting,
                                                              bool validation_input_includes_seed_token) {
    OPENVINO_ASSERT(accounting.target_extended,
                    "Cannot select a linear attention checkpoint when target did not extend the sequence.");
    return accounting.accepted + (validation_input_includes_seed_token ? 1 : 0);
}

}  // namespace ov::genai::dflash_cb
