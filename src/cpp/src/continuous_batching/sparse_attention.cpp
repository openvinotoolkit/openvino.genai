// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <vector>
#include <cstdlib>
#include <cmath>

#include "continuous_batching/sparse_attention.hpp"

namespace ov::genai {
std::set<size_t> SparseAttentionTokenSkipper::get_skipped_blocks(const SequenceGroup::CPtr& sequence_group) const {
    std::set<size_t> skipped_logical_block_ids;
    size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
    size_t num_processed_tokens_after_this_chunk = sequence_group->get_num_processed_tokens() + num_scheduled_tokens;
    size_t prompt_len = sequence_group->get_prompt_len();
    OPENVINO_ASSERT(prompt_len >= num_processed_tokens_after_this_chunk);

    size_t num_remaining_prompt_tokens = prompt_len - num_processed_tokens_after_this_chunk;
    if (num_remaining_prompt_tokens >= m_num_last_dense_tokens) {
        size_t num_cached_tokens = sequence_group->get_num_cached_tokens();
        size_t num_cached_full_logical_blocks = num_cached_tokens / sequence_group->get_block_size();
        if (num_cached_full_logical_blocks > 2) {
            // A-shape phase
            for (size_t i = 0; i < num_cached_full_logical_blocks - 2; i++) {
                skipped_logical_block_ids.insert(i + 1);
            }
        }
    }
    // else skip nothing, dense attention phase
    return skipped_logical_block_ids;
}
}
