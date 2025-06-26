// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>

#include "continuous_batching/attention_output.hpp"
#include "openvino/genai/cache_eviction.hpp"
#include "sequence_group.hpp"

namespace ov::genai {

/**
 * @brief Calculates the set of KV cache logical block IDs that should be skipped from the KV cache block set during the
 * next inference for a given sequence group, in a tri-shape (https://arxiv.org/pdf/2412.10319) fashion
 */
class TriShapeSparseAttentionTokenSkipper {
public:
    TriShapeSparseAttentionTokenSkipper() = delete;

    /**
     * Constructs the TriShapeSparseAttentionTokenSkipper.
     * @param block_size Block size in tokens.
     * @param num_last_dense_tokens_in_prefill The number of tokens in the end of the prompt phase for which sparse attention
     * should not be applied.
     * @param num_retained_start_tokens_in_cache The number of tokens in the beginning of the cache (least recent)
     * to be retained when applying sparse attention. Must be a multiple of block size.
     * @param num_retained_recent_tokens_in_cache The number of most recent tokens in cache to be retained when applying
     * sparse attention. Must be a multiple of block size.
     */
    explicit TriShapeSparseAttentionTokenSkipper(
                                         size_t block_size,
                                         size_t num_last_dense_tokens_in_prefill,
                                         size_t num_retained_start_tokens_in_cache,
                                         size_t num_retained_recent_tokens_in_cache)
        : m_block_size(block_size),
          m_num_last_dense_tokens_in_prefill(num_last_dense_tokens_in_prefill),
          m_num_retained_start_tokens_in_cache(num_retained_start_tokens_in_cache),
          m_num_retained_recent_tokens_in_cache(num_retained_recent_tokens_in_cache) {
            OPENVINO_ASSERT(!(num_retained_start_tokens_in_cache % block_size),
                            "num_last_dense_tokens_in_prefill in tokens must be a multiple of block size ", block_size);
            OPENVINO_ASSERT(!(num_retained_recent_tokens_in_cache % block_size),
                            "num_retained_dense_tokens_in_prefill in tokens must be a multiple of block size ", block_size);
          }

    /**
     * @param sequence_group A pointer to the sequence group.
     * @return The set of logical block IDs that should be skipped during the next inference for this sequence group.
     */
    std::set<size_t> get_skipped_blocks(const SequenceGroup::CPtr& sequence_group) const;

private:
    size_t m_block_size;
    size_t m_num_last_dense_tokens_in_prefill;
    size_t m_num_retained_start_tokens_in_cache;
    size_t m_num_retained_recent_tokens_in_cache;
};

}  // namespace ov::genai
