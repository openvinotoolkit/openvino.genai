// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ov::genai {

enum class SparseAttentionMode { TRISHAPE };

const std::unordered_map<SparseAttentionMode, std::string> SparseAttentionModeToString = {
    {SparseAttentionMode::TRISHAPE, "TRISHAPE"},
};

/**
 * @brief Configuration struct for the sparse attention prefill functionality.
 */
class SparseAttentionConfig {
public:
    SparseAttentionConfig() = default;

    SparseAttentionConfig(SparseAttentionMode mode_,
                          size_t num_last_dense_tokens_in_prefill_,
                          size_t num_retained_start_tokens_in_cache_,
                          size_t num_retained_recent_tokens_in_cache_)
        : mode(mode_),
          num_last_dense_tokens_in_prefill(num_last_dense_tokens_in_prefill_),
          num_retained_start_tokens_in_cache(num_retained_start_tokens_in_cache_),
          num_retained_recent_tokens_in_cache(num_retained_recent_tokens_in_cache_) {}

    /**  Sparse attention mode to be applied. */
    SparseAttentionMode mode;

    /** Number of tokens from the end of the prompt for which full attention across previous KV cache contents
     * will be computed. In contrast, for the rest of the tokens in the prompt only the sparse attention (encompassing
     * a configurable number of least-recent and most-recent blocks) will be computed.
     * Due to the block-wise nature of continuous batching cache management, the actual number of prompt tokens for
     * which the dense attention will be computed may be up to block-size larger than this value (depending on the
     * prompt length and block size).*/
    size_t num_last_dense_tokens_in_prefill = 100;

    /** The number of tokens in the beginning of the cache (least recent) to be retained when applying sparse attention.
     * Must be a multiple of block size. */
    size_t num_retained_start_tokens_in_cache = 128;

    /** @param num_retained_recent_tokens_in_cache The number of most recent tokens in cache to be retained when
     * applying sparse attention. Must be a multiple of block size. */
    size_t num_retained_recent_tokens_in_cache = 1920;

    void print() const {
        std::cout << "SparseAttentionConfig { " << std::endl;
        if (SparseAttentionModeToString.count(mode) > 0) {
            std::cout << "  sparseAttentionMode: " << SparseAttentionModeToString.at(mode) << std::endl;
        }
        std::cout << "  num_last_dense_tokens_in_prefill: " << num_last_dense_tokens_in_prefill << std::endl;
        std::cout << "  num_retained_start_tokens_in_cache: " << num_retained_start_tokens_in_cache << std::endl;
        std::cout << "  num_retained_recent_tokens_in_cache: " << num_retained_recent_tokens_in_cache << std::endl;
        std::cout << " }" << std::endl;
    }
};
}  // namespace ov::genai
