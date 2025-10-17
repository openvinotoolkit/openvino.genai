// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <unordered_map>
#include <sstream>

namespace ov::genai {

enum class SparseAttentionMode {
    TRISHAPE, /** Sparse attention will be applied to prefill stage only, with a configurable number of start and recent
               *  cache tokens to be retained. A number of prefill tokens in the end of the prompt can be configured to
               *  have dense attention applied to them instead, to retain generation accuracy.
               *  https://arxiv.org/pdf/2412.10319 */
    XATTENTION /** Following https://arxiv.org/pdf/2503.16428, introduces
                *  importance score threshold-based block sparsity into the prefill stage.
                *  Computing importance scores introduces an overhead, but the total inference
                *  time is expected to be reduced even more. */
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
                          size_t num_retained_recent_tokens_in_cache_,
                          float xattention_threshold_,
                          size_t xattention_block_size_,
                          size_t xattention_stride_)
        : mode(mode_),
          num_last_dense_tokens_in_prefill(num_last_dense_tokens_in_prefill_),
          num_retained_start_tokens_in_cache(num_retained_start_tokens_in_cache_),
          num_retained_recent_tokens_in_cache(num_retained_recent_tokens_in_cache_),
          xattention_threshold(xattention_threshold_),
          xattention_block_size(xattention_block_size_),
          xattention_stride(xattention_stride_) {}

    /**  Sparse attention mode to be applied. */
    SparseAttentionMode mode = SparseAttentionMode::TRISHAPE;

    /** TRISHAPE and XATTENTION modes - Number of tokens from the end of the prompt for which full attention across previous KV
     * cache contents will be computed. In contrast, for the rest of the tokens in the prompt only the sparse attention
     * will be computed according to the selected algorithm.
     * TRISHAPE: Due to the block-wise nature of continuous batching cache management, the actual number of prompt tokens
     * for which the dense attention will be computed may be up to block-size larger than this value (depending on the
     * prompt length and block size).
     * XATTENTION: Same as above applies, but the dense attention may overspill up to a subsequence chunk (i.e. multiple
     * blocks)
     * */
    size_t num_last_dense_tokens_in_prefill = 100;

    /** TRISHAPE mode only - The number of tokens in the beginning of the cache (least recent) to be retained when
     * applying sparse attention. Must be a multiple of block size. */
    size_t num_retained_start_tokens_in_cache = 128;

    /** TRISHAPE mode only - The number of most recent tokens in cache to be retained when
     * applying sparse attention. Must be a multiple of block size. */
    size_t num_retained_recent_tokens_in_cache = 1920;

    /** XATTENTION mode only - Cumulative importance score threshold to be compared against when determining blocks to
     * exclude from the attention calculations in the block-sparse approach. Only the attention matrix blocks with
     * highest importance score sum not exceeding this threshold will be taking part in the computations. The lower the
     * threshold, the less computation will the main attention operation will take, and vice versa, with the
     * corresponding potential impact on generation accuracy. */
    float xattention_threshold = 0.8;

    /** XATTENTION mode only - Block granularity, in tokens, with which the block-sparse attention calculation will be
     * applied.*/
    size_t xattention_block_size = 64;

    /** XATTENTION mode only - The stride of antidiagonal sampling employed to calculate the importance scores of each
     * `xattention_block_size`-sized block of the attention matrix before the actual attention calculation takes place.
     *  Directly influences the overhead portion of the importance score computations - if full (dense) attention takes
     *  M time to be calculated, then the importance score calculation would be taking `M / xattention_stride` time as
     *  overhead. */
    size_t xattention_stride = 8;

    /**
     * @brief Returns a string representation of the SparseAttentionConfig.
     *
     * The returned string contains the values of all configuration fields in a human-readable format, e.g.:
     * SparseAttentionConfig {
     *   sparseAttentionMode: TRISHAPE
     *   num_last_dense_tokens_in_prefill: 100
     *   num_retained_start_tokens_in_cache: 128
     *   num_retained_recent_tokens_in_cache: 1920
     *   xattention_threshold: 0.8
     *   xattention_block_size: 64
     *   xattention_stride: 8
     * }
     *
     * @return A string describing the current configuration.
     */
    std::string to_string() const {
        static const std::unordered_map<SparseAttentionMode, std::string> sparse_attention_mode_to_string = {
            {SparseAttentionMode::TRISHAPE, "TRISHAPE"},
            {SparseAttentionMode::XATTENTION, "XATTENTION"},
        };
        std::ostringstream oss;
        oss << "SparseAttentionConfig { " << "\n";
        if (sparse_attention_mode_to_string.count(mode) > 0) {
            oss << "  sparseAttentionMode: " << sparse_attention_mode_to_string.at(mode) << "\n";
        }
        oss << "  num_last_dense_tokens_in_prefill: " << num_last_dense_tokens_in_prefill << "\n";
        oss << "  num_retained_start_tokens_in_cache: " << num_retained_start_tokens_in_cache << "\n";
        oss << "  num_retained_recent_tokens_in_cache: " << num_retained_recent_tokens_in_cache << "\n";
        oss << "  xattention_threshold: " << xattention_threshold << "\n";
        oss << "  xattention_block_size: " << xattention_block_size << "\n";
        oss << "  xattention_stride: " << xattention_stride << "\n";
        oss << " }";
        return oss.str();
    }
};

}  // namespace ov::genai
