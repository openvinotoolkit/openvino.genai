// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "openvino/core/except.hpp"

namespace ov::genai {

/**
* @brief Represents the mode of per-token score aggregation when determining least important tokens for eviction
*        from cache
*/
enum class AggregationMode {
    SUM,     /**< In this mode the importance scores of each token will be summed after each step of generation */
    NORM_SUM /**< Same as SUM, but the importance scores are additionally divided by the lifetime (in tokens generated)
                * of a given token in cache */
};

/**
* @brief Configuration struct for the cache eviction algorithm.
*/
class CacheEvictionConfig {
public:
    CacheEvictionConfig() = default;

    CacheEvictionConfig(size_t start_size, size_t recent_size, size_t max_cache_size, AggregationMode aggregation_mode_, bool apply_rotation_ = false, size_t snapkv_window_size_ = 8) : aggregation_mode(aggregation_mode_), apply_rotation(apply_rotation_), snapkv_window_size(snapkv_window_size_), m_start_size(start_size), m_recent_size(recent_size), m_max_cache_size(max_cache_size) {
        OPENVINO_ASSERT(start_size, "CacheEvictionConfig.start_size must be non-zero");
        OPENVINO_ASSERT(recent_size, "CacheEvictionConfig.recent_size must be non-zero");
        OPENVINO_ASSERT(max_cache_size, "CacheEvictionConfig.max_cache_size must be non-zero");
        OPENVINO_ASSERT(snapkv_window_size, "CacheEvictionConfig.snapkv_window_size must be non-zero");

        OPENVINO_ASSERT(max_cache_size > (start_size + recent_size),
                        "CacheEvictionConfig.max_cache_size must be larger than CacheEvictionConfig.start_size + CacheEvictionConfig.recent_size");
        m_evictable_size = m_max_cache_size - m_start_size - m_recent_size;

    }

    /** @return Number of tokens between the "start" and "recent" areas of KV cache that
     * will be considered for eviction. */
    std::size_t get_start_size() const {
        return m_start_size;
    }

    /** @return Number of tokens between the "start" and "recent" areas of KV cache that
     * will be considered for eviction. */
    std::size_t get_recent_size() const {
        return m_recent_size;
    }

    /** @return Number of tokens between the "start" and "recent" areas of KV cache that
     * will be considered for eviction. */
    std::size_t get_max_cache_size() const {
        return m_max_cache_size;
    }

    /** @return Number of tokens between the "start" and "recent" areas of KV cache that
     * will be considered for eviction. */
    std::size_t get_evictable_size() const {
        return m_evictable_size;
    }

    /** The mode used to compute the importance of tokens for eviction */
    AggregationMode aggregation_mode = AggregationMode::NORM_SUM;

    /** Whether to apply cache rotation (RoPE-based) after each eviction.
     *  Set this to false if your model has different RoPE scheme from the one used in the
     *  original llama model and you experience accuracy issues with cache eviction enabled
     *  and apply_rotation=true.**/
    bool apply_rotation = false;

    /** The size of the importance score aggregation window (in token positions from the end of the prompt) for
     * computing initial importance scores at the beginning of the generation phase for purposes of eviction,
     * following the SnapKV article approach (https://arxiv.org/abs/2404.14469). **/
    size_t snapkv_window_size = 8;

private:
    /** Number of tokens in the *beginning* of KV cache that should be retained
     * in the KV cache for this sequence during generation. Must be non-zero and a multiple of the KV cache block size for
     * this pipeline.*/
    std::size_t m_start_size = 32;

    /** Number of tokens in the *end* of KV cache that should be retained
     * in the KV cache for this sequence during generation. Must be non-zero and a multiple of the KV cache block size for
     * this pipeline.*/
    std::size_t m_recent_size = 128;

    /**
     * Maximum cache size (in tokens) that can be occupied by a sequence with cache eviction enabled.
     * Actual occupied size may differ from this by no larger than (block_size) tokens.
     * Eviction area is computed from this size and the "start"/"recent" area sizes.
     */
    std::size_t m_max_cache_size = 672;
    std::size_t m_evictable_size = 512;

};

} // namespace ov::genai

