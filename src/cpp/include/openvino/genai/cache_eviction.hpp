// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <unordered_map>
#include <sstream>

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
 * @brief Represents the mode of how anchor points are formed in KVCrush Cache eviction algorithm
 */
enum class KVCrushAnchorPointMode {
    RANDOM, /**<In this mode the anchor point is a random binary vector of 0s and 1s > */
    ZEROS,  /**<In this mode the anchor point is a vector of 0s */
    ONES,   /**<In this mode the anchor point is a vector of 1s */
    MEAN, /**<In this mode the anchor point is a random binary vector of 0s and 1s, where individual values are decided
             based on majority value */
    ALTERNATE /**In this mode the anchor point is a vector of alternate 0s and 1s */
};

class KVCrushConfig {
public:
    /**
     * @brief Configuration struct for the KVCrush cache eviction algorithm.
     */
    /**
     * @class KVCrushConfig
     * @brief Configuration class for KVCrush cache mechanism.
     *
     * This class encapsulates the configuration parameters for the KVCrush cache,
     * including cache budget, anchor point mode, and random seed.
     */

    KVCrushConfig() = default;

    /**
     * @brief Constructs a KVCrushConfig with the specified parameters.
     * @param budget_ The cache budget, representing the number of blocks to store.
     * @param anchor_point_mode_ The anchor point mode for KVCrush (see KVCrushAnchorPointMode).
     * @param rng_seed_ Optional random seed for reproducibility (default is 0).
     */

    KVCrushConfig(size_t budget_, KVCrushAnchorPointMode anchor_point_mode_, size_t rng_seed_ = 0)
        : budget(budget_),
          anchor_point_mode(anchor_point_mode_),
          rng_seed(rng_seed_) {}

    /*KVCrush Cache budget - number of blocks*/
    std::size_t budget = 0;
    /*KVCrush Anchor point mode*/
    KVCrushAnchorPointMode anchor_point_mode = KVCrushAnchorPointMode::RANDOM;
    size_t rng_seed = 0;
    std::size_t get_budget() const {
        return budget;
    }

    std::string to_string() const {
        static const std::unordered_map<KVCrushAnchorPointMode, std::string> kv_crush_anchor_point_mode_to_string = {
            {KVCrushAnchorPointMode::RANDOM, "RANDOM"},
            {KVCrushAnchorPointMode::ZEROS, "ZEROS"},
            {KVCrushAnchorPointMode::ONES, "ONES"},
            {KVCrushAnchorPointMode::MEAN, "MEAN"},
            {KVCrushAnchorPointMode::ALTERNATE, "ALTERNATE"},
        };

        std::ostringstream oss;
        oss << "KVCrushConfig { " << "\n";
        oss << "  budget: " << budget << "\n";
        oss << "  rng_seed: " << rng_seed << "\n";
        if (kv_crush_anchor_point_mode_to_string.count(anchor_point_mode) > 0) {
            oss << "  anchor_point_mode: " << kv_crush_anchor_point_mode_to_string.at(anchor_point_mode) << "\n";
        }
        oss << " }";
        return oss.str();
    }
};

/**
* @brief Configuration struct for the cache eviction algorithm.
*/
class CacheEvictionConfig {
public:
    CacheEvictionConfig() = default;

    CacheEvictionConfig(size_t start_size,
                        size_t recent_size,
                        size_t max_cache_size,
                        AggregationMode aggregation_mode_,
                        bool apply_rotation_ = false,
                        size_t snapkv_window_size_ = 8,
                        const KVCrushConfig& kvcrush_config_ = KVCrushConfig(0, KVCrushAnchorPointMode::RANDOM))
        : aggregation_mode(aggregation_mode_),
          apply_rotation(apply_rotation_),
          snapkv_window_size(snapkv_window_size_),
          kvcrush_config(kvcrush_config_),
          m_start_size(start_size),
          m_recent_size(recent_size),
          m_max_cache_size(max_cache_size) {
        OPENVINO_ASSERT(start_size, "CacheEvictionConfig.start_size must be non-zero");
        OPENVINO_ASSERT(recent_size, "CacheEvictionConfig.recent_size must be non-zero");
        OPENVINO_ASSERT(max_cache_size, "CacheEvictionConfig.max_cache_size must be non-zero");

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

    std::string to_string() const {
        static const std::unordered_map<AggregationMode, std::string> aggregation_mode_to_string = {
            {AggregationMode::SUM, "SUM"},
            {AggregationMode::NORM_SUM, "NORM_SUM"},
        };

        std::ostringstream oss;
        oss << "CacheEvictionConfig { " << "\n";
        oss << "  start_size: " << m_start_size << "\n";
        oss << "  recent_size: " << m_recent_size << "\n";
        oss << "  max_cache_size: " << m_max_cache_size << "\n";
        oss << "  evictable_size: " << m_evictable_size << "\n";
        if (aggregation_mode_to_string.count(aggregation_mode) > 0) {
            oss << "  aggregation_mode: " << aggregation_mode_to_string.at(aggregation_mode) << "\n";
        }
        oss << "  apply_rotation: " << std::boolalpha << apply_rotation << "\n";
        oss << "  snapkv_window_size: " << snapkv_window_size << "\n";
        oss << kvcrush_config.to_string() << "\n";
        oss << " }";
        return oss.str();
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
     * following the SnapKV article approach (https://arxiv.org/abs/2404.14469). Setting this to 0 disables the SnapKV
     * score aggregation. **/
    size_t snapkv_window_size = 8;

    /** KVCrush configuration for this cache eviction algorithm.
     * KVCrush is an additional mechanism that allows to retain some tokens in the cache
     * even if they are not among the most important ones.*/
    KVCrushConfig kvcrush_config;

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

