// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <vector>
#include <cstdlib>
#include <cmath>

#include "openvino/openvino.hpp"
#include "attention_output.hpp"
#include "openvino/genai/cache_eviction.hpp"

namespace ov::genai {

/**
 * @brief Determines blocks to be evicted from the KV cache of a sequence based on the importance score calculated from the
 * attention scores of each token at each attention layer in the LLM.
 *
 * The KV cache is conceptually divided into three areas as shown below:
 *
 * ```
 * --> *logical KV cache space in blocks*
 * | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
 * |<- start area->|<-   evictable area    ->|<- recent area ->|
 * ```
 *
 * The sizes of each areas are configurable. Once the sequence KV cache utilization is such that all three areas
 * are filled, the algorithm determines the blocks from the *evictable area* that should be freed from this sequence
 * based on the importance scores accumulated after each previous generation step in the pipeline. The least important
 * tokens according to this score are to be evicted. Only the tokens from the *evictable area* are evicted - the tokens
 * in the *start* and *recent* areas are never evicted, but throughout the eviction process the *recent* blocks naturally
 * move into the *evictable* area.
 *
 * Eviction only starts when at least one block *past* the *recent area* is completely filled, and the corresponding number
 * of blocks is selected to be evicted, so that the remaining blocks completely fit into the arena defined by the *start*,
 * *evictable* and *recent* areas. This effectively caps the cache usage for the sequence by the size of the arena (plus,
 * in general, one partially filled block past the recent area).
 *
 * Sizes of *start*, *evictable* and *recent* areas are configurable, but the *evictable* area size specifies the
 * _minimal_ size of the evictable area. When tokens overflow the eviction arena, the acutal evictable area is
 * determined as the tokens between the fixed-size *start area* and the fixed-size *end area*, so at a given eviction step
 * there are in general more tokens considered for eviction than the specified *evictable* size.
 *
 */
class CacheEvictionAlgorithm {
public:
    /**
     * @brief A pair of indices specifying the logical block interval where the blocks may be evicted at this point in time.
     */
    class CacheEvictionRange : public std::pair<std::size_t, std::size_t> {
    public:
        CacheEvictionRange(std::size_t begin, std::size_t end) : std::pair<std::size_t, std::size_t>(begin, end) {}
        static const CacheEvictionRange& invalid() {
            static CacheEvictionRange inv(0, 0);
            return inv;
        }
    };
    CacheEvictionAlgorithm() = default;  // needed only to satisfy DefaultConstructible so that algo objects may be used as values in std::map

    /**
     * Constructs a CacheEvictionAlgorithm.
     * @param eviction_config The configuration struct for this algorithm.
     * @param block_size Block size of the KV cache to evict from.
     * @param num_decoder_layers Number of independent KV caches (each corresponding to a single attention layer) in the underlying LLM.
     */
    explicit CacheEvictionAlgorithm(const CacheEvictionConfig& eviction_config, size_t block_size, size_t num_decoder_layers);

    /**
     * @return Maximum cache size (in tokens) after each eviction step. Could be used as an estimate of the maximum per-sequence cache usage.
     */
    std::size_t get_max_cache_size_after_eviction() const;

    /**
     * @return Current logical range of evictable block indices.
     */
    CacheEvictionRange get_evictable_block_range() const;

    /**
     * Registers attention scores (for each layer) of each token in this sequence that is currently still represented
     * (i.e. not evicted) in the corresponding KV cache. Must be called after each generation step to properly keep track of
     * the tokens' lifetime in the KV cache and of the accumulated importance score of each token.
     * @param attention_scores_for_all_decoder_layers A vector with a size equal to the configured num_decoder_layers, where each entry is a
     * vector of per-token attention scores calculated within this layer.
     */
    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers);

    /**
     * Returns the per-layer sets of logical block indices that should be evicted according to the internally computed importance scores
     * and removes the corresponding blocks from the internal algorithm tracking.
     *
     * @return A vector with size equal to the configured num_decoder_layers, where each entry is a set of logical indices that are to be
     * evicted by the external cache-controlling mechanism.
     */
    std::vector<std::set<std::size_t>> evict_logical_blocks();


private:
    std::size_t get_num_blocks(std::size_t num_tokens) const;
    std::size_t get_num_blocks_to_evict(size_t decoder_layer_idx) const;
    std::size_t get_num_evictable_blocks(size_t decoder_layer_idx) const;

    CacheEvictionRange get_evictable_block_range(size_t layer_idx) const;

    std::vector<double> get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) const;

    std::vector<std::size_t> get_indices_of_blocks_to_evict(const std::vector<double>& scores_for_each_evictable_block, size_t num_blocks_to_evict) const;

    void remove_scores_of_evicted_blocks(const std::vector<std::size_t>& evicted_block_indices, size_t decoder_layer_idx);

    CacheEvictionConfig m_eviction_config;
    std::size_t m_block_size;
    std::size_t m_num_evicted_tokens = 0;
    std::size_t m_num_decoder_layers;
    std::vector<std::vector<double>> m_scores;
    std::vector<std::vector<size_t>> m_cache_counter;
};

class CacheRotationCalculator {
public:
    CacheRotationCalculator(size_t block_size, size_t max_context_length, size_t kv_head_size, double rope_theta = 10000.0f) : m_block_size(block_size) {
        size_t max_position_angle_multiplier = max_context_length / 2 + 1; // adding +1 here and below for good measure in case of odd dividends
        size_t num_freqs = kv_head_size / 2 + 1;
        m_rope_sin_lut.reserve(max_position_angle_multiplier);
        m_rope_cos_lut.reserve(max_position_angle_multiplier);

        for (size_t i = 0; i < max_position_angle_multiplier; i++) {
            m_rope_sin_lut[i].reserve(num_freqs);
            m_rope_cos_lut[i].reserve(num_freqs);
            for (size_t j = 0; j < num_freqs; j++) {
                double exponent = - static_cast<double>(2 * j) / kv_head_size;
                double base_angle = std::pow(rope_theta,  exponent);
                m_rope_sin_lut[i].push_back(-std::sin(i * base_angle)); // minus since we will be rotating by an inverse angle
                m_rope_cos_lut[i].push_back(std::cos(i * base_angle));
            }
        }
    };

    using RotationCoefficientsPerToken = std::vector<std::vector<double>>;
    std::pair<RotationCoefficientsPerToken, RotationCoefficientsPerToken> get_rotation_multipliers(const std::set<size_t>& evicted_block_logical_indices, size_t num_logical_blocks_before_eviction) {
        std::pair<RotationCoefficientsPerToken, RotationCoefficientsPerToken> retval;
        if (evicted_block_logical_indices.empty()) {
            return retval;
        }

        ptrdiff_t current_rotation_delta_in_positions = 0;
        std::vector<size_t> logical_block_space(num_logical_blocks_before_eviction);
        std::iota(logical_block_space.begin(), logical_block_space.end(), 0);

        std::vector<ptrdiff_t> rotation_deltas;
        rotation_deltas.reserve(num_logical_blocks_before_eviction - evicted_block_logical_indices.size());

        for (size_t logical_block_idx : logical_block_space) {
            if (evicted_block_logical_indices.find(logical_block_idx) != evicted_block_logical_indices.end()) {
                current_rotation_delta_in_positions += 1;
            }
            else {
                if (current_rotation_delta_in_positions != 0) {
                    rotation_deltas.push_back(current_rotation_delta_in_positions);
                }
            }
        }

        size_t num_tokens_to_rotate = rotation_deltas.size() * m_block_size;
        retval.first.reserve(num_tokens_to_rotate);
        retval.second.reserve(num_tokens_to_rotate);
        for (ptrdiff_t delta : rotation_deltas) {
            for (size_t i = 0; i < m_block_size; i++) {
                retval.first.push_back(m_rope_cos_lut[delta]);
                retval.second.push_back(m_rope_sin_lut[delta]);
            }
        }

        return retval;
    }

private:
    size_t m_block_size;
    std::vector<std::vector<double>> m_rope_sin_lut;
    std::vector<std::vector<double>> m_rope_cos_lut;
};

}
