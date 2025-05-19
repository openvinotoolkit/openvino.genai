// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>

#include "continuous_batching/attention_output.hpp"
#include "openvino/genai/cache_eviction.hpp"
#include "openvino/openvino.hpp"

namespace ov::genai {

class KVCrushAlgorithm {
public:
    KVCrushAlgorithm() =
        default;  // needed only to satisfy DefaultConstructible so that algo objects may be used as values in std::map
    explicit KVCrushAlgorithm(const KVCrushConfig& eviction_config);

    /** @return A vector of size kvcrush_budget, where each element contains the index of a
     * representative block selected using KVCrush algorithm */
    std::vector<std::size_t> get_indices_of_blocks_to_retain_using_kvcrush(
        size_t decoder_layer_idx,
        size_t block_size,
        size_t num_tokens_in_evictable_blocks,
        std::vector<std::size_t>& evicted_block_indices,
        const std::vector<std::vector<double>>& m_scores);
    /** @return A binary (feature) vector of size num_tokens_in_evictable_blocks, where each element indicates wheather
     * the corresponding token has a high score */
    std::vector<size_t> create_indicators_kvcrush(size_t decoder_layer_idx,
                                                  size_t num_tokens_in_evictable_blocks,
                                                  size_t block_size,
                                                  std::vector<size_t>& evicted_block_indices,
                                                  const std::vector<std::vector<double>>& m_scores);
    /** @return A binary vector of size block_size, where each individual element is selected based on the anchor point
     * mode */
    std::vector<size_t> create_anchor_point_kvcrush(size_t num_tokens_in_evictable_blocks,
                                                    size_t block_size,
                                                    std::vector<size_t>& indicators);
    /** @return A vector of size num_blocks, where each individual element (pair<hamming_distance, block_idx>)
     * represents it's hamming distance from the anchor point */
    std::vector<std::pair<size_t, size_t>> calculate_hamming_distance_kvcrush(size_t num_tokens_in_evictable_blocks,
                                                                              size_t block_size,
                                                                              std::vector<size_t>& indicators,
                                                                              std::vector<size_t>& anchor_point);
    /** @return A vector of size kvcrush_budget, where each element contains the index of a
     * representative block selected using KVCrush algorithm */
    std::vector<std::size_t> get_representative_blocks_kvcrush(size_t decoder_layer_idx,
                                                               size_t block_size,
                                                               size_t num_tokens_in_evictable_blocks,
                                                               std::vector<std::pair<size_t, size_t>>& block_distances,
                                                               const std::vector<size_t>& keep_clus_eligible);

    KVCrushConfig m_kvcrush_config;
};

}  // namespace ov::genai
