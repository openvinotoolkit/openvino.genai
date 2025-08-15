// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/kvcrush.hpp"

#include <random>
namespace ov::genai {

KVCrushAlgorithm::KVCrushAlgorithm(const KVCrushConfig& kvcrush_config, size_t block_size)
    : m_kvcrush_config(kvcrush_config),
      m_block_size(block_size),
      rng(std::mt19937(kvcrush_config.rng_seed)) {}

// step 1: create_indicators_kvcrush()
std::vector<size_t> KVCrushAlgorithm::create_indicators_kvcrush(size_t num_tokens_in_evictable_blocks,

                                                                std::vector<size_t>& evicted_block_indices,
                                                                const std::vector<double>& layer_scores) {
    // Step 1: Sort the scores of the blocks to be evicted
    const auto& blocks_eligible_for_kvcrush = evicted_block_indices;
    std::vector<size_t> indices(num_tokens_in_evictable_blocks);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(),
                      indices.begin() + num_tokens_in_evictable_blocks / 2,
                      indices.end(),
                      [&](size_t i, size_t j) {
                          return layer_scores[i] > layer_scores[j];
                      });

    std::vector<size_t> indicators(num_tokens_in_evictable_blocks, 0);
    for (size_t i = 0; i < num_tokens_in_evictable_blocks / 2; ++i) {
        indicators[indices[i]] = 1;
    }
    return indicators;
}
// step 2: create_anchor_point_kvcrush()
std::vector<size_t> KVCrushAlgorithm::create_anchor_point_kvcrush(size_t num_tokens_in_evictable_blocks,

                                                                  std::vector<size_t>& indicators) {
    // Step 2: Create a binary vector of size block_size as anchor point
    std::vector<size_t> anchor_point(m_block_size);
    // Initialize anchor_point based on anchor using switch-case
    switch (m_kvcrush_config.anchor_point_mode) {
    case KVCrushAnchorPointMode::RANDOM: {
        std::uniform_int_distribution<int> dist(0, 1);
        std::generate(anchor_point.begin(), anchor_point.end(), [&]() {
            return dist(rng);
        });
    } break;
    case KVCrushAnchorPointMode::ZEROS:
        std::fill(anchor_point.begin(), anchor_point.end(), 0);
        break;
    case KVCrushAnchorPointMode::ONES:
        std::fill(anchor_point.begin(), anchor_point.end(), 1);
        break;
    case KVCrushAnchorPointMode::MEAN: {
        size_t num_blocks = num_tokens_in_evictable_blocks / m_block_size;
        for (size_t pos = 0; pos < m_block_size; pos++) {
            // Calculate sum of indicators at this position across all blocks
            size_t sum = 0;
            for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
                size_t idx = block_idx * m_block_size + pos;
                sum += indicators[idx];
            }

            // Calculate mean and set anchor point based on threshold (0.5)
            double mean = static_cast<double>(sum) / num_blocks;
            anchor_point[pos] = (mean > 0.5) ? 1 : 0;
        }
        break;
    }
    case KVCrushAnchorPointMode::ALTERNATE:
        for (size_t i = 0; i < m_block_size; ++i) {
            anchor_point[i] = i % 2;
        }
        break;
    default:
        OPENVINO_THROW("Invalid anchor point type");
    }
    return anchor_point;
}

// step 3: calculate_hamming_distance()
std::vector<std::pair<size_t, size_t>> KVCrushAlgorithm::calculate_hamming_distance_kvcrush(
    size_t num_tokens_in_evictable_blocks,

    std::vector<size_t>& indicators,
    std::vector<size_t>& anchor_point) {
    // Step 3: Calculate Hamming distances between anchor point and each block
    size_t num_blocks = num_tokens_in_evictable_blocks / m_block_size;
    std::vector<std::pair<size_t, size_t>> block_distances;  // pair<hamming_distance, block_idx>
    block_distances.reserve(num_blocks);

    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        size_t hamming_distance = 0;
        for (size_t j = 0; j < m_block_size; ++j) {
            size_t token_idx = block_idx * m_block_size + j;
            if (token_idx < num_tokens_in_evictable_blocks) {
                // Use the indicators vector to determine the bit value of this position
                int bit_value = indicators[token_idx];
                if (bit_value != anchor_point[j]) {
                    hamming_distance++;
                }
            }
        }
        block_distances.emplace_back(hamming_distance, block_idx);
    }
    return block_distances;
}

// step 4: get_representative_blocks()
std::vector<std::size_t> KVCrushAlgorithm::get_representative_blocks_kvcrush(

    size_t num_tokens_in_evictable_blocks,
    std::vector<std::pair<size_t, size_t>>& block_distances,
    const std::vector<size_t>& blocks_eligible_for_kvcrush) {
    // Step 4: Find the representative blocks
    // Filter block indices that are in blocks_eligible_for_kvcrush
    std::vector<size_t> filtered_block_indices;
    filtered_block_indices.reserve(block_distances.size());

    for (const auto& entry : block_distances) {
        size_t block_idx = entry.second;
        // Check if block_idx is in blocks_eligible_for_kvcrush
        if (std::find(blocks_eligible_for_kvcrush.begin(), blocks_eligible_for_kvcrush.end(), block_idx) !=
            blocks_eligible_for_kvcrush.end()) {
            filtered_block_indices.push_back(block_idx);
        }
    }
    // Sort filtered_block_indices based on Hamming distance
    std::sort(filtered_block_indices.begin(), filtered_block_indices.end(), [&](size_t a, size_t b) {
        return block_distances[a].first < block_distances[b].first;
    });
    // select kvcrush_budget number of blocks from filtered_block_indices, uniformly spaced
    size_t num_blocks_to_retain = std::min(filtered_block_indices.size(), m_kvcrush_config.get_budget());
    size_t step = filtered_block_indices.size() / num_blocks_to_retain;
    std::vector<std::size_t> kvcrush_retained_block_indices;
    kvcrush_retained_block_indices.reserve(num_blocks_to_retain);
    for (size_t i = 0; i < num_blocks_to_retain; ++i) {
        size_t idx = i * step;
        if (idx < filtered_block_indices.size()) {
            kvcrush_retained_block_indices.push_back(filtered_block_indices[idx]);
        }
    }

    return kvcrush_retained_block_indices;
}

std::vector<std::size_t> KVCrushAlgorithm::get_indices_of_blocks_to_retain_using_kvcrush(

    size_t num_tokens_in_evictable_blocks,
    std::vector<std::size_t>& evicted_block_indices,
    const std::vector<double>& layer_scores) {
    // step 1: Create indicators_kvcrush makes binary feature vectors based on top-k/2 scores
    const auto& blocks_eligible_for_kvcrush = evicted_block_indices;  // only the blocks that are evicted by the score
                                                                      // based eviction are eligible for kvcrush

    std::vector<size_t> indicators =
        create_indicators_kvcrush(num_tokens_in_evictable_blocks, evicted_block_indices, layer_scores);

    // Step 2: Create anchor_point based on the selected anchor point type
    std::vector<size_t> anchor_point = create_anchor_point_kvcrush(num_tokens_in_evictable_blocks, indicators);

    // Step 3: Calculate Hamming distances between anchor point and each block, where each block is represented by
    // its binary feature vector called indicators
    std::vector<std::pair<size_t, size_t>> block_distances =
        calculate_hamming_distance_kvcrush(num_tokens_in_evictable_blocks, indicators, anchor_point);

    // Step 4: Find the representative blocks
    // Filter block indices that are in blocks_eligible_for_kvcrush
    return get_representative_blocks_kvcrush(num_tokens_in_evictable_blocks,
                                             block_distances,
                                             blocks_eligible_for_kvcrush);
}

}  // namespace ov::genai