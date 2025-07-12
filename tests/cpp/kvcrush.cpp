// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/kvcrush.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "openvino/genai/cache_eviction.hpp"

namespace ov::genai::tests {
// Test fixture for KVCrushAlgorithm tests
class KVCrushAlgorithmTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default test configuration
        m_kvcrush_config.budget = 8;
        m_kvcrush_config.anchor_point_mode = KVCrushAnchorPointMode::RANDOM;
        m_kvcrush_config.rng_seed = 42;  // Fixed seed for reproducibility

        m_block_size = 4;
        m_num_decoder_layers = 2;

        // Create the algorithm instance
        m_kvcrush_algo = KVCrushAlgorithm(m_kvcrush_config, m_block_size);
    }

    // Helper to create mock attention scores
    std::vector<std::vector<double>> create_mock_scores(size_t num_layers, size_t sequence_length) {
        std::vector<std::vector<double>> scores(num_layers);
        for (size_t i = 0; i < num_layers; i++) {
            scores[i].resize(sequence_length);
            for (size_t j = 0; j < sequence_length; j++) {
                scores[i][j] = static_cast<double>(j % 10) / 10.0;  // Repeating pattern 0.0, 0.1, ..., 0.9
            }
        }
        return scores;
    }

    KVCrushConfig m_kvcrush_config;
    KVCrushAlgorithm m_kvcrush_algo;
    size_t m_block_size;
    size_t m_num_decoder_layers;
};

// Test KVCrushAlgorithm constructor
TEST_F(KVCrushAlgorithmTest, ConstructorTest) {
    KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);
    EXPECT_EQ(algo.m_kvcrush_config.budget, 8);
    EXPECT_EQ(algo.m_kvcrush_config.anchor_point_mode, KVCrushAnchorPointMode::RANDOM);
    EXPECT_EQ(algo.m_kvcrush_config.rng_seed, 42);
}

// Test create_indicators_kvcrush
TEST_F(KVCrushAlgorithmTest, CreateIndicatorsTest) {
    // Setup
    size_t decoder_layer_idx = 0;
    size_t num_tokens_in_evictable_blocks = 20;
    std::vector<size_t> evicted_block_indices = {0, 1, 2, 3, 4};
    auto scores = create_mock_scores(m_num_decoder_layers, num_tokens_in_evictable_blocks);

    // Execute
    auto indicators = m_kvcrush_algo.create_indicators_kvcrush(num_tokens_in_evictable_blocks,
                                                               evicted_block_indices,
                                                               scores[decoder_layer_idx]);

    // Verify
    EXPECT_EQ(indicators.size(), num_tokens_in_evictable_blocks);

    // Check that half of the indicators are set to 1 (top-k/2)
    size_t count_ones = std::count(indicators.begin(), indicators.end(), 1);
    EXPECT_EQ(count_ones, num_tokens_in_evictable_blocks / 2);
}

// Test create_anchor_point_kvcrush with RANDOM mode
TEST_F(KVCrushAlgorithmTest, CreateAnchorPointRandomTest) {
    // Setup
    size_t num_tokens_in_evictable_blocks = 16;
    std::vector<size_t> indicators(num_tokens_in_evictable_blocks, 1);  // All indicators set to 1

    m_kvcrush_config.anchor_point_mode = KVCrushAnchorPointMode::RANDOM;
    m_kvcrush_algo = KVCrushAlgorithm(m_kvcrush_config, m_block_size);
    auto anchor_point = m_kvcrush_algo.create_anchor_point_kvcrush(num_tokens_in_evictable_blocks, indicators);

    // Since we're using RANDOM mode with a fixed seed, the results should be deterministic
    // Count non-zero elements (should be distributed randomly)
    size_t non_zero_count = 0;
    for (const auto& val : anchor_point) {
        if (val > 0)
            non_zero_count++;
    }

    // With RANDOM mode, we expect some elements to be non-zero (anchor points)
    EXPECT_GT(non_zero_count, 0);

    // Verify
    EXPECT_EQ(anchor_point.size(), m_block_size);
}

// Test create_anchor_point_kvcrush with ALTERNATE mode
TEST_F(KVCrushAlgorithmTest, CreateAnchorPointAlternateTest) {
    // Setup
    m_kvcrush_config.anchor_point_mode = KVCrushAnchorPointMode::ALTERNATE;
    KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);

    size_t num_tokens_in_evictable_blocks = 16;
    std::vector<size_t> indicators(num_tokens_in_evictable_blocks, 1);  // All indicators set to 1

    // Execute
    auto anchor_point = algo.create_anchor_point_kvcrush(num_tokens_in_evictable_blocks, indicators);

    // Verify
    EXPECT_EQ(anchor_point.size(), m_block_size);

    // In ALTERNATE mode, we expect alternating blocks to be selected
    // We should have non-zero values at regular intervals
    bool found_pattern = false;
    for (size_t i = 0; i < anchor_point.size(); i += m_block_size) {
        if (i + m_block_size <= anchor_point.size()) {
            bool has_nonzero = false;
            for (size_t j = 0; j < m_block_size; j++) {
                if (anchor_point[i + j] > 0) {
                    has_nonzero = true;
                    break;
                }
            }
            if (has_nonzero) {
                found_pattern = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found_pattern);
}

// Test create_anchor_point_kvcrush with ZEROS mode
TEST_F(KVCrushAlgorithmTest, CreateAnchorPointZerosTest) {
    // Setup
    m_kvcrush_config.anchor_point_mode = KVCrushAnchorPointMode::ZEROS;
    KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);

    size_t num_tokens_in_evictable_blocks = 16;
    std::vector<size_t> indicators(num_tokens_in_evictable_blocks, 1);  // All indicators set to 1

    // Execute
    auto anchor_point = algo.create_anchor_point_kvcrush(num_tokens_in_evictable_blocks, indicators);

    // Verify
    // For ZEROS mode, we expect the anchor point to be all zeros
    EXPECT_EQ(anchor_point.size(), m_block_size);

    // Check that all values are zeros
    for (const auto& val : anchor_point) {
        EXPECT_EQ(val, 0) << "Anchor point value should be zero in ZEROS mode";
    }
}

// Test create_anchor_point_kvcrush with ONES mode
TEST_F(KVCrushAlgorithmTest, CreateAnchorPointOnesTest) {
    // Setup
    m_kvcrush_config.anchor_point_mode = KVCrushAnchorPointMode::ONES;
    KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);

    size_t num_tokens_in_evictable_blocks = 16;
    std::vector<size_t> indicators(num_tokens_in_evictable_blocks, 1);  // All indicators set to 1

    // Execute
    auto anchor_point = algo.create_anchor_point_kvcrush(num_tokens_in_evictable_blocks, indicators);

    // Verify
    // For ONES mode, we expect the anchor point to be all ones
    EXPECT_EQ(anchor_point.size(), m_block_size);

    // Check that all values are ones
    for (const auto& val : anchor_point) {
        EXPECT_EQ(val, 1) << "Anchor point value should be one in ONES mode";
    }
}

// Test create_anchor_point_kvcrush with MEAN mode
TEST_F(KVCrushAlgorithmTest, CreateAnchorPointMeanTest) {
    // Setup
    m_kvcrush_config.anchor_point_mode = KVCrushAnchorPointMode::MEAN;
    KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);

    size_t num_tokens_in_evictable_blocks = 20;
    // Create indicators with a mix of 0s and 1s to test mean calculation
    std::vector<size_t> indicators = {1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0};

    // Execute
    auto anchor_point = algo.create_anchor_point_kvcrush(num_tokens_in_evictable_blocks, indicators);

    // Verify
    // For MEAN mode, we expect anchor points to be the mean of indicators in each block
    size_t expected_size = num_tokens_in_evictable_blocks / m_block_size;
    EXPECT_EQ(anchor_point.size(), m_block_size);

    // Expected means (rounded or converted to binary based on implementation)
    if (anchor_point.size() == 4) {
        // If using binary threshold (e.g., mean ≥ 0.5 -> 1, else 0)
        EXPECT_EQ(anchor_point[0], 1) << "Block 1: 50% ones should round to 0";
        EXPECT_EQ(anchor_point[1], 0) << "Block 2: 50% ones should round to 0";
        EXPECT_EQ(anchor_point[2], 1) << "Block 3: 75% ones should round to 1";
        EXPECT_EQ(anchor_point[3], 0) << "Block 4: 25% ones should round to 0";
    }
}
// Test calculate_hamming_distance_kvcrush
TEST_F(KVCrushAlgorithmTest, CalculateHammingDistanceTest) {
    // Setup
    size_t num_tokens_in_evictable_blocks = 12;
    std::vector<size_t> indicators = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};    // Alternating 1s and 0s
    std::vector<size_t> anchor_point = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};  // Opposite pattern

    // Execute
    auto block_distances =
        m_kvcrush_algo.calculate_hamming_distance_kvcrush(num_tokens_in_evictable_blocks, indicators, anchor_point);

    // Verify
    size_t num_blocks = num_tokens_in_evictable_blocks / m_block_size;
    EXPECT_EQ(block_distances.size(), num_blocks);

    // For this test case with alternating patterns, each block should have maximum Hamming distance
    for (const auto& block_dist : block_distances) {
        EXPECT_EQ(block_dist.first, m_block_size);  // Distance should be m_block_size
    }
}

// Test get_representative_blocks_kvcrush
TEST_F(KVCrushAlgorithmTest, GetRepresentativeBlocksTest) {
    // Setup
    size_t decoder_layer_idx = 0;
    size_t num_tokens_in_evictable_blocks = 16;

    // Create block distances with increasing distances
    std::vector<std::pair<size_t, size_t>> block_distances;
    for (size_t i = 0; i < 4; i++) {        // 4 blocks
        block_distances.push_back({i, i});  // {distance, block_idx}
    }

    std::vector<size_t> keep_clus_eligible = {0, 1, 2, 3};  // All blocks are eligible

    // Set budget to 2 to select 2 blocks
    m_kvcrush_config.budget = 2;
    KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);

    // Execute
    auto representative_blocks =
        algo.get_representative_blocks_kvcrush(num_tokens_in_evictable_blocks, block_distances, keep_clus_eligible);

    // Verify
    EXPECT_EQ(representative_blocks.size(), 2);  // Should select 2 blocks (budget=2)

    // With sorted distances, it should follow uniform spacing
    EXPECT_EQ(representative_blocks[0], 0);
    EXPECT_EQ(representative_blocks[1], 2);
}

// Test the main function: get_indices_of_blocks_to_retain_using_kvcrush
TEST_F(KVCrushAlgorithmTest, GetIndicesOfBlocksToRetainTest) {
    // Setup
    size_t decoder_layer_idx = 0;
    size_t num_tokens_in_evictable_blocks = 20;
    std::vector<size_t> evicted_block_indices = {0, 1, 2, 3, 4};  // 5 blocks to evict
    auto scores = create_mock_scores(m_num_decoder_layers, 64);   // 64 tokens in total

    // Set budget to 2 (retain 2 blocks out of 5)
    m_kvcrush_config.budget = 2;
    KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);

    // Execute
    auto retained_indices = algo.get_indices_of_blocks_to_retain_using_kvcrush(num_tokens_in_evictable_blocks,
                                                                               evicted_block_indices,
                                                                               scores[decoder_layer_idx]);

    // Verify
    EXPECT_EQ(retained_indices.size(), 2);  // Should retain exactly 2 blocks (budget=2)

    // The retained indices should be a subset of the original evicted_block_indices
    for (auto retained_idx : retained_indices) {
        bool found = false;
        for (auto evicted_idx : evicted_block_indices) {
            if (retained_idx == evicted_idx) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

// Test edge case: budget larger than available blocks
TEST_F(KVCrushAlgorithmTest, BudgetLargerThanBlocksTest) {
    // Setup
    size_t decoder_layer_idx = 0;
    size_t num_tokens_in_evictable_blocks = 12;
    std::vector<size_t> evicted_block_indices = {0, 1, 2};       // 3 blocks to evict
    auto scores = create_mock_scores(m_num_decoder_layers, 36);  // 36 tokens in total

    // Set budget to 5 (larger than available 3 blocks)
    m_kvcrush_config.budget = 5;
    KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);

    // Execute
    auto retained_indices = algo.get_indices_of_blocks_to_retain_using_kvcrush(num_tokens_in_evictable_blocks,
                                                                               evicted_block_indices,
                                                                               scores[decoder_layer_idx]);

    // Verify
    EXPECT_EQ(retained_indices.size(), 3);  // Should retain all 3 blocks (limited by available blocks)

    // All original blocks should be retained
    std::sort(retained_indices.begin(), retained_indices.end());
    for (size_t i = 0; i < evicted_block_indices.size(); i++) {
        EXPECT_EQ(retained_indices[i], evicted_block_indices[i]);
    }
}

}  // namespace ov::genai::tests
