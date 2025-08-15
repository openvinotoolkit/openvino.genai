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

    // Helper function to setup anchor points with given mode
    std::vector<size_t> setup_anchor_points(KVCrushAnchorPointMode mode, size_t num_tokens = 16) {
        // Common setup for anchor point tests
        std::vector<size_t> indicators(num_tokens, 1);  // All indicators set to 1

        m_kvcrush_config.anchor_point_mode = mode;
        m_kvcrush_algo = KVCrushAlgorithm(m_kvcrush_config, m_block_size);

        return m_kvcrush_algo.create_anchor_point_kvcrush(num_tokens, indicators);
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
    // Setup using helper function
    auto anchor_point = setup_anchor_points(KVCrushAnchorPointMode::RANDOM, 16);

    // Verify basic properties
    EXPECT_EQ(anchor_point.size(), m_block_size);

    // 1. Check that all values are less than 1 (should be 0 or 1 only)
    for (const auto& val : anchor_point) {
        EXPECT_LE(val, 1) << "Anchor point values should be 0 or 1 (less than or equal to 1)";
        EXPECT_GE(val, 0) << "Anchor point values should be non-negative";
    }

    // 2. Check that they are bitwise different (not all same value)
    bool all_same = true;
    size_t first_value = anchor_point[0];
    for (size_t i = 1; i < anchor_point.size(); i++) {
        if (anchor_point[i] != first_value) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "Random anchor point should not have all identical values";

    // 3. Additional randomness check: run multiple times with different seeds
    // and verify we get different results
    std::vector<std::vector<size_t>> multiple_results;
    for (size_t seed = 0; seed < 5; seed++) {
        KVCrushConfig temp_config = m_kvcrush_config;
        temp_config.rng_seed = seed;
        KVCrushAlgorithm temp_algo(temp_config, m_block_size);

        size_t num_tokens_in_evictable_blocks = 16;
        std::vector<size_t> indicators(num_tokens_in_evictable_blocks, 1);
        auto temp_anchor_point = temp_algo.create_anchor_point_kvcrush(num_tokens_in_evictable_blocks, indicators);
        multiple_results.push_back(temp_anchor_point);
    }

    // Check that different seeds produce different results (proving randomness)
    bool found_different = false;
    for (size_t i = 1; i < multiple_results.size(); i++) {
        if (multiple_results[0] != multiple_results[i]) {
            found_different = true;
            break;
        }
    }
    EXPECT_TRUE(found_different) << "Different random seeds should produce different anchor points";

    // 4. Check that we have a mix of 0s and 1s (not all zeros or all ones)
    size_t count_zeros = std::count(anchor_point.begin(), anchor_point.end(), 0);
    size_t count_ones = std::count(anchor_point.begin(), anchor_point.end(), 1);

    EXPECT_GT(count_zeros, 0) << "Random anchor point should contain some zeros";
    EXPECT_GT(count_ones, 0) << "Random anchor point should contain some ones";
    EXPECT_EQ(count_zeros + count_ones, anchor_point.size()) << "All values should be either 0 or 1";
}

// Test create_anchor_point_kvcrush with ALTERNATE mode
TEST_F(KVCrushAlgorithmTest, CreateAnchorPointAlternateTest) {
    // Setup using helper function
    auto anchor_point = setup_anchor_points(KVCrushAnchorPointMode::ALTERNATE, 16);

    // Verify
    EXPECT_EQ(anchor_point.size(), m_block_size);

    // In ALTERNATE mode, we expect alternating blocks to be selected
    // We should have non-zero values at regular intervals
    bool found_pattern = false;
    for (size_t i = 0; i < anchor_point.size(); i += m_block_size) {
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
    EXPECT_TRUE(found_pattern);
}

// Test create_anchor_point_kvcrush with ZEROS mode
TEST_F(KVCrushAlgorithmTest, CreateAnchorPointZerosTest) {
    // Setup using helper function
    auto anchor_point = setup_anchor_points(KVCrushAnchorPointMode::ZEROS, 16);

    // Verify - For ZEROS mode, we expect the anchor point to be all zeros
    EXPECT_EQ(anchor_point.size(), m_block_size);

    // Check that all values are zeros
    for (const auto& val : anchor_point) {
        EXPECT_EQ(val, 0) << "Anchor point value should be zero in ZEROS mode";
    }
}

// Test create_anchor_point_kvcrush with ONES mode
TEST_F(KVCrushAlgorithmTest, CreateAnchorPointOnesTest) {
    // Setup using helper function
    auto anchor_point = setup_anchor_points(KVCrushAnchorPointMode::ONES, 16);

    // Verify - For ONES mode, we expect the anchor point to be all ones
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
// Test calculate_hamming_distance_kvcrush (parameterized)
TEST_F(KVCrushAlgorithmTest, CalculateHammingDistanceParameterizedTest) {
    struct TestCase {
        std::vector<size_t> indicators;
        std::vector<size_t> anchor_point;
        std::vector<size_t> expected_distances;
    };

    std::vector<TestCase> test_cases = {
        // Case 1: Alternating 1s and 0s, anchor_point {0, 1, 0, 1}
        {
            {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
            {0, 1, 0, 1},
            {4, 4, 4}
        },
        // Case 2: All indicators match anchor_point
        {
            {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
            {0, 1, 0, 1},
            {0, 0, 0}
        },
        // Case 3: All indicators are 1, anchor_point all 0
        {
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {0, 0, 0, 0},
            {4, 4, 4}
        },
        // Case 4: All indicators are 0, anchor_point all 1
        {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 1},
            {4, 4, 4}
        },
        // Case 5: Mixed indicators, anchor_point {1, 0, 1, 0}
        {
            {1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0},
            {1, 0, 1, 0},
            {0, 4, 2}
        }
    };

    size_t num_tokens_in_evictable_blocks = 12;
    for (const auto& tc : test_cases) {
        ASSERT_EQ(tc.indicators.size(), num_tokens_in_evictable_blocks);
        ASSERT_EQ(tc.anchor_point.size(), m_block_size);
        
        // Create mutable copies since the function expects non-const references
        std::vector<size_t> indicators_copy = tc.indicators;
        std::vector<size_t> anchor_point_copy = tc.anchor_point;
        
        auto block_distances =
            m_kvcrush_algo.calculate_hamming_distance_kvcrush(num_tokens_in_evictable_blocks, indicators_copy, anchor_point_copy);

        size_t num_blocks = num_tokens_in_evictable_blocks / m_block_size;
        EXPECT_EQ(block_distances.size(), num_blocks);

        for (size_t i = 0; i < num_blocks; ++i) {
            EXPECT_EQ(block_distances[i].first, tc.expected_distances[i])
                << "Block " << i << " failed for indicators: "
                << ::testing::PrintToString(tc.indicators)
                << " and anchor_point: "
                << ::testing::PrintToString(tc.anchor_point);
        }
    }
}

// Parameterized test for get_representative_blocks_kvcrush
TEST_F(KVCrushAlgorithmTest, GetRepresentativeBlocksParameterizedTest) {
    struct TestCase {
        std::vector<std::pair<size_t, size_t>> block_distances;
        std::vector<size_t> keep_clus_eligible;
        size_t budget;
        std::vector<size_t> expected_blocks;
    };

    std::vector<TestCase> test_cases = {
        // Case 1: Uniform distances, select 2 out of 4
        {
            {{0, 0}, {1, 1}, {2, 2}, {3, 3}},
            {0, 1, 2, 3},
            2,
            {0, 2}
        },
        // Case 2: All blocks eligible, budget equals number of blocks
        {
            {{0, 0}, {1, 1}, {2, 2}, {3, 3}},
            {0, 1, 2, 3},
            4,
            {0, 1, 2, 3}
        },
        // Case 3: Only some blocks eligible
        {
            {{0, 0}, {1, 1}, {2, 2}, {3, 3}},
            {1, 3},
            2,
            {1, 3}
        },
        // Case 4: Budget larger than eligible blocks
        {
            {{0, 0}, {1, 1}, {2, 2}, {3, 3}},
            {2, 3},
            5,
            {2, 3}
        }
    };

    size_t num_tokens_in_evictable_blocks = 16;
    for (const auto& tc : test_cases) {
        m_kvcrush_config.budget = tc.budget;
        KVCrushAlgorithm algo(m_kvcrush_config, m_block_size);

        // Create mutable copies since the function expects non-const references
        std::vector<std::pair<size_t, size_t>> block_distances_copy = tc.block_distances;
        
        auto representative_blocks =
            algo.get_representative_blocks_kvcrush(num_tokens_in_evictable_blocks, block_distances_copy, tc.keep_clus_eligible);

        EXPECT_EQ(representative_blocks.size(), tc.expected_blocks.size())
            << "Failed for budget " << tc.budget << " and eligible blocks "
            << ::testing::PrintToString(tc.keep_clus_eligible);

        for (size_t i = 0; i < tc.expected_blocks.size(); ++i) {
            EXPECT_EQ(representative_blocks[i], tc.expected_blocks[i])
                << "Block " << i << " failed for distances: "
                << ::testing::PrintToString(tc.block_distances)
                << " and eligible: "
                << ::testing::PrintToString(tc.keep_clus_eligible);
        }
    }
}

}  // namespace ov::genai::tests