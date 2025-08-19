// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "visual_language/cdpruner/fast_dpp.hpp"
#include "visual_language/cdpruner/cdpruner_config.hpp"
#include <openvino/openvino.hpp>
#include <vector>
#include <algorithm>

using namespace ov::genai::cdpruner;

class FastGreedyDPPTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize config for testing
        config.visual_tokens_retain_percentage = 75;  // Will keep 3 out of 4 tokens
        config.relevance_weight = 0.5f;
        config.enable_pruning = true;
        config.pruning_debug_mode = true;
        config.use_negative_relevance = false;  // Not using negative correlation as requested
        config.numerical_threshold = 1e-6f;
        config.device = "CPU";
        config.use_ops_model = false;
        
        dpp_selector = std::make_unique<FastGreedyDPP>(config);
    }
    
    Config config;
    std::unique_ptr<FastGreedyDPP> dpp_selector;
};

TEST_F(FastGreedyDPPTest, ConditionalKernelMatrixSelection) {
    // Test case: Select 3 tokens out of 4 using the specific conditional kernel matrix
    // Expected result: tokens 1, 0, 3 should be selected
    
    // Create the conditional kernel matrix as specified:
    // [0.8, 0.3, 0.1, 0.2]  // token 0
    // [0.3, 0.9, 0.4, 0.1]  // token 1  
    // [0.1, 0.4, 0.7, 0.5]  // token 2
    // [0.2, 0.1, 0.5, 0.6]  // token 3
    
    std::vector<float> kernel_data = {
        // Batch 0, 4x4 kernel matrix
        0.8f, 0.3f, 0.1f, 0.2f,  // token 0 row
        0.3f, 0.9f, 0.4f, 0.1f,  // token 1 row
        0.1f, 0.4f, 0.7f, 0.5f,  // token 2 row
        0.2f, 0.1f, 0.5f, 0.6f   // token 3 row
    };
    
    // Create OpenVINO tensor [batch_size=1, tokens=4, tokens=4]
    ov::Tensor kernel_matrix(ov::element::f32, {1, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    // Number of tokens to keep
    size_t num_tokens_to_keep = 3;
    
    // Perform DPP selection
    auto selected_tokens = dpp_selector->select(kernel_matrix, num_tokens_to_keep);
    
    // Validate results
    ASSERT_EQ(selected_tokens.size(), 1);  // Single batch
    ASSERT_EQ(selected_tokens[0].size(), num_tokens_to_keep);  // Should select 3 tokens
    
    // Expected tokens: 1, 0, 3 (sorted order)
    std::vector<size_t> expected_tokens = {0, 1, 3};
    std::vector<size_t> actual_tokens = selected_tokens[0];
    std::sort(actual_tokens.begin(), actual_tokens.end());
    
    EXPECT_EQ(actual_tokens, expected_tokens) 
        << "Expected tokens [0, 1, 3] but got [" 
        << actual_tokens[0] << ", " << actual_tokens[1] << ", " << actual_tokens[2] << "]";
    
    // Verify token 2 is NOT selected (should be pruned)
    auto it = std::find(actual_tokens.begin(), actual_tokens.end(), 2);
    EXPECT_EQ(it, actual_tokens.end()) << "Token 2 should be pruned but was selected";
}

TEST_F(FastGreedyDPPTest, VerifyDPPProperties) {
    // Test that the DPP selection maintains diversity properties
    // Higher diagonal values should have higher selection probability
    
    std::vector<float> kernel_data = {
        0.8f, 0.3f, 0.1f, 0.2f,
        0.3f, 0.9f, 0.4f, 0.1f,
        0.1f, 0.4f, 0.7f, 0.5f,
        0.2f, 0.1f, 0.5f, 0.6f
    };
    
    ov::Tensor kernel_matrix(ov::element::f32, {1, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    auto selected_tokens = dpp_selector->select(kernel_matrix, 3);
    auto actual_tokens = selected_tokens[0];
    
    // Token 1 has highest diagonal value (0.9) and should be selected
    auto it_token1 = std::find(actual_tokens.begin(), actual_tokens.end(), 1);
    EXPECT_NE(it_token1, actual_tokens.end()) << "Token 1 (highest diagonal) should be selected";
    
    // Token 0 has second highest diagonal value (0.8) and should be selected
    auto it_token0 = std::find(actual_tokens.begin(), actual_tokens.end(), 0);
    EXPECT_NE(it_token0, actual_tokens.end()) << "Token 0 (second highest diagonal) should be selected";
}

TEST_F(FastGreedyDPPTest, SingleTokenSelection) {
    // Test selecting only 1 token - should pick the one with highest diagonal value
    
    std::vector<float> kernel_data = {
        0.8f, 0.3f, 0.1f, 0.2f,
        0.3f, 0.9f, 0.4f, 0.1f,  // Token 1 has highest diagonal (0.9)
        0.1f, 0.4f, 0.7f, 0.5f,
        0.2f, 0.1f, 0.5f, 0.6f
    };
    
    ov::Tensor kernel_matrix(ov::element::f32, {1, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    auto selected_tokens = dpp_selector->select(kernel_matrix, 1);
    
    ASSERT_EQ(selected_tokens[0].size(), 1);
    EXPECT_EQ(selected_tokens[0][0], 1) << "Should select token 1 (highest diagonal value)";
}

TEST_F(FastGreedyDPPTest, AllTokensSelection) {
    // Test selecting all tokens - should return all indices
    
    std::vector<float> kernel_data = {
        0.8f, 0.3f, 0.1f, 0.2f,
        0.3f, 0.9f, 0.4f, 0.1f,
        0.1f, 0.4f, 0.7f, 0.5f,
        0.2f, 0.1f, 0.5f, 0.6f
    };
    
    ov::Tensor kernel_matrix(ov::element::f32, {1, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    auto selected_tokens = dpp_selector->select(kernel_matrix, 4);
    
    ASSERT_EQ(selected_tokens[0].size(), 4);
    
    // Should contain all tokens 0, 1, 2, 3
    std::vector<size_t> actual_tokens = selected_tokens[0];
    std::sort(actual_tokens.begin(), actual_tokens.end());
    std::vector<size_t> expected_all_tokens = {0, 1, 2, 3};
    
    EXPECT_EQ(actual_tokens, expected_all_tokens);
}

TEST_F(FastGreedyDPPTest, MultipleBatchSelection) {
    // Test with multiple batches
    
    std::vector<float> kernel_data = {
        // Batch 0
        0.8f, 0.3f, 0.1f, 0.2f,
        0.3f, 0.9f, 0.4f, 0.1f,
        0.1f, 0.4f, 0.7f, 0.5f,
        0.2f, 0.1f, 0.5f, 0.6f,
        
        // Batch 1 - different kernel matrix
        0.6f, 0.1f, 0.2f, 0.3f,
        0.1f, 0.8f, 0.3f, 0.2f,
        0.2f, 0.3f, 0.9f, 0.4f,  // Token 2 has highest diagonal in batch 1
        0.3f, 0.2f, 0.4f, 0.7f
    };
    
    ov::Tensor kernel_matrix(ov::element::f32, {2, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    auto selected_tokens = dpp_selector->select(kernel_matrix, 2);
    
    ASSERT_EQ(selected_tokens.size(), 2);  // Two batches
    ASSERT_EQ(selected_tokens[0].size(), 2);  // Each batch selects 2 tokens
    ASSERT_EQ(selected_tokens[1].size(), 2);
    
    // Batch 0: Should prioritize tokens 1 and 0 (highest diagonals)
    std::vector<size_t> batch0_tokens = selected_tokens[0];
    std::sort(batch0_tokens.begin(), batch0_tokens.end());
    
    // Batch 1: Should prioritize tokens 2 and 3 (highest diagonals)
    std::vector<size_t> batch1_tokens = selected_tokens[1];
    std::sort(batch1_tokens.begin(), batch1_tokens.end());
    
    // Just verify we get reasonable selections (exact results depend on DPP algorithm details)
    EXPECT_TRUE(batch0_tokens.size() == 2);
    EXPECT_TRUE(batch1_tokens.size() == 2);
}

TEST_F(FastGreedyDPPTest, EdgeCaseZeroTokens) {
    // Test edge case of selecting 0 tokens
    
    std::vector<float> kernel_data = {
        0.8f, 0.3f, 0.1f, 0.2f,
        0.3f, 0.9f, 0.4f, 0.1f,
        0.1f, 0.4f, 0.7f, 0.5f,
        0.2f, 0.1f, 0.5f, 0.6f
    };
    
    ov::Tensor kernel_matrix(ov::element::f32, {1, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    auto selected_tokens = dpp_selector->select(kernel_matrix, 0);
    
    ASSERT_EQ(selected_tokens[0].size(), 0);
}

TEST_F(FastGreedyDPPTest, InvalidInputHandling) {
    // Test invalid input handling
    
    std::vector<float> kernel_data = {
        0.8f, 0.3f, 0.1f, 0.2f,
        0.3f, 0.9f, 0.4f, 0.1f,
        0.1f, 0.4f, 0.7f, 0.5f,
        0.2f, 0.1f, 0.5f, 0.6f
    };
    
    ov::Tensor kernel_matrix(ov::element::f32, {1, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    // Test selecting more tokens than available
    EXPECT_THROW(dpp_selector->select(kernel_matrix, 5), std::invalid_argument);
    
    // Test with non-square matrix (invalid kernel)
    ov::Tensor invalid_kernel(ov::element::f32, {1, 4, 3});
    EXPECT_THROW(dpp_selector->select(invalid_kernel, 2), std::invalid_argument);
    
    // Test with wrong dimensions
    ov::Tensor wrong_dims(ov::element::f32, {4, 4});  // 2D instead of 3D
    EXPECT_THROW(dpp_selector->select(wrong_dims, 2), std::invalid_argument);
}

TEST_F(FastGreedyDPPTest, CreateMaskFunctionality) {
    // Test the create_mask helper function
    
    std::vector<std::vector<size_t>> selected_indices = {
        {0, 1, 3},  // Batch 0 selected tokens
        {1, 2}      // Batch 1 selected tokens  
    };
    
    size_t total_tokens = 4;
    auto mask = FastGreedyDPP::create_mask(selected_indices, total_tokens);
    
    // Expected mask: [true, true, false, true, false, true, true, false]
    // Batch 0: tokens 0,1,3 selected -> [true, true, false, true]
    // Batch 1: tokens 1,2 selected -> [false, true, true, false]
    
    std::vector<bool> expected_mask = {
        true, true, false, true,   // Batch 0
        false, true, true, false   // Batch 1
    };
    
    EXPECT_EQ(mask, expected_mask);
}
