// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "sampler.hpp"


using namespace ov::genai;

TEST(SamplerStopTokenIdsTest, single_stop_sequence_single_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {7, 8, 9} };
    ASSERT_TRUE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_sequence_single_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {7, 8, 9}, {11, 12}, {13} };
    ASSERT_TRUE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_sequence_multiple_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {7, 8, 9}, {8, 9}, {9} };
    ASSERT_TRUE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, single_stop_sequence_no_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {10} };
    ASSERT_FALSE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_sequence_no_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {10}, {10, 11} };
    ASSERT_FALSE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, single_stop_sequence_exceeding_size) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {3, 4, 5, 6, 7, 8, 9, 10} };
    ASSERT_FALSE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, all_stop_sequence_exceeding_size) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {3, 4, 5, 6, 7, 8, 9, 10}, {3, 4, 5, 6, 7, 8, 9, 10, 11, 12} };
    ASSERT_FALSE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, some_stop_sequence_exceeding_size) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {3, 4, 5, 6, 7, 8, 9, 10}, {5, 6, 7, 8, 9} ,{3, 4, 5, 6, 7, 8, 9, 10, 11, 12} };
    ASSERT_TRUE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_sequence_single_exact_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::vector<std::vector<int64_t>> stop_token_ids = { {9, 10}, {3, 4, 5, 6, 7, 8, 9}, {3, 4, 5, 6, 7, 8, 9, 10, 11, 12} };
    ASSERT_TRUE(is_stop_token_ids_hit(generated_tokens, stop_token_ids));
}
