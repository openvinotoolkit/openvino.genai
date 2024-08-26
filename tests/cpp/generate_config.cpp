// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvino/core/except.hpp>
#include "openvino/genai/generation_config.hpp"
#include "sequence_group.hpp"


using namespace ov::genai;

TEST(GenerationConfigTest, invalid_temperature) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.temperature = -0.1;
    config.do_sample = true;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_temperature) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.temperature = 0.1;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_top_p) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.top_p = -0.5;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.top_p = 1.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_top_p) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.top_p = 0.1;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_repeatition_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.repetition_penalty = -3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.repetition_penalty = -0.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_repeatition_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.repetition_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.repetition_penalty = 0.1;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_presence_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.presence_penalty = 3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.presence_penalty = -3.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_presence_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.presence_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.presence_penalty = -2.0;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, invalid_frequency_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.frequency_penalty = 3.0;
    EXPECT_THROW(config.validate(), ov::Exception);
    config.frequency_penalty = -3.1;
    EXPECT_THROW(config.validate(), ov::Exception);
}

TEST(GenerationConfigTest, valid_frequency_penalty) {
    GenerationConfig config;
    config.max_new_tokens = 20;
    config.do_sample = true;
    config.frequency_penalty = 1.8;
    EXPECT_NO_THROW(config.validate());
    config.frequency_penalty = -2.0;
    EXPECT_NO_THROW(config.validate());
}

TEST(GenerationConfigTest, single_stop_sequence_single_match) {
    GenerationConfig config;
    config.stop_token_ids = { {7, 8, 9} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_TRUE(sequence_group->stop_token_ids_hit(generated_tokens));
}

TEST(GenerationConfigTest, multiple_stop_sequence_single_match) {
    GenerationConfig config;
    config.stop_token_ids = { {7, 8, 9}, {11, 12}, {13} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_TRUE(sequence_group->stop_token_ids_hit(generated_tokens));
}

TEST(GenerationConfigTest, multiple_stop_sequence_multiple_match) {
    GenerationConfig config;
    config.stop_token_ids = { {7, 8, 9}, {8, 9}, {9} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_TRUE(sequence_group->stop_token_ids_hit(generated_tokens));
}

TEST(GenerationConfigTest, single_stop_sequence_no_match) {
    GenerationConfig config;
    config.stop_token_ids = { {10} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_FALSE(sequence_group->stop_token_ids_hit(generated_tokens));
}

TEST(GenerationConfigTest, multiple_stop_sequence_no_match) {
    GenerationConfig config;
    config.stop_token_ids = { {10}, {10, 11} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_FALSE(sequence_group->stop_token_ids_hit(generated_tokens));
}

TEST(GenerationConfigTest, single_stop_sequence_exceeding_size) {
    GenerationConfig config;
    config.stop_token_ids = { {3, 4, 5, 6, 7, 8, 9, 10} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_FALSE(sequence_group->stop_token_ids_hit(generated_tokens));
}

TEST(GenerationConfigTest, all_stop_sequence_exceeding_size) {
    GenerationConfig config;
    config.stop_token_ids = { {3, 4, 5, 6, 7, 8, 9, 10}, {3, 4, 5, 6, 7, 8, 9, 10, 11, 12} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_FALSE(sequence_group->stop_token_ids_hit(generated_tokens));
}

TEST(GenerationConfigTest, some_stop_sequence_exceeding_size) {
    GenerationConfig config;
    config.stop_token_ids = { {3, 4, 5, 6, 7, 8, 9, 10}, {5, 6, 7, 8, 9} ,{3, 4, 5, 6, 7, 8, 9, 10, 11, 12} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_TRUE(sequence_group->stop_token_ids_hit(generated_tokens));
}

TEST(GenerationConfigTest, multiple_stop_sequence_single_exact_match) {
    GenerationConfig config;
    config.stop_token_ids = { {9, 10}, {3, 4, 5, 6, 7, 8, 9}, {3, 4, 5, 6, 7, 8, 9, 10, 11, 12} };

    std::vector<int64_t> prompt_tokens = {0, 1, 2};
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                config, 32, false);

    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    ASSERT_TRUE(sequence_group->stop_token_ids_hit(generated_tokens));
}
