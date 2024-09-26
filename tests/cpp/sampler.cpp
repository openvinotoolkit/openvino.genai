// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "sampler.hpp"
#include "openvino/genai/generation_config.hpp"


using namespace ov::genai;

TEST(SamplerStopTokenIdsTest, single_stop_token_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = {9};
    ASSERT_TRUE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_token_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = {7, 8, 9};
    ASSERT_TRUE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, single_stop_sequence_no_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = { 10 };
    ASSERT_FALSE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_sequence_no_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = { 10, 10, 11 };
    ASSERT_FALSE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerValidationMode, check_validation_mode) {
    auto sampling_config = ov::genai::greedy();
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_ids(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_ids, sampling_config, 32, false)),
    };
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);
    sequence_groups.front()->get_sequences().front()->append_token(1, 1.f);
    sequence_groups.front()->get_sequences().front()->append_token(2, 1.f);
    sequence_groups.front()->get_sequences().front()->append_token(2, 1.f);
    sequence_groups.front()->update_processed_tokens_num(5);
    sequence_groups.front()->set_num_validated_tokens(3);
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Tokenizer t("");
    Sampler sampler(t);
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}
