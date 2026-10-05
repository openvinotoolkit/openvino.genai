// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "sampling/sampler.hpp"
#include "openvino/genai/generation_config.hpp"
#include "utils.hpp"


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

TEST(SamplerValidationMode, gen_phase_to_cut_whole_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    constexpr size_t processed_before = 5;
    sequence_groups.front()->update_processed_tokens_num(processed_before);

    // append candidates [ 2, 3, 4 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i + 1, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 2, 3, 4]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
    EXPECT_EQ(sequence_groups.front()->get_num_processed_tokens(), processed_before + 1)
        << "Full rejection must retain the target replacement token as accepted depth one";
}

TEST(SamplerValidationMode, gen_phase_to_cut_part_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    sequence_groups.front()->update_processed_tokens_num(5);

    // append candidates [ 1, 2, 2 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        int64_t token_id = i == num_validated_tokens ? i - 1 : i;
        sequence_groups.front()->get_sequences().front()->append_token(token_id, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2, 2]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, gen_phase) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    sequence_groups.front()->update_processed_tokens_num(5);

    // append candidates [ 1, 2, 3 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2, 3]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3, 4};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase_to_cut_part_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 0, 1, 1 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        int64_t token_id = i + 1 == num_validated_tokens ? i - 1 : i;
        sequence_groups.front()->get_sequences().front()->append_token(token_id, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 1]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase_to_cut_whole_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 1, 2, 3 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i + 1, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [1, 2, 3]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 0, 1, 2 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

// Speculative sampling rejects the last candidate: its replacement is drawn from the residual of the target
// distribution at the rejected position, and the rejected token is rolled back from the KV cache.
TEST(SamplerValidationMode, gen_phase_multinomial_rejection) {
    GenerationConfig sampling_config;
    sampling_config.max_new_tokens = 30;
    sampling_config.do_sample = true;
    // top_k = 1 makes every target distribution one-hot, so acceptance and resampling are deterministic.
    sampling_config.top_k = 1;
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };
    const auto sequence = sequence_groups.front()->get_sequences().front();

    // to emulate processed prompt and add next token [ 0 ]
    sequence->append_token(0, 0.f);
    constexpr size_t processed_before = 5;
    sequence_groups.front()->update_processed_tokens_num(processed_before);

    // append candidates [ 1, 2, 3 ], each drafted with q(t) = 1
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        sequence->append_token(i, 0.f);
    }
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    sequence_groups.front()->schedule_tokens(sequence_groups.front()->get_num_available_tokens_for_batching());

    // the target accepts 1 and 2, prefers 4 over the candidate 3, and prefers 0 at the next position
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
    };
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.set_candidate_distributions(0, 0, {{0, 1.f, 0, 0, 0}, {0, 0, 1.f, 0, 0}, {0, 0, 0, 1.f, 0}});
    sampler.sample(sequence_groups, gen_input_ids, true);

    const TokenIds expected{0, 1, 2, 4};
    EXPECT_EQ(sequence->get_generated_ids(), expected);
    EXPECT_EQ(sequence_groups.front()->get_num_processed_tokens(), processed_before + 3)
        << "The rejected candidate must be removed from the KV cache";
}
