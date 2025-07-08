// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "sampling/sampler.hpp"
#include "openvino/genai/generation_config.hpp"


using namespace ov::genai;

// main pipeline greedy sampling
// eagle pipeline beam seach for top k tree
TEST(SamplerValidationMode, eagle2_mode_initial) {
    auto sampling_config = ov::genai::greedy();

    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config, 32)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    sequence_groups.front()->update_processed_tokens_num(5);

    // append candidates, 3 branchs from draft model, a) seq group id 0: [ 2, 3, 4 ] b) seq group id: 3 [2, 3] c) seq group id : 5 [2, 1, 4]
    // simulate the generated results from the draft model, where each branch has its own sequence id
    std::map<size_t, std::vector<int64_t>> branches_to_validate = {
        {0, {2, 3, 4}},
        {3, {2, 3}},
        {5, {2, 1, 4}}
    };
    for (const auto& branch : branches_to_validate) {
        if (branch.first == sequence_groups.front()->get_sequences().front()->get_grouped_id()) {
            // reuse the first sequence
            for (auto iter : branch.second) {
                sequence_groups.front()->get_sequences().front()->append_token(iter, 1.f);
            }
            //sequence_groups.front()->get_sequences().front()->set_num_validated_tokens(branch.second.size());
        } else {
            // other branches are created as new sequences
            auto sequence = Sequence::create(branch.first, SequenceGroupType::TOKENS, 0);
            //sequence->set_grouped_id(branch.first);
            sequence->append_token(0, 1.f); // also append the first token
            for (auto iter : branch.second) {
                sequence->append_token(iter, 1.f);
            }
            //sequence->set_num_validated_tokens(branch.second.size());
            sequence_groups.front()->add_sequence(sequence);
        }
    }
    sequence_groups.front()->set_num_validated_tokens(8);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);
    /*
    std::vector<size_t> num_validated_tokens = 3;
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
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);*/
}