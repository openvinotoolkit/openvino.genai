// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "speculative_decoding/continuous_batching_for_speculative_decoding_impl.hpp"

class CBForSDTest : public testing::Test, public ov::genai::ContinuousBatchingPipeline {
protected:
    class PipelineTestInstance : public ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl {
    public:
        PipelineTestInstance() {
            m_sampler = std::make_shared<ov::genai::Sampler>();
        };

        ov::genai::GenerationHandle add_request(uint64_t request_id, const ov::Tensor& input_ids) {
            auto sampling_params = ov::genai::greedy();
            sampling_params.num_assistant_tokens = 1;

            ov::genai::SequenceGroup::Ptr sequence_group = std::make_shared<ov::genai::SequenceGroup>(request_id, input_ids,
                                                                                sampling_params, 
                                                                                32);

            {
                std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
                m_awaiting_requests.push_back(sequence_group);
            }
            pull_awaiting_requests();
            return std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(), sampling_params);
        };

    };

    PipelineTestInstance m_pipeline = PipelineTestInstance();
};

TEST_F(CBForSDTest, init_sequence_by_not_empty__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, init_sequence_by_empty__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = {};
    std::vector<float> log_probs = {};
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_EQ(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, no_updated_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    ov::genai::GeneratedSequences candidate_1{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_EQ(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, remove_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    tokens = { 0, 1 };
    log_probs = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate_1{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, remove_and_replace_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    tokens = { 0, 1, 4 };
    log_probs = { 0.1f, 0.2f, 0.4f };
    ov::genai::GeneratedSequences candidate_1{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 1);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, add_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    tokens = { 0, 1, 2, 3, 4 };
    log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
    ov::genai::GeneratedSequences candidate_1{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, update_empty_sequence_by_not_empty__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens_0 = { 0, 1, 2 },
                         tokens_1 = { 0, 1 };
    std::vector<float> log_probs_0 = { 0.1f, 0.2f, 0.3f },
                       log_probs_1 = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens_0, log_probs_0) },
        { 1, ov::genai::GeneratedSequence(tokens_1, log_probs_1) }
    };
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens_0);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs_0);

    ASSERT_EQ(after.at(0).size(), 1);
    ASSERT_EQ(after.at(0).size(), 1);
}

TEST_F(CBForSDTest, init_sequence_by_not_empty__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens_0 = { 0, 1, 2 },
                         tokens_1 = { 0, 1 };
    std::vector<float> log_probs_0 = { 0.1f, 0.2f, 0.3f },
                       log_probs_1 = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens_0, log_probs_0) },
        { 1, ov::genai::GeneratedSequence(tokens_1, log_probs_1) }
    };
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens_1);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs_1);

    ASSERT_EQ(after.at(0).at(1).token_ids, tokens_1);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs_1);
}

TEST_F(CBForSDTest, init_sequence_by_empty__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = {};
    std::vector<float> log_probs = {};
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens, log_probs) },
    };
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_EQ(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
    ASSERT_EQ(after.at(0).at(1).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs);
}

TEST_F(CBForSDTest, no_updated_tokens__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens_0 = { 0, 1, 2 }, tokens_1 = { 0, 1 };
    std::vector<float> log_probs_0 = { 0.1f, 0.2f, 0.3f }, log_probs_1 = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens_0, log_probs_0) },
        { 1, ov::genai::GeneratedSequence(tokens_1, log_probs_1) },
    };
    
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens_1, log_probs_1) },
        { 1, ov::genai::GeneratedSequence(tokens_1, log_probs_1) },
    };

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens_1);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs_1);
    ASSERT_EQ(after.at(0).at(1).token_ids, tokens_1);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs_1);
}

TEST_F(CBForSDTest, remove_tokens__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens, log_probs) },
    };
    
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    std::vector<int64_t> tokens_new = { 0, 1 };
    std::vector<float> log_probs_new = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens_new, log_probs_new) },
    };

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_NE(after.at(0).at(1).token_ids, before.at(0).at(1).token_ids);
    ASSERT_NE(after.at(0).at(1).log_probs, before.at(0).at(1).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens_new);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs_new);
    ASSERT_EQ(after.at(0).at(1).token_ids, tokens_new);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs_new);
}

TEST_F(CBForSDTest, remove_and_replace_tokens__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens, log_probs) },
    };
    
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    std::vector<int64_t> new_tokens = { 0, 1, 4 };
    std::vector<float> new_log_probs = { 0.1f, 0.2f, 0.4f };
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(new_tokens, new_log_probs) },
    };

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 1);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_EQ(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
    ASSERT_NE(after.at(0).at(1).token_ids, before.at(0).at(1).token_ids);
    ASSERT_NE(after.at(0).at(1).log_probs, before.at(0).at(1).log_probs);
    ASSERT_EQ(after.at(0).at(1).token_ids, new_tokens);
    ASSERT_EQ(after.at(0).at(1).log_probs, new_log_probs);
}

TEST_F(CBForSDTest, add_tokens__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens, log_probs) },
    };
    
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    tokens = { 0, 1, 2, 3, 4 };
    log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
    std::vector<int64_t> new_tokens = { 0, 1, 2, 3, 4, 5 };
    std::vector<float> new_log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(new_tokens, new_log_probs) },
    };

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
    ASSERT_NE(after.at(0).at(1).token_ids, before.at(0).at(1).token_ids);
    ASSERT_NE(after.at(0).at(1).log_probs, before.at(0).at(1).log_probs);
    ASSERT_EQ(after.at(0).at(1).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs);
}

