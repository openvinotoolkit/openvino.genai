// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include "gtest/gtest.h"

#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "speculative_decoding/continuous_batching/pipeline_impl.hpp"
#include "utils.hpp"

class CBForSDTest : public testing::Test, public ov::genai::ContinuousBatchingPipeline {
protected:
    class PipelineTestInstance : public ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl {
    public:
        PipelineTestInstance() {
            m_sampler = std::make_shared<ov::genai::Sampler>();
        };

        ov::genai::GenerationHandle add_request(uint64_t request_id, const ov::Tensor& input_ids) {
            auto sampling_params = ov::genai::utils::get_greedy_config();
            sampling_params.num_assistant_tokens = 1;

            ov::genai::SequenceGroup::Ptr sequence_group = std::make_shared<ov::genai::SequenceGroup>(request_id, input_ids,
                                                                                sampling_params);

            {
                std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
                m_awaiting_requests.push_back(sequence_group);
            }
            pull_awaiting_requests();
            return std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(), sampling_params);
        };

        void enable_mtp_mode() {
            mtp_mode_enabled = true;
        }

        bool is_waiting(uint64_t request_id) const {
            auto request_it = std::find_if(m_requests.begin(), m_requests.end(), [request_id](const ov::genai::SequenceGroup::Ptr& request) {
                return request->get_request_id() == request_id;
            });
            OPENVINO_ASSERT(request_it != m_requests.end(), "Request is not found");
            return (*request_it)->is_waiting();
        }

    };

    PipelineTestInstance m_pipeline = PipelineTestInstance();
};

TEST(SDPerModelsPerfMetrics, DraftOverheadDiagnostics) {
    ov::genai::SDPerModelsPerfMetrics metrics;
    metrics.num_draft_tokens = 5;
    metrics.num_accepted_tokens = 3;

    metrics.main_model_metrics.raw_metrics.m_durations = {ov::genai::MicroSeconds(1000.0f)};
    metrics.main_model_metrics.raw_metrics.m_batch_sizes = {4};
    metrics.main_model_metrics.raw_metrics.m_inference_durations = {ov::genai::MicroSeconds(4000.0f)};

    metrics.draft_model_metrics.raw_metrics.m_durations = {ov::genai::MicroSeconds(1000.0f),
                                                            ov::genai::MicroSeconds(1000.0f)};
    metrics.draft_model_metrics.raw_metrics.m_batch_sizes = {4, 4};
    metrics.draft_model_metrics.raw_metrics.m_inference_durations = {ov::genai::MicroSeconds(2000.0f)};

    EXPECT_EQ(metrics.get_num_draft_processed_tokens(), 8);
    EXPECT_FLOAT_EQ(metrics.get_draft_processed_to_candidate_ratio(), 8.0f / 5.0f);
    EXPECT_FLOAT_EQ(metrics.get_draft_to_main_inference_duration_ratio(), 0.5f);
}

TEST(SDPerModelsPerfMetrics, DraftOverheadDiagnosticsReturnNanWithoutDenominator) {
    ov::genai::SDPerModelsPerfMetrics metrics;

    EXPECT_TRUE(std::isnan(metrics.get_draft_processed_to_candidate_ratio()));
    EXPECT_TRUE(std::isnan(metrics.get_draft_to_main_inference_duration_ratio()));
}

TEST(MtpDraftUpdatePlan, PreservesAcceptedPrefixAfterPartialRejection) {
    struct TestCase {
        size_t removed_draft_tokens;
        size_t accepted_draft_tokens;
        size_t hidden_state_start;
        size_t processed_tokens_to_rewind;
    };
    constexpr size_t hidden_state_len = 5;
    constexpr size_t num_draft_tokens = hidden_state_len - 1;
    constexpr size_t processed_tokens_before_update = 100;
    const std::vector<TestCase> test_cases{
        {4, 0, 0, 3},  // first candidate rejected
        {2, 2, 2, 1},  // two candidates accepted
        {1, 3, 3, 0},  // only the unforwarded tail candidate rejected
    };

    for (const auto& test_case : test_cases) {
        SCOPED_TRACE(test_case.removed_draft_tokens);
        const auto plan =
            ov::genai::detail::make_mtp_draft_update_plan(hidden_state_len, test_case.removed_draft_tokens);

        EXPECT_EQ(plan.hidden_state_start, test_case.hidden_state_start);
        EXPECT_EQ(plan.hidden_state_count, 1);
        EXPECT_EQ(plan.processed_tokens_to_rewind, test_case.processed_tokens_to_rewind);
        EXPECT_EQ(plan.num_tokens_to_validate, 0);
        EXPECT_EQ(plan.hidden_state_count, plan.num_tokens_to_validate + 1);
        EXPECT_EQ(processed_tokens_before_update - plan.processed_tokens_to_rewind,
                  processed_tokens_before_update - test_case.removed_draft_tokens + 1);
        EXPECT_EQ(test_case.accepted_draft_tokens,
                  num_draft_tokens - test_case.removed_draft_tokens);
        if (test_case.accepted_draft_tokens > 0) {
            EXPECT_LT(plan.hidden_state_count, test_case.accepted_draft_tokens + 1);
        }
    }
}

TEST(MtpDraftUpdatePlan, FullAcceptanceProcessesOnlyUnforwardedTailAndBonus) {
    constexpr size_t hidden_state_len = 5;
    constexpr size_t num_draft_tokens = hidden_state_len - 1;
    constexpr size_t processed_tokens_before_update = 100;
    const auto plan = ov::genai::detail::make_mtp_draft_update_plan(hidden_state_len, 0);

    EXPECT_EQ(plan.hidden_state_start, num_draft_tokens - 1);
    EXPECT_EQ(plan.hidden_state_count, 2);
    EXPECT_EQ(plan.processed_tokens_to_rewind, 0);
    EXPECT_EQ(plan.num_tokens_to_validate, 1);
    EXPECT_EQ(plan.hidden_state_count, plan.num_tokens_to_validate + 1);
    EXPECT_LT(plan.hidden_state_count, hidden_state_len);
    EXPECT_EQ(processed_tokens_before_update - plan.processed_tokens_to_rewind,
              processed_tokens_before_update);
}

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

TEST_F(CBForSDTest, mtp_rejection_pauses_draft_generation) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = {0, 1, 2};
    std::vector<float> log_probs = {0.1f, 0.2f, 0.3f};
    ov::genai::GeneratedSequences candidate{{0, ov::genai::GeneratedSequence(tokens, log_probs)}};

    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);
    ASSERT_FALSE(m_pipeline.is_waiting(0));

    m_pipeline.enable_mtp_mode();
    tokens = {0, 1};
    log_probs = {0.1f, 0.2f};
    ov::genai::GeneratedSequences rejected_candidate{{0, ov::genai::GeneratedSequence(tokens, log_probs)}};

    update_result = m_pipeline.update_request(0, rejected_candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);
    ASSERT_TRUE(m_pipeline.is_waiting(0));
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
