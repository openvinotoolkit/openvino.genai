// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include "gtest/gtest.h"

#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "speculative_decoding/continuous_batching/mtp_strategy.hpp"
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

        void register_generated_tokens(uint64_t request_id, const std::vector<int64_t>& token_ids) {
            auto& logit_processor = m_sampler->get_logit_processor(request_id);
            for (auto token_id : token_ids) {
                logit_processor.register_new_generated_token(token_id);
            }
        }

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

        bool can_publish_kv_only_completed_blocks() const {
            return _can_publish_kv_only_completed_blocks();
        }

    };

    class MtpPipelineTestInstance : public ContinuousBatchingPipeline::MtpDecodingImpl {
    public:
        MtpPipelineTestInstance() = default;
        using MtpDecodingImpl::make_draft_generation_config;
        using MtpDecodingImpl::make_draft_sampling_params;
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

TEST(SDPerModelsPerfMetrics, DraftStepMetricsAreEvaluatedAndAccumulated) {
    using Metrics = ov::genai::SDPerModelsPerfMetrics;
    for (auto part : {&Metrics::draft_step_metrics, &Metrics::draft_markov_head_metrics, &Metrics::draft_sampling_metrics}) {
        auto make_metrics = [part](float third_step_us) {
            Metrics metrics;
            auto& raw_metrics = (metrics.*part).raw_metrics;
            raw_metrics.m_durations = {ov::genai::MicroSeconds(5000.0f),
                                       ov::genai::MicroSeconds(4000.0f),
                                       ov::genai::MicroSeconds(third_step_us)};
            raw_metrics.m_batch_sizes = {7, 7, 7};
            raw_metrics.m_inference_durations = {ov::genai::MicroSeconds(9000.0f + third_step_us)};
            return metrics;
        };

        auto first = make_metrics(3000.0f);
        EXPECT_FLOAT_EQ((first.*part).get_latency().mean, 3.0f);

        auto accumulated = make_metrics(3000.0f) + make_metrics(1000.0f);
        accumulated.evaluate_statistics();
        EXPECT_EQ((accumulated.*part).raw_metrics.m_durations.size(), 6);
        EXPECT_FLOAT_EQ((accumulated.*part).get_latency().mean, 2.0f);
    }
}

TEST(SDPerModelsPerfMetrics, DraftOverheadDiagnosticsReturnNanWithoutDenominator) {
    ov::genai::SDPerModelsPerfMetrics metrics;

    EXPECT_TRUE(std::isnan(metrics.get_draft_processed_to_candidate_ratio()));
    EXPECT_TRUE(std::isnan(metrics.get_draft_to_main_inference_duration_ratio()));
}

TEST_F(CBForSDTest, DraftPipelineKeepsKvOnlyCompletedBlocksUnpublished) {
    EXPECT_FALSE(m_pipeline.can_publish_kv_only_completed_blocks());
}

TEST_F(CBForSDTest, MtpDraftConfigWithoutTokenLimitIgnoresEos) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 2;
    config.stop_strings = {"stop"};

    const auto draft_config = MtpPipelineTestInstance::make_draft_generation_config(config);

    EXPECT_EQ(config.max_new_tokens, SIZE_MAX);
    EXPECT_EQ(config.max_length, SIZE_MAX);
    EXPECT_EQ(draft_config.max_new_tokens, SIZE_MAX - 1);
    EXPECT_EQ(draft_config.max_length, SIZE_MAX);
    EXPECT_TRUE(draft_config.ignore_eos);
    EXPECT_TRUE(draft_config.stop_strings.empty());
    EXPECT_NO_THROW(draft_config.validate());
}

TEST_F(CBForSDTest, MtpDraftConfigPreservesExplicitTokenLimit) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 2;
    config.max_new_tokens = 200;
    const auto draft_config = MtpPipelineTestInstance::make_draft_generation_config(config);

    EXPECT_EQ(draft_config.max_new_tokens, 200);
    EXPECT_NO_THROW(draft_config.validate());

    config.max_new_tokens = SIZE_MAX;
    config.max_length = 512;
    const auto draft_length_config = MtpPipelineTestInstance::make_draft_generation_config(config);

    EXPECT_EQ(draft_length_config.max_new_tokens, SIZE_MAX);
    EXPECT_EQ(draft_length_config.max_length, 512);
    EXPECT_NO_THROW(draft_length_config.validate());
}

// The draft copy of a request samples from a stream independent of the main request's, whose draws accept or reject
// the drafted tokens. FastDraft and EAGLE3 use make_draft_sampling_params, MTP make_draft_generation_config.
TEST_F(CBForSDTest, DraftConfigsSampleFromAnIndependentStream) {
    ov::genai::GenerationConfig config;
    config.rng_seed = 42;
    const size_t draft_seed = ov::genai::detail::proposal_rng_seed(config.rng_seed);

    EXPECT_NE(draft_seed, config.rng_seed);
    EXPECT_EQ(MtpPipelineTestInstance::make_draft_sampling_params(config).rng_seed, draft_seed);
    EXPECT_EQ(MtpPipelineTestInstance::make_draft_generation_config(config).rng_seed, draft_seed);
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

namespace {
template <typename Pipeline>
void expect_mtp_request_rejected(Pipeline& pipeline,
                                 const ov::genai::GenerationConfig& config,
                                 const std::string& expected_message) {
    // The default-constructed pipeline and empty tensor are deliberate: unsupported configurations
    // must be rejected before add_request touches model state or validates its input tensor.
    try {
        pipeline.add_request(0, ov::Tensor{}, config);
        FAIL() << "Expected MTP request to be rejected at admission";
    } catch (const ov::Exception& exception) {
        EXPECT_NE(std::string(exception.what()).find(expected_message), std::string::npos)
            << exception.what();
    }
}

ov::genai::GenerationConfig valid_mtp_config() {
    auto config = ov::genai::utils::get_greedy_config();
    config.num_assistant_tokens = 1;
    return config;
}
}  // namespace

TEST_F(CBForSDTest, MtpAdmissionRejectsConfidenceThreshold) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.assistant_confidence_threshold = 0.5f;

    expect_mtp_request_rejected(pipeline, config, "assistant_confidence_threshold must be 0.f");
}

TEST_F(CBForSDTest, MtpAdmissionRejectsTreeSearch) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.tree_depth = 2;

    expect_mtp_request_rejected(pipeline, config, "does not support tree search");
}

TEST_F(CBForSDTest, MtpAdmissionRejectsBeamSearch) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.num_beams = 2;

    expect_mtp_request_rejected(pipeline, config, "supports greedy and multinomial decoding only");
}

TEST_F(CBForSDTest, MtpAdmissionRejectsParallelSampling) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.num_return_sequences = 2;

    expect_mtp_request_rejected(pipeline, config, "num_return_sequences must be 1");
}

TEST_F(CBForSDTest, MtpAdmissionRejectsZeroAssistantTokens) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.num_assistant_tokens = 0;

    expect_mtp_request_rejected(pipeline, config, "num_assistant_tokens > 0");
}

TEST_F(CBForSDTest, MtpSupportedConfigPassesAdmissionValidation) {
    MtpPipelineTestInstance pipeline;
    ov::Tensor invalid_input(ov::element::f32, ov::Shape{});
    auto sampled_config = valid_mtp_config();
    sampled_config.do_sample = true;
    sampled_config.top_k = 20;

    for (const auto& config : {valid_mtp_config(), sampled_config}) {
        try {
            pipeline.add_request(0, invalid_input, config);
            FAIL() << "Expected the scalar test tensor to be rejected after admission validation";
        } catch (const ov::Exception& exception) {
            EXPECT_NE(std::string(exception.what()).find("MTP draft input embeds expect shape"), std::string::npos)
                << exception.what();
        }
    }
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

// The main sampler verifies each candidate against the q(.) stored at its position, so proposals must stay attached
// to the tokens they were drafted for across update rounds.
TEST_F(CBForSDTest, draft_proposals_follow_inserted_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    ASSERT_EQ(m_pipeline.update_request(0, candidate, false).inserted_tokens_cnt, 3);

    // The draft proposed tokens 3 and 4 this round.
    tokens = { 0, 1, 2, 3, 4 };
    log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
    const auto q3 = std::make_shared<const ov::genai::DraftProposal>(ov::genai::DraftProposal{{3, 4}, {0.75f, 0.25f}});
    const auto q4 = std::make_shared<const ov::genai::DraftProposal>(ov::genai::DraftProposal{{}, {0, 0, 0, 0, 1.f}});
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs, 0, {}, nullptr, {nullptr, nullptr, nullptr, q3, q4}) }};
    ASSERT_EQ(m_pipeline.update_request(0, candidate_1, false).inserted_tokens_cnt, 2);

    auto proposals = m_pipeline.get_generated_requests().at(0).at(0).draft_proposals;
    ASSERT_EQ(proposals, (std::vector<ov::genai::DraftProposalPtr>{nullptr, nullptr, nullptr, q3, q4}));

    // A round without proposals leaves the pending ones in place.
    tokens.push_back(5);
    log_probs.push_back(0.6f);
    ov::genai::GeneratedSequences candidate_2{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    ASSERT_EQ(m_pipeline.update_request(0, candidate_2, false).inserted_tokens_cnt, 1);
    proposals = m_pipeline.get_generated_requests().at(0).at(0).draft_proposals;
    ASSERT_EQ(proposals, (std::vector<ov::genai::DraftProposalPtr>{nullptr, nullptr, nullptr, q3, q4, nullptr}));

    // Synchronizing a draft sequence with the main model's tokens commits them, so their proposals are dropped.
    ASSERT_EQ(m_pipeline.update_request(0, candidate_2, true).inserted_tokens_cnt, 0);
    EXPECT_TRUE(m_pipeline.get_generated_requests().at(0).at(0).draft_proposals.empty());
}

TEST_F(CBForSDTest, rejects_invalid_draft_proposal__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0 };
    std::vector<float> log_probs = { 0.1f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    ASSERT_EQ(m_pipeline.update_request(0, candidate, false).inserted_tokens_cnt, 1);

    tokens = { 0, 1 };
    log_probs = { 0.1f, 0.2f };
    const auto incomplete = std::make_shared<const ov::genai::DraftProposal>(ov::genai::DraftProposal{{1}, {0.5f}});
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs, 0, {}, nullptr, {nullptr, incomplete}) }};
    EXPECT_THROW(m_pipeline.update_request(0, candidate_1, false), ov::Exception);
}

TEST_F(CBForSDTest, dflash_candidate_update_without_logit_processor_update__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> target_seed_tokens = {10, 11};
    std::vector<float> target_seed_log_probs = {0.1f, 0.2f};
    ov::genai::GeneratedSequences target_seed{
        {0, ov::genai::GeneratedSequence(target_seed_tokens, target_seed_log_probs)}
    };
    auto update_result = m_pipeline.update_request(0, target_seed, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    std::vector<int64_t> draft_candidate_tokens = {10, 11, 20, 21, 22};
    std::vector<float> draft_candidate_log_probs = {0.0f, 0.0f, 0.3f, 0.4f, 0.5f};
    ov::genai::GeneratedSequences draft_candidate{
        {0, ov::genai::GeneratedSequence(draft_candidate_tokens, draft_candidate_log_probs)}
    };
    update_result = m_pipeline.update_request(0, draft_candidate, false);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    auto after_draft_update = m_pipeline.get_generated_requests();
    std::vector<float> expected_draft_log_probs = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    ASSERT_EQ(after_draft_update.at(0).at(0).token_ids, draft_candidate_tokens);
    ASSERT_EQ(after_draft_update.at(0).at(0).log_probs, expected_draft_log_probs);
    m_pipeline.register_generated_tokens(0, {20, 21, 22});

    std::vector<int64_t> validated_tokens = {10, 11, 20, 99};
    std::vector<float> validated_log_probs = {0.0f, 0.0f, 0.3f, 0.9f};
    ov::genai::GeneratedSequences validated_target{
        {0, ov::genai::GeneratedSequence(validated_tokens, validated_log_probs)}
    };
    update_result = m_pipeline.update_request(0, validated_target, false);
    ASSERT_EQ(update_result.removed_tokens_cnt, 2);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 1);

    auto after_validation_update = m_pipeline.get_generated_requests();
    std::vector<float> expected_validated_log_probs = {0.1f, 0.2f, 0.3f, 0.9f};
    ASSERT_EQ(after_validation_update.at(0).at(0).token_ids, validated_tokens);
    ASSERT_EQ(after_validation_update.at(0).at(0).log_probs, expected_validated_log_probs);
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
