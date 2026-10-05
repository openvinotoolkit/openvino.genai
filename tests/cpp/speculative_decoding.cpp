// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include "gtest/gtest.h"

#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "sampling/sampler.hpp"
#include "speculative_decoding/continuous_batching/mtp_strategy.hpp"
#include "speculative_decoding/continuous_batching/pipeline_impl.hpp"
#include "utils.hpp"

namespace {
// Total variation distance between two discrete distributions of equal length.
float total_variation_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float tvd = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        tvd += std::abs(a[i] - b[i]);
    return 0.5f * tvd;
}

// Frequencies of the token ids returned by `draw` over `num_draws` calls.
template <typename Draw>
std::vector<float> empirical_distribution(size_t vocab_size, size_t num_draws, Draw draw) {
    std::vector<size_t> counts(vocab_size, 0);
    for (size_t i = 0; i < num_draws; ++i)
        ++counts.at(static_cast<size_t>(draw()));  // throws for an id outside the vocabulary
    std::vector<float> frequencies(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i)
        frequencies[i] = static_cast<float>(counts[i]) / num_draws;
    return frequencies;
}

// Logits on the m_vector path as the logit processor leaves them: the surviving top_k candidates holding
// scaled raw logits when expf is deferred, otherwise probabilities (top_p truncates without renormalising).
ov::genai::Logits candidate_logits(std::vector<ov::genai::Token> candidates, bool defer_expf) {
    ov::genai::Logits logits(nullptr, candidates.size());
    logits.m_vector = std::move(candidates);
    logits.m_defer_expf = defer_expf;
    return logits;
}

// Softmax over the candidates' logits in double precision, where these magnitudes neither overflow nor underflow.
std::vector<float> reference_softmax(const std::vector<ov::genai::Token>& candidates, size_t vocab_size) {
    double total = 0.0;
    for (const auto& candidate : candidates)
        total += std::exp(static_cast<double>(candidate.m_log_prob));
    std::vector<float> probabilities(vocab_size, 0.0f);
    for (const auto& candidate : candidates) {
        const double probability = std::exp(static_cast<double>(candidate.m_log_prob)) / total;
        probabilities[candidate.m_index] = static_cast<float>(probability);
    }
    return probabilities;
}
}  // namespace

// With q == p the residual is empty, so the replacement is drawn from p itself rather than uniformly.
TEST(SpeculativeResidualSampling, FallsBackToTargetWhenResidualDegenerate) {
    const std::vector<float> p = {0.1f, 0.2f, 0.3f, 0.4f};

    std::mt19937 rng(7);
    const auto empirical = empirical_distribution(p.size(), 40000, [&] {
        return ov::genai::detail::residual_sample(p, p, rng).m_index;
    });

    EXPECT_LT(total_variation_distance(empirical, p), 0.02f);
}

// One speculative step emits tokens distributed exactly as the target p (Leviathan et al. 2023, Theorem 1):
// draw t ~ q, accept it with min(1, p(t) / q(t)), otherwise resample from the residual. The target is on the
// deferred top_k path with logits that overflow a naive expf; the draft holds top_p-truncated probabilities.
TEST(SpeculativeSampling, EmittedTokensFollowTargetDistribution) {
    constexpr size_t vocab_size = 6;
    const std::vector<ov::genai::Token> target_candidates = {{120.0f, 4}, {119.0f, 1}, {117.0f, 3}};
    const auto target = candidate_logits(target_candidates, true);
    const auto draft = candidate_logits({{0.4f, 1}, {0.2f, 3}, {0.2f, 4}, {0.1f, 0}}, false);
    const auto p = ov::genai::detail::materialize_distribution(target, vocab_size);
    const auto q = ov::genai::detail::materialize_distribution(draft, vocab_size);

    std::mt19937 rng(42);
    std::discrete_distribution<int64_t> propose(q.begin(), q.end());
    const auto empirical = empirical_distribution(vocab_size, 200000, [&] {
        const int64_t t = propose(rng);
        const float p_t = ov::genai::detail::get_token_probability(target, t);
        const float q_t = ov::genai::detail::get_token_probability(draft, t);
        if (ov::genai::detail::accept_draft_token(p_t, q_t, rng))
            return t;
        return ov::genai::detail::residual_sample(p, q, rng).m_index;
    });

    // Token 0 is proposed by the draft but removed by the target's top_k, so it must never be emitted.
    EXPECT_EQ(empirical[0], 0.0f);
    EXPECT_LT(total_variation_distance(empirical, reference_softmax(target_candidates, vocab_size)), 0.01f);
}

namespace {
// Samples one token for request 0 (a single sequence, grouped id 0) from logits {0, 1, 2, 3} with T = 0.5,
// top_k = 2 and logprobs = 1. Post-filter, tokens 3 and 2 remain with scaled logits 6 and 4.
ov::genai::Sequence::Ptr sample_one_token(ov::genai::Sampler& sampler) {
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 10;
    config.do_sample = true;
    config.temperature = 0.5f;
    config.top_k = 2;
    config.logprobs = 1;

    std::vector<int64_t> prompt = {7};
    ov::Tensor input_ids(ov::element::i64, {1, 1}, prompt.data());
    auto sequence_group = std::make_shared<ov::genai::SequenceGroup>(0, input_ids, config);
    sequence_group->get_sequences().front()->append_token(1, 0.0f);
    sequence_group->update_processed_tokens_num(1);
    sequence_group->schedule_tokens(1);

    std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
    sampler.sample({sequence_group}, ov::Tensor(ov::element::f32, {1, 1, logits.size()}, logits.data()));
    return sequence_group->get_sequences().front();
}
}  // namespace

// Only the draft sampler stores q(.), so regular generation never keeps a full-vocab vector per token. The draft
// also records each token's log-prob under q(.), because the main sampler reads q(t) from it, even when
// logprobs > 0 makes the sampler report raw full-vocab log-probs.
TEST(SpeculativeSampling, OnlyDraftSamplerRecordsPostFilterDistribution) {
    const std::vector<float> q = {0.0f, 0.0f, 1.0f / (1.0f + std::exp(2.0f)), 1.0f / (1.0f + std::exp(-2.0f))};

    ov::genai::Sampler regular_sampler;
    sample_one_token(regular_sampler);
    EXPECT_TRUE(regular_sampler.extract_draft_distributions(0, 0).empty());

    ov::genai::Sampler draft_sampler;
    draft_sampler.set_speculative_draft(true);
    const auto sequence = sample_one_token(draft_sampler);
    const int64_t token = sequence->get_generated_ids().back();
    ASSERT_TRUE(token == 2 || token == 3) << "token " << token;
    EXPECT_NEAR(std::exp(sequence->get_generated_log_probs().back()), q[token], 1e-5f);

    const auto distributions = draft_sampler.extract_draft_distributions(0, 0);
    ASSERT_EQ(distributions.size(), 1u);
    ASSERT_EQ(distributions[0].size(), q.size());
    for (size_t i = 0; i < q.size(); ++i)
        EXPECT_NEAR(distributions[0][i], q[i], 1e-6f) << "token " << i;
}

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

        void enable_validation_mode() {
            m_is_validation_mode_enabled = true;
        }

        ov::genai::Sampler& get_sampler() {
            return *m_sampler;
        }

        bool is_waiting(uint64_t request_id) const {
            auto request_it = std::find_if(m_requests.begin(), m_requests.end(), [request_id](const ov::genai::SequenceGroup::Ptr& request) {
                return request->get_request_id() == request_id;
            });
            OPENVINO_ASSERT(request_it != m_requests.end(), "Request is not found");
            return (*request_it)->is_waiting();
        }

    };

    class MtpPipelineTestInstance : public ContinuousBatchingPipeline::MtpDecodingImpl {
    public:
        MtpPipelineTestInstance() = default;
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

TEST_F(CBForSDTest, MtpAdmissionRejectsNonGreedyDecoding) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.do_sample = true;

    expect_mtp_request_rejected(pipeline, config, "supports greedy decoding only");
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

    try {
        pipeline.add_request(0, invalid_input, valid_mtp_config());
        FAIL() << "Expected the scalar test tensor to be rejected after admission validation";
    } catch (const ov::Exception& exception) {
        EXPECT_NE(std::string(exception.what()).find("MTP draft input embeds expect shape"), std::string::npos)
            << exception.what();
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

// The main sampler resamples a rejected token from the q(.) it finds at that token's offset from the end of
// the sequence, so the transferred window must line up with the draft tokens just inserted.
TEST_F(CBForSDTest, draft_distributions_follow_inserted_tokens__one_sequence) {
    m_pipeline.enable_validation_mode();
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    // The draft proposed tokens 3 and 4 this round, with one q(.) per proposal in generation order.
    tokens = { 0, 1, 2, 3, 4 };
    log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
    const std::vector<std::vector<float>> draft_distributions = {{0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
                                                                 {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs, 0, {}, nullptr, draft_distributions) }};
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    auto& sampler = m_pipeline.get_sampler();
    EXPECT_EQ(sampler.get_candidate_distribution(0, 0, 1), draft_distributions[1]);
    EXPECT_EQ(sampler.get_candidate_distribution(0, 0, 2), draft_distributions[0]);
    EXPECT_TRUE(sampler.get_candidate_distribution(0, 0, 3).empty());

    // A round without q(.) must not leave the previous window behind.
    tokens.push_back(5);
    log_probs.push_back(0.6f);
    ov::genai::GeneratedSequences candidate_2{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    update_result = m_pipeline.update_request(0, candidate_2, true);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 1);
    EXPECT_TRUE(sampler.get_candidate_distribution(0, 0, 1).empty());
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
