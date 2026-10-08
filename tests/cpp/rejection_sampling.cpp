// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "openvino/op/constant.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"

using namespace ov::genai;

namespace {
SequenceGroup::Ptr make_group(const GenerationConfig& config, uint64_t request_id = 0, size_t prompt_len = 5) {
    std::vector<int64_t> prompt(prompt_len);
    for (size_t i = 0; i < prompt_len; ++i) {
        prompt[i] = static_cast<int64_t>(i);
    }
    ov::Tensor input(ov::element::i64, ov::Shape{1, prompt_len}, prompt.data());
    return std::make_shared<SequenceGroup>(request_id, input, config);
}

// Emulates a processed prompt followed by the main model's token 0, then appends the draft candidates with the
// distributions they were drafted from and schedules them for validation.
Sequence::Ptr add_candidates(SequenceGroup::Ptr group,
                             const std::vector<int64_t>& candidates,
                             const std::vector<DraftProposal>& proposals) {
    auto sequence = group->get_sequences().front();
    sequence->append_token(0, 0.f);
    group->update_processed_tokens_num(group->get_prompt_len());
    for (size_t i = 0; i < candidates.size(); ++i) {
        sequence->append_token(candidates[i], 0.f);
        sequence->set_draft_proposal(sequence->get_generated_len() - 1, proposals[i]);
    }
    group->set_num_validated_tokens(candidates.size());
    group->schedule_tokens(group->get_num_available_tokens_for_batching());
    return sequence;
}

ov::Tensor logits_tensor(std::vector<float>& values, size_t num_rows) {
    return ov::Tensor(ov::element::f32, ov::Shape{num_rows, 1, values.size() / num_rows}, values.data());
}

// Logits on the m_vector path as the logit processor leaves them: the surviving candidates holding scaled raw
// logits when expf is deferred, otherwise probabilities (top_p truncates without renormalising).
Logits candidate_logits(std::vector<Token> candidates, bool defer_expf) {
    Logits logits(nullptr, candidates.size());
    logits.m_vector = std::move(candidates);
    logits.m_defer_expf = defer_expf;
    return logits;
}

float total_variation_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float distance = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += std::abs(a[i] - b[i]);
    }
    return 0.5f * distance;
}

// Frequencies of the token emitted at the validated position when the main model with target p verifies a
// single candidate drawn from q.
std::vector<float> emitted_frequencies(const std::vector<float>& p,
                                       const std::vector<float>& q,
                                       size_t top_k,
                                       size_t trials) {
    std::mt19937 draft_rng(12345);
    std::discrete_distribution<int64_t> draw_from_q(q.begin(), q.end());
    std::vector<float> frequencies(p.size(), 0.0f);
    Sampler sampler;
    for (size_t trial = 0; trial < trials; ++trial) {
        GenerationConfig config;
        config.max_new_tokens = 30;
        config.do_sample = true;
        config.rng_seed = trial + 1;
        if (top_k > 0) {
            config.top_k = top_k;
        }
        auto group = make_group(config, trial, 4);
        auto sequence = add_candidates(group, {draw_from_q(draft_rng)}, {DraftProposal{{}, q}});
        std::vector<float> logits;
        for (float probability : p) {
            logits.push_back(std::log(probability));
        }
        logits.resize(2 * p.size(), 0.0f);
        sampler.sample({group}, logits_tensor(logits, 2), true);
        frequencies.at(static_cast<size_t>(sequence->get_generated_ids().at(1))) += 1.0f / trials;
        sampler.clear_request_info(trial);
    }
    return frequencies;
}
}  // namespace

TEST(RejectionSampling, SamplesExactResidualOutsideRejectedProposalMass) {
    const DraftProposal proposal{{1}, {1.0f}};
    ASSERT_FLOAT_EQ(detail::proposal_probability(proposal, 1), 1.0f);
    ASSERT_FLOAT_EQ(detail::proposal_probability(proposal, 0), 0.0f);

    std::mt19937 rng(0);
    std::vector<float> p{0.25f, 0.75f};
    const auto sampled = detail::sample_residual(Logits(p.data(), p.size()), proposal, rng);
    ASSERT_EQ(sampled.m_index, 0);
    ASSERT_NEAR(std::exp(sampled.m_log_prob), 0.25f, 1e-6f);
}

// With q == p the residual has no mass, so the replacement is drawn from p itself.
TEST(RejectionSampling, FallsBackToTargetWhenResidualIsDegenerate) {
    std::vector<float> p{0.1f, 0.2f, 0.3f, 0.4f};
    const DraftProposal proposal{{}, p};
    std::mt19937 rng(7);
    std::vector<float> frequencies(p.size(), 0.0f);
    constexpr size_t trials = 40000;
    for (size_t trial = 0; trial < trials; ++trial) {
        frequencies.at(detail::sample_residual(Logits(p.data(), p.size()), proposal, rng).m_index) += 1.0f / trials;
    }
    EXPECT_LT(total_variation_distance(frequencies, p), 0.02f);
}

// Every tail token is below half a float ulp of the residual mass before it, so a float sum never reaches the tail.
// The tail holds 1.6% of the residual, about 8 of the 500 draws.
TEST(RejectionSampling, SamplesTheResidualTailOfADenseTarget) {
    constexpr size_t tail_size = 400000;
    constexpr float tail_probability = 2.5e-8f;
    std::vector<float> p(2 + tail_size, tail_probability);
    p[1] = 0.6f;
    p[0] = 1.0f - p[1] - tail_size * tail_probability;
    const DraftProposal proposal{{0}, {1.0f}};
    std::mt19937 rng(11);
    size_t tail_draws = 0;
    for (size_t trial = 0; trial < 500; ++trial) {
        tail_draws += detail::sample_residual(Logits(p.data(), p.size()), proposal, rng).m_index > 1;
    }
    EXPECT_GT(tail_draws, 0u);
}

TEST(RejectionSampling, RejectsInvalidDistributions) {
    EXPECT_THROW(detail::validate_draft_proposal(DraftProposal{{0, 0}, {0.5f, 0.5f}}), ov::Exception);
    EXPECT_THROW(detail::validate_draft_proposal(DraftProposal{{0}, {0.75f}}), ov::Exception);
    EXPECT_THROW(detail::validate_draft_proposal(DraftProposal{{-1}, {1.0f}}), ov::Exception);
    EXPECT_THROW(detail::validate_draft_proposal(DraftProposal{}), ov::Exception);
}

TEST(RejectionSampling, UsesUniformFallbackForInvalidSamplingLogits) {
    float values[] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    const Logits logits(values, 2);

    const auto proposal = detail::make_draft_proposal(logits);
    ASSERT_TRUE(proposal.is_dense());
    ASSERT_EQ(proposal.probabilities.size(), 2);
    EXPECT_FLOAT_EQ(proposal.probabilities[0], 0.5f);
    EXPECT_FLOAT_EQ(proposal.probabilities[1], 0.5f);
    EXPECT_FLOAT_EQ(detail::sampling_probability(logits, 1), 0.5f);
}

// One likely token and a vocabulary-sized tail of equal probabilities: float sums over it drift by more than the
// proposal tolerance, since the rounding of equal terms does not cancel.
TEST(RejectionSampling, NormalizesFullVocabularyProposalsWithEqualTails) {
    constexpr size_t vocab_size = 151936;
    std::vector<float> probabilities(vocab_size, 1.0f);
    probabilities[0] = std::exp(12.25f);
    double total = 0.0;
    for (const float probability : probabilities) {
        total += probability;
    }
    for (auto& probability : probabilities) {
        probability = static_cast<float>(probability / total);
    }
    const Logits logits(probabilities.data(), vocab_size);

    const auto proposal = detail::make_draft_proposal(logits);
    ASSERT_TRUE(proposal.is_dense());
    EXPECT_NO_THROW(detail::validate_draft_proposal(proposal));
    EXPECT_NEAR(detail::sampling_probability(logits, 0), proposal.probabilities[0], 1e-6f);
}

// The same row as a candidate list, as top-p or min-p without top-k can leave it.
TEST(RejectionSampling, NormalizesLargeCandidateProposalsWithEqualTails) {
    constexpr size_t vocab_size = 151936;
    const double total = std::exp(12.25) + static_cast<double>(vocab_size - 1);
    std::vector<Token> candidates{Token(static_cast<float>(std::exp(12.25) / total), 0)};
    for (size_t i = 1; i < vocab_size; ++i) {
        candidates.emplace_back(static_cast<float>(1.0 / total), static_cast<int64_t>(i));
    }
    const Logits logits = candidate_logits(std::move(candidates), false);

    const auto proposal = detail::make_draft_proposal(logits);
    ASSERT_FALSE(proposal.is_dense());
    EXPECT_NO_THROW(detail::validate_draft_proposal(proposal));
    EXPECT_NEAR(detail::sampling_probability(logits, 0), proposal.probabilities[0], 1e-6f);
}

TEST(RejectionSampling, CapturesTheDistributionUsedToSampleDraftTokens) {
    GenerationConfig config;
    config.do_sample = true;
    config.max_new_tokens = 8;
    auto sequence_group = std::make_shared<SequenceGroup>(1, TokenIds{0}, config);
    sequence_group->update_processed_tokens_num(1);
    sequence_group->schedule_tokens(1);

    std::vector<float> values{0.0f, 1.0f, 2.0f};
    Sampler sampler;
    sampler.set_collect_draft_proposals(true);
    sampler.sample({sequence_group}, logits_tensor(values, 1));

    const auto sequence = (*sequence_group)[0];
    ASSERT_EQ(sequence->get_generated_len(), 1);
    const auto& proposal = sequence->get_draft_proposal(0);
    ASSERT_TRUE(proposal.is_dense());
    EXPECT_NO_THROW(detail::validate_draft_proposal(proposal));
    EXPECT_GT(proposal.probabilities[2], proposal.probabilities[1]);
    EXPECT_GT(proposal.probabilities[1], proposal.probabilities[0]);
}

namespace {
// Samples one token for request 0 from logits {0, 1, 2, 3} with T = 0.5, top_k = 2 and logprobs = 1. Post-filter,
// tokens 3 and 2 remain with scaled logits 6 and 4.
Sequence::Ptr sample_one_top_k_token(Sampler& sampler) {
    GenerationConfig config;
    config.max_new_tokens = 10;
    config.do_sample = true;
    config.temperature = 0.5f;
    config.top_k = 2;
    config.logprobs = 1;
    auto group = std::make_shared<SequenceGroup>(0, TokenIds{7}, config);
    group->update_processed_tokens_num(1);
    group->schedule_tokens(1);

    std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
    sampler.sample({group}, logits_tensor(logits, 1));
    return group->get_sequences().front();
}
}  // namespace

// Regular generation keeps no proposals; a draft keeps a sparse q(.) over the candidates surviving top_k.
TEST(RejectionSampling, RecordsSparseProposalsOnlyForDraftPipelines) {
    Sampler regular_sampler;
    EXPECT_TRUE(sample_one_top_k_token(regular_sampler)->get_draft_proposals().empty());

    Sampler draft_sampler;
    draft_sampler.set_collect_draft_proposals(true);
    const auto sequence = sample_one_top_k_token(draft_sampler);
    const auto& proposal = sequence->get_draft_proposal(0);
    ASSERT_FALSE(proposal.is_dense());
    const float q3 = 1.0f / (1.0f + std::exp(-2.0f));
    EXPECT_NEAR(detail::proposal_probability(proposal, 3), q3, 1e-6f);
    EXPECT_NEAR(detail::proposal_probability(proposal, 2), 1.0f - q3, 1e-6f);
    EXPECT_EQ(proposal.token_ids.size(), 2u);
}

// An EAGLE draft samples over its own vocabulary, where draft id i is target id i + d2t[i]. It records q(.) by
// target ids, the space of the emitted token and of the main model's p(.).
TEST(RejectionSampling, RecordsEagleDraftProposalsInTargetVocabulary) {
    const std::vector<int64_t> d2t = {0, 2, 3, 5};  // draft ids 0..3 -> target ids 0, 3, 5, 8
    Sampler draft_sampler;
    draft_sampler.set_d2t_for_decoding(
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{d2t.size()}, d2t));
    draft_sampler.set_collect_draft_proposals(true);
    const auto sequence = sample_one_top_k_token(draft_sampler);

    const int64_t token = sequence->get_generated_ids().back();
    ASSERT_TRUE(token == 5 || token == 8) << "token " << token;
    const auto& proposal = sequence->get_draft_proposal(0);
    const float q8 = 1.0f / (1.0f + std::exp(-2.0f));
    EXPECT_NEAR(detail::proposal_probability(proposal, 8), q8, 1e-6f);
    EXPECT_NEAR(detail::proposal_probability(proposal, 5), 1.0f - q8, 1e-6f);
    EXPECT_FLOAT_EQ(detail::proposal_probability(proposal, 3), 0.0f);
}

TEST(RejectionSampling, RejectsSampledSpeculationWithoutCompleteProposalRows) {
    GenerationConfig config;
    config.do_sample = true;
    config.max_new_tokens = 8;
    auto sequence_group = std::make_shared<SequenceGroup>(1, TokenIds{0}, config);
    (*sequence_group)[0]->append_token(1, 0.0f);
    sequence_group->update_processed_tokens_num(1);
    sequence_group->schedule_tokens(2);
    sequence_group->set_num_validated_tokens(1);

    std::vector<float> values{0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f};
    Sampler sampler;
    EXPECT_THROW(sampler.sample({sequence_group}, logits_tensor(values, 2), true), ov::Exception);
}

// The target is on the deferred top_k path with logits that overflow a naive expf; the draft holds top_p-truncated
// probabilities. The emitted tokens follow the target softmax (Leviathan et al., 2023, Theorem 1).
TEST(RejectionSampling, RecoversTheTargetDistributionOverFilteredCandidates) {
    constexpr size_t vocab_size = 6;
    const std::vector<Token> target_candidates = {{120.0f, 4}, {119.0f, 1}, {117.0f, 3}};
    const auto target = candidate_logits(target_candidates, true);
    const auto proposal =
        detail::make_draft_proposal(candidate_logits({{0.4f, 1}, {0.2f, 3}, {0.2f, 4}, {0.1f, 0}}, false));
    ASSERT_FALSE(proposal.is_dense());
    EXPECT_NO_THROW(detail::validate_draft_proposal(proposal));

    std::vector<float> reference(vocab_size, 0.0f);
    double total = 0.0;
    for (const auto& candidate : target_candidates) {
        total += std::exp(static_cast<double>(candidate.m_log_prob) - 120.0);
    }
    for (const auto& candidate : target_candidates) {
        const double weight = std::exp(static_cast<double>(candidate.m_log_prob) - 120.0);
        reference[candidate.m_index] = static_cast<float>(weight / total);
    }

    std::mt19937 rng(42);
    std::discrete_distribution<size_t> draw_from_q(proposal.probabilities.begin(), proposal.probabilities.end());
    std::vector<float> frequencies(vocab_size, 0.0f);
    constexpr size_t trials = 200'000;
    for (size_t trial = 0; trial < trials; ++trial) {
        const int64_t draft_token = proposal.token_ids[draw_from_q(rng)];
        const auto verdict = detail::verify_draft_token(target, draft_token, proposal, rng);
        frequencies.at(static_cast<size_t>(verdict.token.m_index)) += 1.0f / trials;
    }

    // Token 0 is proposed by the draft but removed by the target's top_k, so it must never be emitted.
    EXPECT_EQ(frequencies[0], 0.0f);
    EXPECT_LT(total_variation_distance(frequencies, reference), 0.01f);
}

// The target accepts candidates 1 and 2, rejects 3 in favour of 4, and the rejected token leaves the KV cache. With
// max_new_tokens = 4 the candidates fill the budget exactly, so the replacement is the last token.
TEST(RejectionSampling, ReplacesRejectedCandidateAndRollsBackTheCache) {
    for (const size_t max_new_tokens : {30u, 4u}) {
        GenerationConfig config;
        config.max_new_tokens = max_new_tokens;
        config.do_sample = true;
        // top_k = 1 makes every target distribution one-hot, so acceptance and resampling are deterministic.
        config.top_k = 1;
        auto group = make_group(config);
        auto sequence = add_candidates(group, {1, 2, 3}, {{{1}, {1.f}}, {{2}, {1.f}}, {{3}, {1.f}}});
        std::vector<float> logits = {
            0, 1.f, 0, 0, 0,
            0, 0, 1.f, 0, 0,
            0, 0, 0, 0, 1.f,
            1.f, 0, 0, 0, 0,
        };
        Sampler sampler;
        sampler.sample({group}, logits_tensor(logits, 4), true);

        EXPECT_EQ(sequence->get_generated_ids(), (TokenIds{0, 1, 2, 4})) << "max_new_tokens " << max_new_tokens;
        EXPECT_EQ(group->get_num_processed_tokens(), 5u + 3u);
        for (size_t i = 0; i < sequence->get_generated_len(); ++i) {
            EXPECT_TRUE(sequence->get_draft_proposal(i).empty()) << "position " << i;
        }
    }
}

// p(1) = 0.6 >= q(1) = 0.5, so candidate 1 is accepted without a random draw, and it reports log p(1).
TEST(RejectionSampling, AcceptedCandidateReportsItsTargetLogProb) {
    GenerationConfig config;
    config.max_new_tokens = 30;
    config.do_sample = true;
    config.top_k = 2;
    auto group = make_group(config);
    auto sequence = add_candidates(group, {1}, {{{}, {0, 0.5f, 0.5f, 0, 0}}});
    std::vector<float> logits = {
        -10.f, std::log(0.6f), std::log(0.4f), -10.f, -10.f,
        0.f, 0.f, 0.f, 5.f, 0.f,
    };
    Sampler sampler;
    sampler.sample({group}, logits_tensor(logits, 2), true);

    ASSERT_GE(sequence->get_generated_len(), 2u);
    ASSERT_EQ(sequence->get_generated_ids()[1], 1);
    EXPECT_NEAR(std::exp(sequence->get_generated_log_probs()[1]), 0.6f, 1e-4f);
}

// With logprobs > 0, an accepted candidate the target would rarely sample itself and the replacement of a rejected
// candidate both report the raw target log-prob of their position, and the score sums the reported log-probs.
TEST(RejectionSampling, ReportsRawTargetLogProbsWhenRequested) {
    GenerationConfig config;
    config.max_new_tokens = 30;
    config.do_sample = true;
    config.top_k = 2;
    config.logprobs = 1;
    auto group = make_group(config);
    // Candidate 1 is accepted, as p(1) = 1 / (1 + e^10) exceeds q(1). Candidate 3 is outside the target's top_k, so it
    // is rejected and replaced by 0 or 1.
    auto sequence = add_candidates(group, {1, 3}, {{{0, 1}, {1.f - 1e-5f, 1e-5f}}, {{3}, {1.f}}});
    constexpr size_t vocab_size = 5;
    std::vector<float> logits = {
        10.f, 0, -10.f, -10.f, -10.f,
        1.f, 0.5f, 0, 0, 0,
        0, 0, 0, 0, 0,
    };
    Sampler sampler;
    sampler.sample({group}, logits_tensor(logits, 3), true);

    const auto raw_log_prob = [&](size_t position, int64_t token) {
        float sum = 0.f;
        for (size_t i = 0; i < vocab_size; ++i) {
            sum += std::exp(logits[position * vocab_size + i]);
        }
        return logits[position * vocab_size + token] - std::log(sum);
    };
    const auto& ids = sequence->get_generated_ids();
    const auto& log_probs = sequence->get_generated_log_probs();
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[1], 1);
    ASSERT_TRUE(ids[2] == 0 || ids[2] == 1) << "replacement " << ids[2];
    EXPECT_NEAR(log_probs[1], raw_log_prob(0, 1), 1e-4f);
    EXPECT_NEAR(log_probs[2], raw_log_prob(1, ids[2]), 1e-4f);
    EXPECT_NEAR(sequence->get_cumulative_log_prob(), log_probs[0] + log_probs[1] + log_probs[2], 1e-4f);
}

// The draft logits may be wider than the target's, e.g. by a padding row as for Qwen2.5-0.5B and Qwen2.5-7B, so q(.)
// covers an id the target cannot emit, which has p = 0. They may also be narrower, so target ids beyond them have
// q = 0.
TEST(RejectionSampling, ToleratesDraftLogitsOfAnotherWidth) {
    const std::vector<DraftProposal> proposals = {{{}, {0, 0, 0, 0.5f, 0, 0.5f}}, {{}, {0, 0, 0, 1.f}}};
    for (const auto& proposal : proposals) {
        GenerationConfig config;
        config.max_new_tokens = 30;
        config.do_sample = true;
        config.top_k = 1;
        auto group = make_group(config);
        auto sequence = add_candidates(group, {3}, {proposal});
        std::vector<float> logits = {
            0, 0, 0, 0, 1.f,
            1.f, 0, 0, 0, 0,
        };
        Sampler sampler;
        EXPECT_NO_THROW(sampler.sample({group}, logits_tensor(logits, 2), true));
        EXPECT_EQ(sequence->get_generated_ids(), (TokenIds{0, 4})) << "draft width " << proposal.probabilities.size();
    }
}

// The token emitted for a validated position follows the target distribution p, whatever q is.
TEST(RejectionSampling, EmittedTokensFollowTheTargetDistribution) {
    const std::vector<float> p{0.1f, 0.2f, 0.3f, 0.4f};
    const auto frequencies = emitted_frequencies(p, {0.4f, 0.3f, 0.2f, 0.1f}, 0, 20000);
    for (size_t token = 0; token < p.size(); ++token) {
        EXPECT_NEAR(frequencies[token], p[token], 0.02f) << "token " << token;
    }
}

TEST(RejectionSampling, EmittedTokensFollowTheTopKTargetDistribution) {
    const std::vector<float> p{0.1f, 0.2f, 0.3f, 0.4f};
    const auto frequencies = emitted_frequencies(p, {0.4f, 0.3f, 0.2f, 0.1f}, 3, 20000);
    const std::vector<float> expected{0.0f, 0.2f / 0.9f, 0.3f / 0.9f, 0.4f / 0.9f};
    for (size_t token = 0; token < p.size(); ++token) {
        EXPECT_NEAR(frequencies[token], expected[token], 0.02f) << "token " << token;
    }
}

// Candidate 1 accepted at the first position is history for the second: presence_penalty turns logit 5 of token 1
// into -5 there, so p(1) = 0 and the repeated candidate is replaced by 2, as in target-only decoding.
TEST(RejectionSampling, PenalizesCandidatesAcceptedEarlierInTheWindow) {
    GenerationConfig config;
    config.max_new_tokens = 30;
    config.do_sample = true;
    config.top_k = 1;
    config.presence_penalty = 10.0f;
    auto group = make_group(config);
    auto sequence = add_candidates(group, {1, 1}, {{{1}, {1.f}}, {{1}, {1.f}}});
    std::vector<float> logits = {
        0, 5.f, 0, 0, 0,
        0, 5.f, 4.f, 0, 0,
        5.f, 0, 0, 0, 0,
    };
    Sampler sampler;
    sampler.sample({group}, logits_tensor(logits, 3), true);

    EXPECT_EQ(sequence->get_generated_ids(), (TokenIds{0, 1, 2}));
    EXPECT_EQ(group->get_num_processed_tokens(), 5u + 2u);
}

// An accepted stop token ends generation and discards the candidates drafted after it.
TEST(RejectionSampling, AcceptedStopTokenDiscardsTrailingCandidates) {
    GenerationConfig config;
    config.max_new_tokens = 30;
    config.do_sample = true;
    config.top_k = 1;
    config.stop_token_ids = {4};
    auto group = make_group(config);
    auto sequence = add_candidates(group, {1, 4, 2}, {{{1}, {1.f}}, {{4}, {1.f}}, {{2}, {1.f}}});
    std::vector<float> logits = {
        0, 5.f, 0, 0, 0,
        0, 0, 0, 0, 5.f,
        0, 0, 5.f, 0, 0,
        5.f, 0, 0, 0, 0,
    };
    Sampler sampler;
    sampler.sample({group}, logits_tensor(logits, 4), true);

    EXPECT_EQ(sequence->get_generated_ids(), (TokenIds{0, 1, 4}));
    EXPECT_TRUE(sequence->has_finished());
}
