// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <random>

#include "sampling/draft_proposal.hpp"
#include "sampling/logit_transformers.hpp"
#include "sampling/sampling_distribution.hpp"

namespace ov::genai::detail {

// The main model either accepts a candidate, which keeps its token, or replaces it with the token it emits at the
// candidate's position.
struct CandidateVerdict {
    Token token;
    bool accepted = false;
};

// Greedy decoding and prompt lookup: the candidate is accepted if it is the token the main model sampled at its
// position. For prompt lookup, whose q(.) is one-hot on the candidate, this is the speculative sampling test.
inline CandidateVerdict match_candidate(int64_t candidate, const Token& sampled_token) {
    return {sampled_token, sampled_token.m_index == candidate};
}

// Speculative sampling (Leviathan et al., 2023): accepts draft_token with probability min(1, p / q), otherwise
// replaces it with a residual sample, so the emitted token follows the target distribution p exactly. p is the
// distribution the main model samples from after its logit processor, q the proposal the draft recorded.
CandidateVerdict verify_draft_token(const Logits& target_logits,
                                    int64_t draft_token,
                                    const DraftProposal& proposal,
                                    std::mt19937& rng_engine);

// Draws from the residual max(0, p - q) normalized. Draft ids outside the target candidates have p = 0 and target
// ids absent from q have q = 0. Falls back to p when the residual has no mass, e.g. q == p up to rounding.
Token sample_residual(const SamplingDistribution& target, const DraftProposal& proposal, std::mt19937& rng_engine);

// The proposal a draft records with a sampled token: the distribution it drew the token from, sparse over the
// candidates with mass when the logit processor selected them, dense otherwise. With d2t (EAGLE), draft id i is
// recorded as target id i + d2t[i].
DraftProposal make_draft_proposal(const SamplingDistribution& distribution, const int64_t* d2t = nullptr);

// q(token_id); 0 for ids outside the proposal.
float proposal_probability(const DraftProposal& proposal, int64_t token_id);

// Checks a proposal that reaches the main pipeline: it is not empty and a sparse one has a probability per id.
// Debug builds also check that it is a distribution over non-negative ids; the draft sampler builds every proposal
// from a normalized distribution, so release builds skip that pass. Sparse ids are distinct by construction.
void validate_draft_proposal(const DraftProposal& proposal);

}  // namespace ov::genai::detail
