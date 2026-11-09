// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sampling/candidates_validation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov::genai::detail {

namespace {
// Accepts with probability min(1, p / q): always when p >= q, never when p = 0, and otherwise when u * q < p for a
// uniform u in [0, 1), which avoids a division.
bool accept_draft_token(float target_probability, float draft_probability, std::mt19937& rng_engine) {
    if (target_probability <= 0.0f) {
        return false;
    }
    if (target_probability >= draft_probability) {
        return true;
    }
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    return uniform(rng_engine) * draft_probability < target_probability;
}
}  // namespace

CandidateVerdict verify_draft_token(const Logits& target_logits,
                                    int64_t draft_token,
                                    const DraftProposal& proposal,
                                    std::mt19937& rng_engine) {
    OPENVINO_ASSERT(!proposal.empty(), "Sampled speculative candidates require a proposal distribution.");
    const float draft_probability = proposal_probability(proposal, draft_token);
    OPENVINO_ASSERT(draft_probability > 0.0f, "Selected draft token is absent from its proposal distribution.");
    const SamplingDistribution target(target_logits);
    const std::optional<size_t> index = target.find(draft_token);
    const float target_probability = index ? target.probability(*index) : 0.0f;
    if (accept_draft_token(target_probability, draft_probability, rng_engine)) {
        return {Token(target.log_probability(*index), draft_token), true};
    }
    return {sample_residual(target, proposal, rng_engine), false};
}

Token sample_residual(const SamplingDistribution& target, const DraftProposal& proposal, std::mt19937& rng_engine) {
    std::vector<float> residual(target.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        residual[i] = target.probability(i);
    }
    if (!target.is_sparse()) {
        // Only q's support changes p, and draft ids beyond the target logits carry no target mass.
        if (proposal.is_dense()) {
            const size_t shared_size = std::min(residual.size(), proposal.probabilities.size());
            for (size_t token_id = 0; token_id < shared_size; ++token_id) {
                residual[token_id] = std::max(0.0f, residual[token_id] - proposal.probabilities[token_id]);
            }
        } else {
            for (size_t index = 0; index < proposal.token_ids.size(); ++index) {
                const int64_t token_id = proposal.token_ids[index];
                if (token_id >= 0 && static_cast<size_t>(token_id) < residual.size()) {
                    residual[token_id] = std::max(0.0f, residual[token_id] - proposal.probabilities[index]);
                }
            }
        }
    } else if (proposal.is_dense()) {
        for (size_t i = 0; i < residual.size(); ++i) {
            residual[i] = std::max(0.0f, residual[i] - proposal_probability(proposal, target.token_id(i)));
        }
    } else {
        std::unordered_map<int64_t, float> draft_probabilities;
        draft_probabilities.reserve(proposal.token_ids.size());
        for (size_t index = 0; index < proposal.token_ids.size(); ++index) {
            draft_probabilities.emplace(proposal.token_ids[index], proposal.probabilities[index]);
        }
        for (size_t i = 0; i < residual.size(); ++i) {
            const auto it = draft_probabilities.find(target.token_id(i));
            if (it != draft_probabilities.end()) {
                residual[i] = std::max(0.0f, residual[i] - it->second);
            }
        }
    }

    // Summed in double like the sampling total: a float sum drops tail tokens smaller than half an ulp of the sum
    // so far.
    double residual_sum = std::accumulate(residual.begin(), residual.end(), 0.0);
    if (!(residual_sum > 0.0 && std::isfinite(residual_sum))) {
        for (size_t i = 0; i < residual.size(); ++i) {
            residual[i] = target.probability(i);
        }
        residual_sum = std::accumulate(residual.begin(), residual.end(), 0.0);
    }
    std::uniform_real_distribution<double> uniform(0.0, residual_sum);
    const double sample = uniform(rng_engine);
    // The draw can round up to residual_sum, so the scan must end on a token with residual mass.
    double cumulative = 0.0;
    size_t selected = 0;
    for (size_t i = 0; i < residual.size(); ++i) {
        if (residual[i] <= 0.0f) {
            continue;
        }
        selected = i;
        cumulative += residual[i];
        if (sample < cumulative) {
            break;
        }
    }
    return Token(target.log_probability(selected), target.token_id(selected));
}

DraftProposal make_draft_proposal(const SamplingDistribution& distribution, const int64_t* d2t) {
    DraftProposal proposal;
    if (!distribution.is_sparse() && d2t == nullptr) {
        proposal.probabilities.resize(distribution.size());
        for (size_t i = 0; i < distribution.size(); ++i) {
            proposal.probabilities[i] = distribution.probability(i);
        }
        return proposal;
    }
    for (size_t i = 0; i < distribution.size(); ++i) {
        const float probability = distribution.probability(i);
        if (probability <= 0.0f) {
            continue;
        }
        int64_t token_id = distribution.token_id(i);
        if (d2t != nullptr) {
            token_id += d2t[token_id];
        }
        proposal.token_ids.push_back(token_id);
        proposal.probabilities.push_back(probability);
    }
    return proposal;
}

float proposal_probability(const DraftProposal& proposal, int64_t token_id) {
    if (proposal.is_dense()) {
        return token_id >= 0 && static_cast<size_t>(token_id) < proposal.probabilities.size()
                   ? proposal.probabilities[static_cast<size_t>(token_id)]
                   : 0.0f;
    }
    for (size_t index = 0; index < proposal.token_ids.size(); ++index) {
        if (proposal.token_ids[index] == token_id) {
            return proposal.probabilities[index];
        }
    }
    return 0.0f;
}

void validate_draft_proposal(const DraftProposal& proposal) {
    OPENVINO_ASSERT(!proposal.empty(), "Sampled speculative candidates require a proposal distribution.");
    OPENVINO_ASSERT(proposal.is_dense() || proposal.token_ids.size() == proposal.probabilities.size(),
                    "Sparse draft proposal IDs and probabilities must stay aligned.");
#ifndef NDEBUG
    for (const auto token_id : proposal.token_ids) {
        OPENVINO_ASSERT(token_id >= 0, "Sparse draft proposal contains a negative token ID.");
    }
    double probability_sum = 0.0;
    for (const auto probability : proposal.probabilities) {
        OPENVINO_ASSERT(probability >= 0.0f && std::isfinite(probability),
                        "Draft proposal probabilities must be finite and non-negative.");
        probability_sum += probability;
    }
    OPENVINO_ASSERT(std::isfinite(probability_sum) && std::abs(probability_sum - 1.0) <= 1e-3,
                    "Draft proposal probabilities must sum to one.");
#endif
}

}  // namespace ov::genai::detail
