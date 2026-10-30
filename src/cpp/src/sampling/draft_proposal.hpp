// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace ov::genai {

// Post-filter distribution q(.) a draft token was sampled from, in the target vocabulary. It is sparse when
// token_ids is populated, otherwise dense with probabilities indexed by token id. An empty proposal means no
// distribution was recorded.
struct DraftProposal {
    std::vector<int64_t> token_ids;
    std::vector<float> probabilities;

    bool empty() const {
        return probabilities.empty();
    }

    bool is_dense() const {
        return token_ids.empty() && !probabilities.empty();
    }
};

// Proposals are immutable once recorded and travel by pointer between the draft and main pipelines.
using DraftProposalPtr = std::shared_ptr<const DraftProposal>;

namespace detail {
inline constexpr uint32_t PROPOSAL_RNG_SEED_SALT = 0x9E3779B9U;

// Seed of the draft sampler. It differs from the main sampler's seed so that the uniform draws accepting
// draft tokens are independent of the draws that proposed them.
inline std::mt19937::result_type proposal_rng_seed(size_t generation_seed) {
    return static_cast<std::mt19937::result_type>(generation_seed) ^ PROPOSAL_RNG_SEED_SALT;
}
}  // namespace detail

}  // namespace ov::genai
