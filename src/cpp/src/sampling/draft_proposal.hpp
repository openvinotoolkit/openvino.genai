// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace ov::genai {

// Post-filter distribution q(.) a draft token was sampled from, in the target vocabulary. It is sparse when
// token_ids is populated, otherwise dense with probabilities indexed by token id. An empty proposal means no
// distribution was recorded. Sparse ids must be distinct.
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

}  // namespace ov::genai
