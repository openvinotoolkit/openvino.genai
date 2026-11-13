// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "sampling/draft_proposal.hpp"

namespace ov::genai {
struct TreeMetaData;

struct GeneratedSequence {
    std::vector<int64_t> token_ids;
    std::vector<float> log_probs;
    // Store the number of tokens that have been processed by the main model when this sequence is generated.
    // This is used to ensure proper alignment between the main model and draft model during speculative decoding.
    size_t num_processed_tokens = 0;
    // Stores the hidden states tensor associated with the generated sequence.
    // This field is used for the "eagle speculative" decoding algorithm,
    // where hidden states are required to efficiently validate and extend speculative tokens.
    // If not using eagle speculative decoding, this field may remain empty.
    ov::Tensor hidden_states;
    std::shared_ptr<const TreeMetaData> tree_metadata;
    // Optional proposal distributions aligned 1:1 with token_ids; null for tokens without one.
    std::vector<DraftProposalPtr> draft_proposals;
    GeneratedSequence(const std::vector<int64_t>& generated_token_ids,
                      const std::vector<float>& generated_log_probs,
                      size_t num_processed_tokens = 0,
                      const ov::Tensor& generated_hidden_states = {},
                      std::shared_ptr<const TreeMetaData> metadata = nullptr,
                      std::vector<DraftProposalPtr> proposals = {})
        : token_ids(generated_token_ids),
          log_probs(generated_log_probs),
          num_processed_tokens(num_processed_tokens),
          hidden_states(generated_hidden_states),
          tree_metadata(std::move(metadata)),
          draft_proposals(std::move(proposals)) {
        OPENVINO_ASSERT(draft_proposals.empty() || draft_proposals.size() == token_ids.size(),
                        "Draft proposals must be empty or aligned with the candidate token IDs.");
    }
};

struct UpdateRequestResult {
    size_t inserted_tokens_cnt, removed_tokens_cnt;

    UpdateRequestResult(size_t to_insert = 0, size_t to_remove = 0) :
        inserted_tokens_cnt(to_insert),
        removed_tokens_cnt(to_remove) {};
};

// { sequence_id : generated_tokens_and_log_probs }
using GeneratedSequences = std::map<uint64_t, GeneratedSequence>;

// { request_id : generated_sequence }
using GeneratedRequests = std::map<uint64_t, GeneratedSequences>;
}  // namespace ov::genai
