// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>

namespace ov::genai {
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
    GeneratedSequence(const std::vector<int64_t>& generated_token_ids,
                    const std::vector<float>& generated_log_probs,
                    size_t num_processed_tokens = 0,
                    const ov::Tensor& generated_hidden_states = {}) :
        token_ids(generated_token_ids),
        log_probs(generated_log_probs),
        num_processed_tokens(num_processed_tokens),
        hidden_states(generated_hidden_states) {};
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
}
