// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>
namespace ov::genai {
struct GeneratedSequence {
    std::vector<int64_t> token_ids;
    std::vector<float> log_probs;

    GeneratedSequence(const std::vector<int64_t>& generated_token_ids,
                    const std::vector<float>& generated_log_probs) :
        token_ids(generated_token_ids),
        log_probs(generated_log_probs) {};
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
class EagleGeneratedSequence : public GeneratedSequence {
public:
    ov::Tensor feature_vector;

    EagleGeneratedSequence(const std::vector<int64_t>& generated_token_ids,
                           const std::vector<float>& generated_log_probs,
                           const ov::Tensor& generated_hidden_states) :
        GeneratedSequence(generated_token_ids, generated_log_probs),
        feature_vector(generated_hidden_states) {};
};

using EagleGeneratedSequences = std::map<uint64_t, EagleGeneratedSequence>;
using EagleGeneratedRequests = std::map<uint64_t, EagleGeneratedSequences>;
}
