// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>

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

inline bool operator==(const GeneratedRequests& lhs, const GeneratedRequests& rhs) {
    for (const auto& l_req : lhs) {
        const auto& request_id = l_req.first;
        if (rhs.count(request_id) == 0) {
            return false;
        }
        const auto& r_req = rhs.at(request_id);
        for (const auto& l_seq : l_req.second) {
            const auto& sequence_id = l_seq.first;
            if (r_req.count(sequence_id) == 0) {
                return false;
            }
            const auto& r_seq = r_req.at(sequence_id);

            const auto& l_token_ids = l_seq.second.token_ids;
            const auto& r_token_ids = r_seq.token_ids;

            if (l_token_ids != r_token_ids) {
                return false;
            }

            const auto& l_log_probs = l_seq.second.log_probs;
            const auto& r_log_probs = r_seq.log_probs;
            if (l_log_probs != r_log_probs) {
                return false;
            }
        }
    }
    return true;
}

inline bool operator!=(const GeneratedRequests& lhs, const GeneratedRequests& rhs) {
    return !(lhs == rhs);
}
}
