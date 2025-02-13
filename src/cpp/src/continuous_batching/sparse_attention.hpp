// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdlib>
#include <cmath>

#include "sequence_group.hpp"
#include "continuous_batching/attention_output.hpp"
#include "openvino/genai/cache_eviction.hpp"

namespace ov::genai {

/**
* @brief Calculates the set of KV cache logical block IDs that should be skipped from the KV cache block set during the
* next inference for a given sequence group.
*/
class SparseAttentionTokenSkipper {
public:
    SparseAttentionTokenSkipper() = delete;

    /**
    * Constructs the SparseAttentionTokenSkipper.
    * @param num_last_dense_tokens The number of tokens in the end of the prompt phase for which sparse attention should
    * not be applied.
    */
    explicit SparseAttentionTokenSkipper(size_t num_last_dense_tokens) : m_num_last_dense_tokens(num_last_dense_tokens) {}

    /**
    * @param sequence_group A pointer to the sequence group.
    * @return The set of logical block IDs that should be skipped during the next inference for this sequence group.
    */
    std::set<size_t> get_skipped_blocks(const SequenceGroup::CPtr& sequence_group) const;

private:
    size_t m_num_last_dense_tokens;
};

}
