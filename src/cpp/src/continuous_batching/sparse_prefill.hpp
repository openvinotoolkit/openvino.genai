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

class SparseAttentionTokenSkipper {
public:
    SparseAttentionTokenSkipper() = delete;
    explicit SparseAttentionTokenSkipper(size_t num_last_dense_tokens) : m_num_last_dense_tokens(num_last_dense_tokens) {}
    std::set<size_t> get_skipped_blocks(const SequenceGroup::CPtr& sequence_group) const;

private:
    size_t m_num_last_dense_tokens;
};

}
