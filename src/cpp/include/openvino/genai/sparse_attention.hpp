// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ov::genai {

class SparseAttentionConfig {
public:
    SparseAttentionConfig() = default;

    SparseAttentionConfig(size_t num_last_dense_tokens_) : num_last_dense_tokens(num_last_dense_tokens_) {  }

    size_t num_last_dense_tokens = 100;

};

} // namespace ov::genai

