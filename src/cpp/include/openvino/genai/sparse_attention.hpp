// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ov::genai {

/**
* @brief Configuration struct for the sparse attention prefill functionality.
*/
class SparseAttentionConfig {
public:
    SparseAttentionConfig() = default;

    SparseAttentionConfig(size_t num_last_dense_tokens_) : num_last_dense_tokens(num_last_dense_tokens_) {  }

    /** Number of tokens from the end of the prompt for which full attention across previous KV cache contents
     * will be computed. In contrast, for the rest of the tokens in the prompt only the sparse attention (encompassing first
     * and currently latest blocks) will be computed. Due to the block-wise nature of continuous batching cache management,
     * the actual number of prompt tokens for which the dense attention will be computed may be up to block-size larger than
     * this value (depending on the prompt length and block size).*/
    size_t num_last_dense_tokens = 100;

};

} // namespace ov::genai

