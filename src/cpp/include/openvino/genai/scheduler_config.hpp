// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ov::genai {
struct SchedulerConfig {
    // a maximum number of tokens to batch
    // (in constrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
    std::size_t max_num_batched_tokens = 256;

    // total number of KV blocks available to scheduler logic
    // Note, if it's set to 0, then `cache_size` must be specified
    std::size_t num_kv_blocks = 0;

    // total size of KV cache in GB
    // Note, if it's set to 0, then `num_kv_blocks` must be specified
    std::size_t cache_size = 1;

    // block size for KV cache
    std::size_t block_size = 32;

    // whether to split prompt / generate to different scheduling phases
    // - Dynamic split fuse schdules requests in generation phase first, then
    // schdules requests in prompt phase. If request cannot be fully fit into
    // remaining space of 'max_num_batched_tokens' group, it's scheduled only partially
    // and other tokens can be scheduled only next iterations
    // - vLLM mode priorities requests in prompt phase over requests on generation phase
    bool dynamic_split_fuse = true;

    // Enable caching of KV-blocks.
    // When turned on all previously calculated KV-caches are kept in memory for future usages.
    // KV-caches can be rewritten if KV-cache limit is reached, but blocks are not released.
    // This results in more RAM usage, maximum RAM usage is determined by cache_size or num_kv_blocks parameters.
    // When turned off, only KV-cache required for batch calculation is kept in memory and
    // when a sequence has finished genegartion its KV cache blocks are released.
    bool enable_prefix_caching = false;

    //
    // vLLM-like settings
    //

    // max number of scheduled sequences (you can think of it as "max batch size")
    std::size_t max_num_seqs = 256;
};
}
