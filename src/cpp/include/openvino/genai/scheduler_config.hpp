// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include "cache_eviction.hpp"

namespace ov::genai {
struct SchedulerConfig {
    // a maximum number of tokens to batch
    // (in contrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
    // TODO: benchmark this value and understand a required value to ensure inference is not memory bound
    std::size_t max_num_batched_tokens = 256;

    // total number of KV blocks available to scheduler logic
    std::size_t num_kv_blocks = 0;

    // total size of KV cache in GB
    std::size_t cache_size = 0;

    // whether to split prompt / generate to different scheduling phases
    bool dynamic_split_fuse = true;


    /**
     * Whether to use cache eviction for all sequences processed by this pipeline. When cache eviction is enabled,
     * the per-sequence KV cache usage is capped by a user-configurable value, leading to memory savings at cost
     * to generation quality.
     */
    bool use_cache_eviction = false;

    /**
     * Configuration struct for the cache eviction algorithm. Setting this has effect only if `use_cache_eviction` is
     * set to `true`.
     */
    CacheEvictionConfig cache_eviction_config;

    //
    // vLLM-like settings
    //

    // max number of scheduled sequences (you can think of it as "max batch size")
    std::size_t max_num_seqs = 256;

    // Enable caching of KV-blocks.
    // When turned on all previously calculated KV-caches are kept in memory for future usages.
    // KV-caches can be overridden if KV-cache limit is reached, but blocks are not released.
    // This results in more RAM usage, maximum RAM usage is determined by cache_size or num_kv_blocks parameters. 
    // When turned off only KV-cache required for batch calculation is kept in memory and
    // when a sequence has finished generation its cache is released.
    bool enable_prefix_caching = false;

    bool operator==(const SchedulerConfig& other) const {
        return max_num_batched_tokens == other.max_num_batched_tokens && num_kv_blocks == other.num_kv_blocks &&
               cache_size == other.cache_size &&
               dynamic_split_fuse == other.dynamic_split_fuse && use_cache_eviction == other.use_cache_eviction &&
               max_num_seqs == other.max_num_seqs && enable_prefix_caching == other.enable_prefix_caching;
    }
};
}
