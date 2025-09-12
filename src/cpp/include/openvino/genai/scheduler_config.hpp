// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <sstream>

#include "openvino/genai/cache_eviction.hpp"
#include "openvino/genai/sparse_attention.hpp"

namespace ov::genai {
struct SchedulerConfig {
    // a maximum number of tokens to batch
    // (in contrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
    // TODO: benchmark this value and understand a required value to ensure inference is not memory bound
    // When ContinuousBatching is invoked from LLMPipeline (client scenario) by default max_num_batched_tokens is not limited.
    std::size_t max_num_batched_tokens = 256;

    // total number of KV blocks available to scheduler logic
    std::size_t num_kv_blocks = 0;

    // total size of KV cache in GB
    // When both num_kv_blocks and cache_size are set, num_kv_blocks is used. 
    // When both num_kv_blocks and cache_size are equal to zero dynamic KV-cache allocation is turned on.
    std::size_t cache_size = 0;

    // whether to split prompt / generate to different scheduling phases
    // Allows to process prompt partially in case when batch size is limited. 
    // If dynamic_split_fuse is turned off any prompt that is longer than batch size will lead to error.
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

    // max number of scheduled sequences. 
    // You can think of it as "max batch size" on generate phase, on prompt phase number of scheduled tokens usually more than number of sequences.
    std::size_t max_num_seqs = 256;

    // Enable caching of KV-blocks.
    // When turned on all previously calculated KV-caches are kept in memory for future usages.
    // KV-caches can be overridden if KV-cache limit is reached, but blocks are not released.
    // This results in more RAM usage, maximum RAM usage is determined by cache_size or num_kv_blocks parameters. 
    // When turned off only KV-cache required for batch calculation is kept in memory and
    // when a sequence has finished generation its cache is released.
    // When ContinuousBatching is invoked from LLMPipeline (client scenario) by default prefix caching is turned on.
    bool enable_prefix_caching = false;

    /** Whether to apply block-wise sparse attention to the prefill stage.
     */
    bool use_sparse_attention = false;
    /** Configuration struct for the sparse attention prefill functionality.
     */
    SparseAttentionConfig sparse_attention_config;

    bool operator==(const SchedulerConfig& other) const {
        return max_num_batched_tokens == other.max_num_batched_tokens && num_kv_blocks == other.num_kv_blocks &&
               cache_size == other.cache_size &&
               dynamic_split_fuse == other.dynamic_split_fuse && use_cache_eviction == other.use_cache_eviction &&
               max_num_seqs == other.max_num_seqs && enable_prefix_caching == other.enable_prefix_caching;
    }

    /**
     * Returns a human-readable string representation of the SchedulerConfig.
     * The output is a multi-line string listing each configuration field and its value.
     * This is useful for debugging, logging, or inspecting the current configuration.
     *
     * @return A string describing the current SchedulerConfig in a readable format.
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "SchedulerConfig { \n";
        oss << "  max_num_batched_tokens: " << max_num_batched_tokens << "\n";
        oss << "  num_kv_blocks: " << num_kv_blocks << "\n";
        oss << "  cache_size: " << cache_size << "\n";
        oss << "  dynamic_split_fuse: " << std::boolalpha << dynamic_split_fuse << "\n";
        oss << "  use_cache_eviction: " << std::boolalpha << use_cache_eviction << "\n";
        if (use_cache_eviction) {
            oss << cache_eviction_config.to_string() << "\n";
        }
        oss << "  max_num_seqs: " << max_num_seqs << "\n";
        oss << "  enable_prefix_caching: " << std::boolalpha << enable_prefix_caching << "\n";
        oss << "  use_sparse_attention: " << std::boolalpha << use_sparse_attention << "\n";
        if (use_sparse_attention) {
            oss << sparse_attention_config.to_string() << "\n";
        }
        oss << " }";
        return oss.str();
    }
};
}
