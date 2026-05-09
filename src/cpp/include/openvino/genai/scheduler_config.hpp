// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <sstream>

#include "openvino/genai/cache_eviction.hpp"
#include "openvino/genai/sparse_attention.hpp"

namespace ov::genai {

inline constexpr std::size_t DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL_MULTIPLIER = 8;

struct SchedulerConfig {
    // a maximum number of tokens to batch
    // (in contrast to max_batch_size which combines independent sequences, we consider total amount of tokens in a batch)
    // TODO: benchmark this value and understand a required value to ensure inference is not memory bound
    // When ContinuousBatching is invoked from LLMPipeline (client scenario) by default max_num_batched_tokens is not limited.
    std::size_t max_num_batched_tokens = 256;

    // total number of KV blocks available to scheduler logic
    std::size_t num_kv_blocks = 0;

    // total size of cache in GB
    // When both num_kv_blocks and cache_size are set, num_kv_blocks is used. 
    // When both num_kv_blocks and cache_size are equal to zero dynamic cache allocation is turned on.
    std::size_t cache_size = 0;

    // total number of linear attention blocks available to scheduler logic.
    // Each block holds the full state for one sequence across all linear attention ops.
    // When 0, automatically derived from max_num_seqs if linear attention layers are detected.
    std::size_t num_linear_attention_blocks = 0;

    // Multiplier used to derive the linear-attention checkpoint interval for prefix caching.
    // The internal cache interval is calculated as KV cache block size * cache_interval_multiplier.
    // When unset, the default value 8 is used for hybrid models with prefix caching.
    // Explicit values are supported only for models with linear attention cache inputs.
    // 0 is valid only when prefix caching is disabled.
    std::optional<std::size_t> cache_interval_multiplier = std::nullopt;

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

    std::size_t get_cache_interval(std::size_t kv_block_size) const {
        const std::size_t effective_cache_interval_multiplier =
            cache_interval_multiplier.value_or(DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL_MULTIPLIER);
        OPENVINO_ASSERT(effective_cache_interval_multiplier == 0 ||
                            kv_block_size <= std::numeric_limits<std::size_t>::max() / effective_cache_interval_multiplier,
                        "SchedulerConfig cache_interval_multiplier is too large for KV cache block size. cache_interval_multiplier: ",
                        effective_cache_interval_multiplier,
                        ", KV cache block size: ",
                        kv_block_size);
        return kv_block_size * effective_cache_interval_multiplier;
    }

    void validate() const {
        OPENVINO_ASSERT(!enable_prefix_caching || !cache_interval_multiplier.has_value() || cache_interval_multiplier.value() > 0,
                "SchedulerConfig cache_interval_multiplier must be greater than 0 when prefix caching is enabled");
    }

    bool operator==(const SchedulerConfig& other) const {
        return max_num_batched_tokens == other.max_num_batched_tokens && num_kv_blocks == other.num_kv_blocks &&
               cache_size == other.cache_size && num_linear_attention_blocks == other.num_linear_attention_blocks &&
               dynamic_split_fuse == other.dynamic_split_fuse && use_cache_eviction == other.use_cache_eviction &&
               max_num_seqs == other.max_num_seqs && enable_prefix_caching == other.enable_prefix_caching &&
               cache_interval_multiplier == other.cache_interval_multiplier;
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
        oss << "  num_linear_attention_blocks: " << num_linear_attention_blocks << "\n";
        if (cache_interval_multiplier.has_value()) {
            oss << "  cache_interval_multiplier: " << cache_interval_multiplier.value() << "\n";
        } else {
            oss << "  cache_interval_multiplier: unset\n";
        }
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
