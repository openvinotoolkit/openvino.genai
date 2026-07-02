// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <list>
#include <map>
#include <set>
#include <string>

namespace ov::genai {

/**
 * @brief Abstract interface for physical cache management on a device.
 *
 * Each implementation manages a specific cache type (e.g., KV cache, sliding window cache).
 * The interface exposes only cache-type-agnostic operations used by the Scheduler and CacheOrchestrator.
 * Cache-type-specific accessors (e.g., get_key_cache/get_value_cache for KV caches) remain on concrete subclasses.
 */
class ICacheManager {
public:
    virtual ~ICacheManager() = default;

    /**
     * @brief Allocate (or grow) physical cache storage to hold at least @p num_blocks blocks.
     * If enough blocks are already allocated, this is a no-op.
     */
    virtual void allocate_cache_if_needed(size_t num_blocks) = 0;

    /**
     * @brief Copy block contents from source physical blocks to destination physical blocks.
     * Used for copy-on-write when a forked sequence modifies a shared block.
     * @param block_copy_map Map of source block index -> list of destination block indices.
     */
    virtual void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) = 0;

    /**
     * @brief Zero physical cache blocks before their first use by a new sequence.
     * Cache types that do not read recycled blocks as initial state can keep the default no-op.
     * @param block_indices Physical block indices to zero.
     */
    virtual void zero_blocks(const std::set<size_t>& block_indices) {}

    /**
     * @brief Clear all cache storage, resetting to an empty state.
     */
    virtual void clear() = 0;

    /// @return Number of decoder layers managed by this cache manager.
    virtual size_t get_num_layers() const = 0;

    /// @return Number of cache tensors managed by this cache manager.
    virtual size_t get_num_cache_tensors() const = 0;

    /// @return Block size in tokens.
    virtual size_t get_block_size() const = 0;

    /// @return The device on which the cache resides (e.g. "CPU", "GPU").
    virtual std::string get_device() const = 0;

    /// @return Size in bytes of a single block across all layers.
    virtual size_t get_block_size_in_bytes() const = 0;

    /// @return Number of blocks currently allocated in physical storage.
    virtual size_t get_num_allocated_blocks() const = 0;
};

}  // namespace ov::genai
