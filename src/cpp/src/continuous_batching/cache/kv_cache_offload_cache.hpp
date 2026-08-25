// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_offload_manager.hpp"

namespace ov::genai {

/**
 * @brief Second-level prefix cache holding KV block contents on disk.
 *
 * Blocks are written when the in-memory prefix cache is about to overwrite them, which is the last
 * point where their contents are still intact. Entries are keyed by the prefix-cache block hash;
 * once the backing file is full the oldest entry is replaced.
 *
 * Offload is best effort: a failed store is reported through the statistics and never propagates to
 * the caller, since losing a cache entry only costs recomputation.
 */
class KVCacheOffloadCache : public IOverwrittenBlockObserver, public IExternalPrefixSource {
public:
    struct Statistics {
        std::size_t num_stored = 0;
        std::size_t num_replaced = 0;
        std::size_t num_already_present = 0;
        std::size_t num_loaded = 0;
        std::size_t num_failed = 0;
    };

    KVCacheOffloadCache(KVCacheManager& cache_manager, std::unique_ptr<KVCacheOffloadManager> backend);

    void on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) override;

    bool contains(std::size_t hash) const override;

    bool load_into(std::size_t hash, std::size_t block_index) override;

    /**
     * @brief Reads the stored contents for @p hash.
     * @return false if @p hash has no entry, in which case @p block_data is left untouched.
     */
    bool read(std::size_t hash, std::vector<uint8_t>& block_data) const;

    std::size_t get_num_entries() const;

    std::size_t get_num_free_slots() const {
        return m_backend->get_num_free_slots();
    }

    Statistics get_statistics() const;

private:
    struct Entry {
        std::size_t slot_id = 0;
        std::list<std::size_t>::iterator order_it;
    };

    /// @return A slot freed by dropping the oldest entry, or std::nullopt when there is nothing to drop.
    std::optional<std::size_t> reclaim_oldest_slot();

    KVCacheManager& m_cache_manager;
    std::unique_ptr<KVCacheOffloadManager> m_backend;
    std::unordered_map<std::size_t, Entry> m_entries;
    // Front is the oldest entry and the first to be replaced when the file is full.
    std::list<std::size_t> m_insertion_order;
    std::vector<uint8_t> m_staging;
    Statistics m_statistics;
    mutable std::mutex m_mutex;
};

}  // namespace ov::genai
