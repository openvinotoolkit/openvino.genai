// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
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
 * point where their contents are still intact. The contents are copied out synchronously at that
 * moment and reach the file on a background thread, so the allocation path does not wait for disk.
 * A queued block is served straight from its staging copy, which makes the hand-off invisible to
 * callers. Entries are keyed by the prefix-cache block hash; once the backing file is full the
 * oldest entry is replaced.
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
        std::size_t num_dropped_no_buffer = 0;
    };

    /**
     * @brief Keeps existing entries from being replaced for as long as it exists.
     *
     * Warming a prefix chain evicts memory blocks in order to host that very chain, so persisting them
     * must not be allowed to reclaim a slot the chain still has to read back.
     */
    class ScopedReclamationPause {
    public:
        explicit ScopedReclamationPause(KVCacheOffloadCache& cache);
        ~ScopedReclamationPause();

        ScopedReclamationPause(const ScopedReclamationPause&) = delete;
        ScopedReclamationPause& operator=(const ScopedReclamationPause&) = delete;

    private:
        KVCacheOffloadCache& m_cache;
    };

    KVCacheOffloadCache(KVCacheManager& cache_manager,
                        std::unique_ptr<KVCacheOffloadManager> backend,
                        std::size_t max_queued_stores = 2);
    ~KVCacheOffloadCache();

    void on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) override;

    bool contains(std::size_t hash) const override;

    bool load_into(std::size_t hash, std::size_t block_index) override;

    /// Waits until every queued store has reached the backing file.
    void flush();

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

    /// A block whose contents are already copied out but not yet on disk.
    struct QueuedStore {
        std::size_t hash = 0;
        std::size_t slot_id = 0;
        bool reclaimed = false;
        std::vector<uint8_t> data;
    };

    /// @return A slot freed by dropping the oldest entry, or std::nullopt when there is nothing to drop.
    std::optional<std::size_t> reclaim_oldest_slot();

    const QueuedStore* find_queued(std::size_t hash) const;

    void publish(std::size_t hash, std::size_t slot_id, bool reclaimed);

    void run_writer();

    KVCacheManager& m_cache_manager;
    std::unique_ptr<KVCacheOffloadManager> m_backend;
    std::unordered_map<std::size_t, Entry> m_entries;
    // Front is the oldest entry and the first to be replaced when the file is full.
    std::list<std::size_t> m_insertion_order;
    // A store stays queued until it is published, so readers never lose sight of it.
    std::deque<QueuedStore> m_queued_stores;
    std::size_t m_max_queued_stores;
    std::vector<uint8_t> m_staging;
    Statistics m_statistics;
    bool m_reclamation_paused = false;
    bool m_stopping = false;
    mutable std::mutex m_mutex;
    std::condition_variable m_queued_cv;
    std::condition_variable m_drained_cv;
    std::thread m_writer;
};

}  // namespace ov::genai
