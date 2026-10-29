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
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"

namespace ov::genai {

/**
 * @brief Second-level prefix cache holding linear attention state block contents on disk.
 *
 * Structurally identical to KVCacheOffloadCache (see that class for the detailed design
 * rationale), but serializes LinearAttentionCacheManager's state-table-group tensors instead of
 * KVCacheManager's key/value tensors. Kept as a separate class rather than a generalization of
 * KVCacheOffloadCache so the well-exercised KV offload path is untouched by this addition.
 *
 * Reuses the FileKVCacheStore default IKVCacheStore backend: despite living in the KV cache offload
 * file, that class only deals in raw byte blocks keyed by content hash and has no KV-specific logic.
 */
class LinearAttentionOffloadCache : public IOverwrittenBlockObserver, public IExternalPrefixSource {
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
        explicit ScopedReclamationPause(LinearAttentionOffloadCache& cache);
        ~ScopedReclamationPause();

        ScopedReclamationPause(const ScopedReclamationPause&) = delete;
        ScopedReclamationPause& operator=(const ScopedReclamationPause&) = delete;

    private:
        LinearAttentionOffloadCache& m_cache;
    };

    LinearAttentionOffloadCache(LinearAttentionCacheManager& cache_manager,
                                std::unique_ptr<IKVCacheStore> store,
                                std::size_t max_queued_stores = 2);
    ~LinearAttentionOffloadCache();

    void on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) override;

    bool contains(std::size_t hash) const override;

    bool load_into(std::size_t hash, std::size_t block_index) override;

    /// Waits until every queued store has reached the backing file.
    void flush();

    /// Like flush(), then makes everything written so far durable in the store (IKVCacheStore::flush). Throws on failure.
    void flush_durable();
    /// Same contract as KVCacheOffloadCache::persist.
    bool persist(std::size_t hash, std::size_t block_index);

    /// @return false if @p hash has no published entry to pin.
    bool pin(std::size_t hash);

    /// Releases one pin; unpinning an entry that is not pinned does nothing.
    void unpin(std::size_t hash);

    /**
     * @brief Reads the stored contents for @p hash.
     * @return false if @p hash has no entry, in which case @p block_data is left untouched.
     */
    bool read(std::size_t hash, std::vector<uint8_t>& block_data) const;

    std::size_t get_num_entries() const;

    std::size_t get_num_free_slots() const {
        return m_store->get_num_free_blocks();
    }

    Statistics get_statistics() const;

private:
    struct Entry {
        std::list<std::size_t>::iterator order_it;
    };

    /// A block whose contents are already copied out but not yet on disk.
    struct QueuedStore {
        std::size_t hash = 0;
        bool reclaimed = false;
        std::vector<uint8_t> data;
    };

    /// @return Whether an existing entry was dropped to free up room for a new one.
    /// Pinned entries are never dropped.
    bool reclaim_oldest_entry();

    const QueuedStore* find_queued(std::size_t hash) const;

    void publish(std::size_t hash, bool reclaimed);

    void run_writer();

    LinearAttentionCacheManager& m_cache_manager;
    std::unique_ptr<IKVCacheStore> m_store;
    std::unordered_map<std::size_t, Entry> m_entries;
    std::unordered_map<std::size_t, std::size_t> m_pin_counts;
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
