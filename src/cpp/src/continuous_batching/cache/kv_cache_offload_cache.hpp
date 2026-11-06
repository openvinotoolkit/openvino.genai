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
#include <unordered_set>
#include <vector>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"

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
        // TEMPORARY diagnostics, to be removed once the GPU cost breakdown is settled.
        std::size_t store_read_us = 0;
        std::size_t store_write_us = 0;
        std::size_t load_disk_us = 0;
        std::size_t load_write_us = 0;
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
                        std::unique_ptr<IKVCacheStore> store,
                        std::size_t max_queued_stores = 2);
    ~KVCacheOffloadCache();

    void on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) override;

    /// With set_store_completed_blocks(true), copies a block out as soon as its forward pass has finished and
    /// lets the writer thread store it; the pipeline thread never waits for the store.
    void on_blocks_published(std::size_t hash, const BlocksPerLayer& blocks) override;

    void set_store_completed_blocks(bool enabled) {
        m_store_completed_blocks = enabled;
    }

    void on_prefix_link(std::size_t hash, std::size_t parent_hash, bool has_parent) override;

    bool contains(std::size_t hash) const override;

    /**
     * @brief Asks the store about a whole prefix chain in one call, from block 0 in order, and remembers the answer.
     *
     * This is the only way the store is asked whether it holds a block: contains(), pin() and persist() rely on
     * what the latest probes found. Probe a chain before writing blocks of it so the store learns their place.
     */
    std::vector<bool> probe(const std::vector<std::size_t>& chain) override;

    bool load_into(std::size_t hash, std::size_t block_index) override;

    /// Reads the blocks of @p hashes the store holds in one batch, so the load_into calls of a warm-up are served from memory.
    void prefetch(const std::vector<std::size_t>& hashes) override;

    void release_prefetched() override;

    /// Waits until every queued store has reached the backing file.
    void flush();

    /// Like flush(), then makes everything written so far durable in the store (IKVCacheStore::flush). Throws on failure.
    void flush_durable();
    /**
     * @brief Synchronously stores the block at @p block_index under @p hash and pins the entry.
     *
     * Unlike the opportunistic path this is reliable: it waits for in-flight stores, reuses an existing
     * entry for the same hash, and reclaims the oldest unpinned entry if the store is full. Every
     * successful call adds one pin that the caller must release with unpin().
     *
     * @return false if no slot can be freed because every entry is pinned or the store rejects the write.
     * @throws ov::Exception if reading the block or writing to the store fails; nothing is published then.
     */
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

    /// Whether the backing store is a third-party plugin that may outlive this pipeline.
    bool is_plugin_backed() const {
        return m_store->is_plugin_backed();
    }
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
        // The prefix chain from block 0 up to this block, probed right before the write. Empty when not tracked.
        std::vector<std::size_t> chain;
        std::vector<uint8_t> data;
    };

    /// Where a block sits in its chain.
    struct ChainLink {
        std::size_t parent = 0;
        bool has_parent = false;
    };

    void record_link_locked(std::size_t hash, std::size_t parent, bool has_parent);

    /// Copies the block out and queues it for the writer thread unless it is already stored or queued.
    void queue_store(std::size_t hash, const BlocksPerLayer& blocks, const char* reason);

    /// @return The chain from block 0 up to @p hash as far as it is known, in order.
    std::vector<std::size_t> chain_to_locked(std::size_t hash) const;

    void remember_store_hit_locked(std::size_t hash);

    /// @return Whether an existing entry was dropped to free up room for a new one.
    /// Pinned entries are never dropped.
    bool reclaim_oldest_entry();

    const QueuedStore* find_queued(std::size_t hash) const;

    void publish(std::size_t hash, bool reclaimed);

    /// Starts tracking an entry that is already in the store, e.g. one written by an earlier pipeline.
    void adopt_locked(std::size_t hash);

    void run_writer();

    KVCacheManager& m_cache_manager;
    std::unique_ptr<IKVCacheStore> m_store;
    std::unordered_map<std::size_t, Entry> m_entries;
    std::unordered_map<std::size_t, std::size_t> m_pin_counts;
    // Hashes the latest probes found in the store; the store itself is never asked about a single block.
    std::unordered_set<std::size_t> m_store_hits;
    // Chain positions of published blocks, kept only for a plugin store, which wants the chain before a write.
    std::unordered_map<std::size_t, ChainLink> m_links;
    std::deque<std::size_t> m_link_order;
    std::size_t m_max_links = 0;
    bool m_track_chains = false;
    bool m_store_completed_blocks = false;
    // Front is the oldest entry and the first to be replaced when the file is full.
    std::list<std::size_t> m_insertion_order;
    // A store stays queued until it is published, so readers never lose sight of it.
    std::deque<QueuedStore> m_queued_stores;
    std::size_t m_max_queued_stores;
    std::vector<uint8_t> m_staging;
    // Blocks read ahead by prefetch() and not yet consumed by load_into().
    std::unordered_map<std::size_t, std::vector<uint8_t>> m_prefetched;
    Statistics m_statistics;
    bool m_reclamation_paused = false;
    bool m_stopping = false;
    mutable std::mutex m_mutex;
    std::condition_variable m_queued_cv;
    std::condition_variable m_drained_cv;
    std::thread m_writer;
};

}  // namespace ov::genai
