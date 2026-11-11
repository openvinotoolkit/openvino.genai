// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "logger.hpp"

namespace ov::genai {

namespace offload_tier_detail {
// TEMPORARY diagnostics helper, to be removed once the GPU cost breakdown is settled.
class ScopedTimer {
public:
    explicit ScopedTimer(std::size_t& accumulator_us)
        : m_accumulator_us(accumulator_us), m_started(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        m_accumulator_us += static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_started).count());
    }

private:
    std::size_t& m_accumulator_us;
    std::chrono::steady_clock::time_point m_started;
};
}  // namespace offload_tier_detail

/// Text an instantiation's traces and diagnostics use; specialized for KVCacheManager and
/// LinearAttentionCacheManager below so the two offload paths read exactly as they did before
/// they shared one implementation.
template <typename CacheManagerT>
struct BlockOffloadTierTraits;

/**
 * @brief Second-level prefix cache holding block contents of one physical cache type (KV or linear
 * attention) in a key-addressed store (an in-tree file or a third-party plugin, see IKVCacheStore).
 *
 * Blocks are written when the in-memory prefix cache is about to overwrite them, which is the last
 * point where their contents are still intact (or, with store_completed_blocks, as soon as the
 * forward pass that completed them has finished). Contents are copied out synchronously at that
 * moment and reach the store on a background thread, so the allocation path never waits for it. A
 * queued block is served straight from its staging copy, which makes the hand-off invisible to
 * callers. Entries are keyed by the prefix-cache block hash; once the store is full the oldest
 * entry is replaced, unless the store evicts by its own policy (IKVCacheStore::evicts_itself).
 *
 * Offload is best effort: a failed store is reported through the statistics and never propagates to
 * the caller, since losing a cache entry only costs recomputation.
 *
 * Templated on the physical cache manager (KVCacheManager or LinearAttentionCacheManager) so the KV
 * and linear-attention offload paths share one implementation instead of two parallel copies - the
 * manager type is the only thing that differs (how a block's raw bytes are read from/written to the
 * device). See BlockOffloadTierTraits for the per-instantiation diagnostic text.
 */
template <typename CacheManagerT>
class BlockOffloadTier : public IOverwrittenBlockObserver, public IExternalPrefixSource {
public:
    using Traits = BlockOffloadTierTraits<CacheManagerT>;

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
     * Warming a prefix chain evicts memory blocks in order to host that very chain, so storing them
     * must not be allowed to reclaim a slot the chain still has to read back.
     */
    class ScopedReclamationPause {
    public:
        explicit ScopedReclamationPause(BlockOffloadTier& cache) : m_cache(cache) {
            std::lock_guard<std::mutex> lock(m_cache.m_mutex);
            m_cache.m_reclamation_paused = true;
        }
        ~ScopedReclamationPause() {
            std::lock_guard<std::mutex> lock(m_cache.m_mutex);
            m_cache.m_reclamation_paused = false;
        }

        ScopedReclamationPause(const ScopedReclamationPause&) = delete;
        ScopedReclamationPause& operator=(const ScopedReclamationPause&) = delete;

    private:
        BlockOffloadTier& m_cache;
    };

    BlockOffloadTier(CacheManagerT& cache_manager, std::unique_ptr<IKVCacheStore> store, std::size_t max_queued_stores = 2)
        : m_cache_manager(cache_manager),
          m_store(std::move(store)),
          m_max_queued_stores(max_queued_stores) {
        OPENVINO_ASSERT(m_store != nullptr, Traits::kKind, " store must not be null");
        OPENVINO_ASSERT(m_max_queued_stores > 0, Traits::kKind, " needs at least one staging buffer");
        OPENVINO_ASSERT(m_store->get_block_size() == m_cache_manager.get_block_layout().get_slot_size(),
                        Traits::kKind, " store block size does not match the cache block layout");
        m_track_chains = m_store->is_plugin_backed();
        m_self_evicting = m_store->evicts_itself();
        m_max_links = std::max<std::size_t>(65536, 8 * m_store->get_capacity_blocks());
        m_writer = std::thread(&BlockOffloadTier::run_writer, this);
    }

    ~BlockOffloadTier() override {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping = true;
        }
        m_queued_cv.notify_all();
        if (m_writer.joinable()) {
            m_writer.join();
        }
        // TEMPORARY diagnostics, to be removed once the GPU cost breakdown is settled.
        GENAI_INFO("[KV_TRACE] %s stored=%zu loaded=%zu store_read=%zu us store_write=%zu us "
                   "load_disk=%zu us load_write=%zu us",
                   Traits::kCostTrace,
                   m_statistics.num_stored,
                   m_statistics.num_loaded,
                   m_statistics.store_read_us,
                   m_statistics.store_write_us,
                   m_statistics.load_disk_us,
                   m_statistics.load_write_us);
    }

    BlockOffloadTier(const BlockOffloadTier&) = delete;
    BlockOffloadTier& operator=(const BlockOffloadTier&) = delete;

    void on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) override {
        queue_store(hash, blocks, "overwritten");
    }

    /// With set_store_completed_blocks(true), copies a block out as soon as its forward pass has finished and
    /// lets the writer thread store it; the pipeline thread never waits for the store.
    void on_blocks_published(std::size_t hash, const BlocksPerLayer& blocks) override {
        if (m_store_completed_blocks) {
            queue_store(hash, blocks, "published");
        }
    }

    void set_store_completed_blocks(bool enabled) {
        m_store_completed_blocks = enabled;
    }

    void on_prefix_link(std::size_t hash, std::size_t parent_hash, bool has_parent) override {
        if (!m_track_chains) {
            return;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        record_link_locked(hash, parent_hash, has_parent);
    }

    bool contains(std::size_t hash) const override {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (find_queued(hash) != nullptr) {
                return true;
            }
            if (!m_self_evicting) {
                // The store may hold entries this instance never wrote, e.g. blocks of an earlier pipeline on a
                // persistent store; probe() finds them.
                return m_entries.find(hash) != m_entries.end() || m_store_hits.find(hash) != m_store_hits.end();
            }
        }
        // A self-evicting store may have dropped any block since the last probe, so ask it, with no lock held.
        return m_store->has_block(hash);
    }

    /**
     * @brief Asks the store about a whole prefix chain in one call, from block 0 in order, and remembers the answer.
     *
     * This is the only way the store is asked whether it holds a block: contains() and load_into() rely on
     * what the latest probes found. Probe a chain before writing blocks of it so the store learns their place.
     */
    std::vector<bool> probe(const std::vector<std::size_t>& chain) override {
        std::vector<bool> present(chain.size(), false);
        if (chain.empty()) {
            return present;
        }
        std::size_t reclaim_epoch = 0;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            reclaim_epoch = m_reclaim_epoch;
        }
        std::vector<bool> in_store(chain.size(), false);
        try {
            in_store = m_store->has_blocks(chain);
        } catch (const std::exception& error) {
            GENAI_WARN("%s store probe failed: %s", Traits::kKind, error.what());
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        // An entry reclaimed during the unlocked store call may still be reported present; remembering that hit
        // would make contains() claim a block that is gone, so such an answer is not trusted.
        const bool store_answer_current = reclaim_epoch == m_reclaim_epoch;
        for (std::size_t i = 0; i < chain.size(); ++i) {
            bool stored = false;
            if (store_answer_current) {
                stored = i < in_store.size() && in_store[i];
                if (stored) {
                    remember_store_hit_locked(chain[i]);
                } else {
                    // An authoritative miss outweighs any earlier hit, e.g. the store evicted it since the last probe.
                    m_store_hits.erase(chain[i]);
                }
            }
            present[i] = stored || m_entries.find(chain[i]) != m_entries.end() || find_queued(chain[i]) != nullptr;
            if (m_track_chains) {
                record_link_locked(chain[i], i > 0 ? chain[i - 1] : 0, i > 0);
            }
        }
        return present;
    }

    bool load_into(std::size_t hash, std::size_t block_index) override {
        std::unique_lock<std::mutex> lock(m_mutex);

        try {
            if (const QueuedStore* queued = find_queued(hash)) {
                // Still on its way to disk, so the staging copy is the closest source. Copied out now, under
                // the lock, since the writer thread could pop this entry as soon as the lock below is released.
                m_staging = queued->data;
                GENAI_INFO("[KV_TRACE] %s load hash=%zu source=staging physical_block=%zu", Traits::kClassName, hash, block_index);
                // The device write is not a plugin call, but releasing the lock for it too keeps this path
                // consistent with the disk path below and matches the ABI's "no host lock held" contract.
                const auto started = std::chrono::steady_clock::now();
                lock.unlock();
                m_cache_manager.write_block(block_index, m_staging);
                const auto write_us = static_cast<std::size_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started)
                        .count());
                lock.lock();
                m_statistics.load_write_us += write_us;
            } else {
                auto it = m_entries.find(hash);
                if (!m_self_evicting && it == m_entries.end() && m_store_hits.find(hash) == m_store_hits.end()) {
                    return false;
                }
                // Captured now: `it` must not be dereferenced once the lock below is released, since the writer
                // thread may concurrently insert into m_entries (a rehash would invalidate the iterator).
                const bool was_indexed = it != m_entries.end();
                GENAI_INFO("[KV_TRACE] %s load hash=%zu source=disk physical_block=%zu", Traits::kClassName, hash, block_index);
                const auto prefetched = m_prefetched.find(hash);
                bool have_data = prefetched != m_prefetched.end();
                if (have_data) {
                    m_staging = std::move(prefetched->second);
                    m_prefetched.erase(prefetched);
                }

                lock.unlock();
                bool read_failed = false;
                std::size_t disk_us = 0;
                if (!have_data) {
                    const auto started = std::chrono::steady_clock::now();
                    read_failed = !m_store->read_block(hash, m_staging);
                    disk_us = static_cast<std::size_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started)
                            .count());
                }
                std::size_t write_us = 0;
                if (!read_failed) {
                    const auto started = std::chrono::steady_clock::now();
                    m_cache_manager.write_block(block_index, m_staging);
                    write_us = static_cast<std::size_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started)
                            .count());
                }
                lock.lock();
                m_statistics.load_disk_us += disk_us;
                if (read_failed) {
                    m_store_hits.erase(hash);
                    // A self-evicting store drops blocks behind our back; that is a miss, not a broken index.
                    OPENVINO_ASSERT(m_self_evicting || !was_indexed,
                                    Traits::kKind, " index and store are out of sync for hash ", hash);
                    return false;
                }
                m_statistics.load_write_us += write_us;
            }
        } catch (const std::exception& error) {
            GENAI_WARN("%s load failed for hash %zu: %s", Traits::kKind, hash, error.what());
            ++m_statistics.num_failed;
            return false;
        }

        ++m_statistics.num_loaded;
        // TEMPORARY diagnostics, to be removed once the GPU cost breakdown is settled.
        GENAI_INFO("[KV_TRACE] %s index=%zu", Traits::kLoadBlockTrace, block_index);
        return true;
    }

    /// Reads the blocks of @p hashes the store holds in one batch, so the load_into calls of a warm-up are served from memory.
    void prefetch(const std::vector<std::size_t>& hashes) override {
        // Bounds the memory a warm-up can hold before it copies the blocks into the device cache.
        constexpr std::size_t max_prefetch_bytes = 256ull * 1024 * 1024;
        const std::size_t max_blocks =
            std::max<std::size_t>(1, max_prefetch_bytes / std::max<std::size_t>(1, m_store->get_block_size()));
        std::vector<std::size_t> wanted;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (const std::size_t hash : hashes) {
                if (wanted.size() >= max_blocks) {
                    break;
                }
                // A block still queued for writing is served from its staging copy, which is closer than the store.
                if (m_prefetched.find(hash) != m_prefetched.end() || find_queued(hash) != nullptr) {
                    continue;
                }
                if (m_self_evicting || m_entries.find(hash) != m_entries.end() ||
                    m_store_hits.find(hash) != m_store_hits.end()) {
                    wanted.push_back(hash);
                }
            }
        }
        if (wanted.empty()) {
            return;
        }

        std::vector<std::vector<uint8_t>> data;
        std::vector<bool> found;
        bool read_succeeded = true;
        const auto started = std::chrono::steady_clock::now();
        try {
            found = m_store->read_blocks(wanted, data);
        } catch (const std::exception& error) {
            // The blocks are then read one by one by load_into, which reports its own failures.
            GENAI_WARN("%s prefetch failed: %s", Traits::kKind, error.what());
            read_succeeded = false;
        }
        const auto read_us = static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count());

        std::lock_guard<std::mutex> lock(m_mutex);
        m_statistics.load_disk_us += read_us;
        if (!read_succeeded) {
            return;
        }
        for (std::size_t i = 0; i < wanted.size(); ++i) {
            if (found[i]) {
                m_prefetched.emplace(wanted[i], std::move(data[i]));
            }
        }
    }

    void release_prefetched() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_prefetched.clear();
    }

    /// Waits until every queued store has reached the backing file.
    void flush() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });
    }

    /**
     * @brief Reads the stored contents for @p hash.
     * @return false if @p hash has no entry, in which case @p block_data is left untouched.
     */
    bool read(std::size_t hash, std::vector<uint8_t>& block_data) const {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (const QueuedStore* queued = find_queued(hash)) {
                block_data = queued->data;
                return true;
            }
        }
        return m_store->read_block(hash, block_data);
    }

    std::size_t get_num_entries() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_entries.size();
    }

    /// Whether the backing store is a third-party plugin that may outlive this pipeline.
    bool is_plugin_backed() const {
        return m_store->is_plugin_backed();
    }

    /// Whether the backing store decides by itself which blocks to evict when it is full.
    bool evicts_itself() const {
        return m_self_evicting;
    }

    std::size_t get_num_free_slots() const {
        return m_store->get_num_free_blocks();
    }

    Statistics get_statistics() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_statistics;
    }

private:
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

    void record_link_locked(std::size_t hash, std::size_t parent, bool has_parent) {
        const auto [it, inserted] = m_links.try_emplace(hash, ChainLink{parent, has_parent});
        if (!inserted) {
            it->second = ChainLink{parent, has_parent};
            return;
        }
        m_link_order.push_back(hash);
        if (m_links.size() > m_max_links) {
            m_links.erase(m_link_order.front());
            m_link_order.pop_front();
        }
    }

    /// Copies the block out and queues it for the writer thread unless it is already stored or queued.
    void queue_store(std::size_t hash, const BlocksPerLayer& blocks, const char* reason) {
        // Offload is only enabled without cache eviction, where one block table is shared by all decoder
        // layers and a single physical block index therefore addresses the whole block set.
        OPENVINO_ASSERT(blocks.size() == 1,
                        Traits::kKind,
                        " requires the shared block table used when cache eviction is disabled, got ",
                        blocks.size(),
                        " block-table layers");
        const auto block_index = blocks.front()->get_index();
        OPENVINO_ASSERT(block_index >= 0, "Invalid physical block index ", block_index);

        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_entries.find(hash) != m_entries.end() || find_queued(hash) != nullptr) {
            // Contents are content-addressed by `hash`, so the stored copy already holds the same data.
            ++m_statistics.num_already_present;
            return;
        }

        if (m_queued_stores.size() >= m_max_queued_stores) {
            // Waiting here would stall block allocation, and a missing entry only costs recomputation.
            ++m_statistics.num_dropped_no_buffer;
            return;
        }

        bool reclaimed = false;
        try {
            // Blocks already queued will each take a free slot once the writer thread gets to them. A
            // self-evicting store makes room itself.
            if (!m_self_evicting && m_store->get_num_free_blocks() <= m_queued_stores.size()) {
                if (m_reclamation_paused) {
                    ++m_statistics.num_failed;
                    return;
                }
                reclaimed = reclaim_oldest_entry();
                if (!reclaimed) {
                    ++m_statistics.num_failed;
                    return;
                }
            }
        } catch (const std::exception& error) {
            GENAI_WARN("%s capacity check failed for hash %zu: %s", Traits::kKind, hash, error.what());
            ++m_statistics.num_failed;
            return;
        }

        QueuedStore queued;
        queued.hash = hash;
        queued.reclaimed = reclaimed;
        if (m_track_chains) {
            queued.chain = chain_to_locked(hash);
        }
        try {
            // Copy the contents out now; the block is handed over for overwriting as soon as this returns.
            offload_tier_detail::ScopedTimer timer(m_statistics.store_read_us);
            queued.data = m_cache_manager.read_block(static_cast<std::size_t>(block_index));
        } catch (const std::exception& error) {
            GENAI_WARN("%s store failed for hash %zu: %s", Traits::kKind, hash, error.what());
            ++m_statistics.num_failed;
            return;
        }

        m_queued_stores.push_back(std::move(queued));
        GENAI_INFO("[KV_TRACE] %s queue_store hash=%zu physical_block=%d queued=%zu reason=%s",
                   Traits::kClassName,
                   hash,
                   block_index,
                   m_queued_stores.size(),
                   reason);
        m_queued_cv.notify_one();
    }

    void run_writer() {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (true) {
            m_queued_cv.wait(lock, [this] { return m_stopping || !m_queued_stores.empty(); });
            if (m_queued_stores.empty()) {
                if (m_stopping) {
                    return;
                }
                continue;
            }

            // Everything queued so far goes to the store in one call. Only this thread pops, and deque never
            // invalidates references to existing elements, so these entries stay readable while the mutex is
            // released for the write.
            std::vector<const QueuedStore*> batch;
            batch.reserve(m_queued_stores.size());
            for (const QueuedStore& queued : m_queued_stores) {
                batch.push_back(&queued);
            }
            std::vector<bool> already_present(batch.size(), false);
            std::vector<bool> stored(batch.size(), false);

            const auto write_started = std::chrono::steady_clock::now();
            lock.unlock();
            try {
                std::vector<std::size_t> hashes;
                std::vector<const std::vector<uint8_t>*> data;
                std::vector<std::size_t> to_write;
                for (std::size_t i = 0; i < batch.size(); ++i) {
                    if (!batch[i]->chain.empty()) {
                        // Every write is preceded by a probe of its chain from block 0, so the store can place the block.
                        const std::vector<bool> present = m_store->has_blocks(batch[i]->chain);
                        // No index here knows what a self-evicting store holds, so the probe also tells whether to skip.
                        already_present[i] = m_self_evicting && !present.empty() && present.back();
                    }
                    if (!already_present[i]) {
                        to_write.push_back(i);
                        hashes.push_back(batch[i]->hash);
                        data.push_back(&batch[i]->data);
                    }
                }
                if (!to_write.empty()) {
                    const std::vector<bool> written = m_store->write_blocks(hashes, data);
                    for (std::size_t i = 0; i < to_write.size(); ++i) {
                        stored[to_write[i]] = written[i];
                        if (!written[i]) {
                            GENAI_WARN("%s store ran out of capacity after reservation for hash %zu", Traits::kKind, hashes[i]);
                        }
                    }
                }
            } catch (const std::exception& error) {
                GENAI_WARN("%s store failed for a batch of %zu blocks: %s", Traits::kKind, batch.size(), error.what());
                std::fill(stored.begin(), stored.end(), false);
            }
            // Stop the clock before re-taking the mutex, so lock contention is not counted as write time.
            const auto write_us = static_cast<std::size_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - write_started)
                    .count());
            lock.lock();
            // Accumulated only under the mutex, which get_statistics() also takes.
            m_statistics.store_write_us += write_us;

            for (std::size_t i = 0; i < batch.size(); ++i) {
                const std::size_t hash = m_queued_stores.front().hash;
                const bool reclaimed = m_queued_stores.front().reclaimed;
                m_queued_stores.pop_front();
                if (already_present[i]) {
                    ++m_statistics.num_already_present;
                } else if (stored[i]) {
                    publish(hash, reclaimed);
                } else {
                    ++m_statistics.num_failed;
                }
            }
            m_drained_cv.notify_all();
        }
    }

    /// @return The chain from block 0 up to @p hash as far as it is known, in order.
    std::vector<std::size_t> chain_to_locked(std::size_t hash) const {
        std::vector<std::size_t> reversed;
        std::size_t current = hash;
        for (std::size_t steps = 0; steps <= m_links.size(); ++steps) {
            reversed.push_back(current);
            const auto it = m_links.find(current);
            if (it == m_links.end() || !it->second.has_parent) {
                break;
            }
            current = it->second.parent;
        }
        return std::vector<std::size_t>(reversed.rbegin(), reversed.rend());
    }

    void remember_store_hit_locked(std::size_t hash) {
        if (m_store_hits.size() >= m_max_links) {
            m_store_hits.clear();
        }
        m_store_hits.insert(hash);
    }

    /// @return Whether an existing entry was dropped to free up room for a new one.
    bool reclaim_oldest_entry() {
        if (m_insertion_order.empty()) {
            return false;
        }
        const std::size_t oldest_hash = m_insertion_order.front();
        OPENVINO_ASSERT(m_entries.find(oldest_hash) != m_entries.end(),
                        Traits::kKind, " index and insertion order are out of sync");
        m_insertion_order.pop_front();
        m_entries.erase(oldest_hash);
        ++m_reclaim_epoch;
        m_store_hits.erase(oldest_hash);
        m_store->drop_block(oldest_hash);
        return true;
    }

    const QueuedStore* find_queued(std::size_t hash) const {
        for (const auto& queued : m_queued_stores) {
            if (queued.hash == hash) {
                return &queued;
            }
        }
        return nullptr;
    }

    void publish(std::size_t hash, bool reclaimed) {
        if (m_self_evicting) {
            // Nothing to index: the store alone knows whether the block is still there.
            ++m_statistics.num_stored;
            return;
        }
        m_insertion_order.push_back(hash);
        m_entries.insert(hash);
        ++m_statistics.num_stored;
        if (reclaimed) {
            ++m_statistics.num_replaced;
        }
        GENAI_INFO("[KV_TRACE] %s publish hash=%zu entries=%zu replaced=%s",
                   Traits::kClassName,
                   hash,
                   m_entries.size(),
                   reclaimed ? "true" : "false");
    }

    CacheManagerT& m_cache_manager;
    std::unique_ptr<IKVCacheStore> m_store;
    std::unordered_set<std::size_t> m_entries;
    // Hashes the latest probes found in the store; the store itself is never asked about a single block.
    std::unordered_set<std::size_t> m_store_hits;
    // Chain positions of published blocks, kept only for a plugin store, which wants the chain before a write.
    std::unordered_map<std::size_t, ChainLink> m_links;
    std::deque<std::size_t> m_link_order;
    std::size_t m_max_links = 0;
    bool m_track_chains = false;
    // The store drops blocks by its own policy, so no index of its content is kept here and nothing is reclaimed.
    bool m_self_evicting = false;
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
    // Counts reclaimed entries, so probe() can tell that a block was dropped while it asked the store without a lock.
    std::size_t m_reclaim_epoch = 0;
    bool m_reclamation_paused = false;
    bool m_stopping = false;
    mutable std::mutex m_mutex;
    std::condition_variable m_queued_cv;
    std::condition_variable m_drained_cv;
    std::thread m_writer;
};

template <>
struct BlockOffloadTierTraits<KVCacheManager> {
    static constexpr const char* kKind = "KV cache offload";
    static constexpr const char* kClassName = "KVCacheOffloadCache";
    static constexpr const char* kCostTrace = "offload_cost";
    static constexpr const char* kLoadBlockTrace = "offload_load_block";
};

template <>
struct BlockOffloadTierTraits<LinearAttentionCacheManager> {
    static constexpr const char* kKind = "Linear attention cache offload";
    static constexpr const char* kClassName = "LinearAttentionOffloadCache";
    static constexpr const char* kCostTrace = "la_offload_cost";
    static constexpr const char* kLoadBlockTrace = "la_offload_load_block";
};

}  // namespace ov::genai
