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
#include <optional>
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
#include "prefix_digest_registry.hpp"
#include "prefix_token_registry.hpp"

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
 * callers. Blocks are keyed by the prefix-cache block hash. The store alone decides what it holds and what it evicts
 * when it is full, so this class keeps no index of its content: it only remembers what the latest probes found.
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
     * @brief Keeps the store from losing blocks to the blocks a warm-up pushes out of memory, for as long as it exists.
     *
     * Warming a prefix chain evicts memory blocks in order to host that very chain, so storing them must not be
     * allowed to push out a block the chain still has to read back. The store gets no writes meanwhile: the blocks wait
     * in the staging queue (where loads still find them) and are written once the pause ends. A full queue drops
     * further blocks.
     */
    class ScopedReclamationPause {
    public:
        explicit ScopedReclamationPause(BlockOffloadTier& cache) : m_cache(cache) {
            std::lock_guard<std::mutex> lock(m_cache.m_mutex);
            m_cache.m_reclamation_paused = true;
        }
        ~ScopedReclamationPause() {
            {
                std::lock_guard<std::mutex> lock(m_cache.m_mutex);
                m_cache.m_reclamation_paused = false;
            }
            m_cache.m_queued_cv.notify_all();
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
        OPENVINO_ASSERT(m_store->block_bytes() == m_cache_manager.get_block_layout().get_slot_size(),
                        Traits::kKind, " store block size does not match the cache block layout");
        m_track_chains = m_store->wants_chain_context();
        if (m_store->is_persistent()) {
            // A store that outlives this process is keyed by the 128-bit digest, which Sequence records on request.
            m_digest_lease.emplace();
        }
        m_max_links = std::max<std::size_t>(65536, 8 * m_store->capacity_blocks_hint());
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

    /**
     * @brief Starts telling a store that wants it which tokens each of its blocks covers.
     * @param block_size Tokens between two blocks of this tier: the checkpoint interval of linear attention.
     *
     * Sequence records the tokens of the prefixes it hashes with @p block_size while this tier lives. Only a store of
     * state checkpoints asks for them, so a KV tier never calls this.
     */
    void enable_prefix_tokens(std::size_t block_size) {
        if (m_store->wants_prefix_tokens()) {
            m_token_lease.emplace(block_size);
        }
    }

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
        const std::optional<ContentHash> digest = PrefixDigestRegistry::instance().find(hash);
        std::lock_guard<std::mutex> lock(m_mutex);
        record_link_locked(hash, parent_hash, has_parent, digest);
    }

    bool contains(std::size_t hash) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (find_queued(hash) != nullptr) {
            return true;
        }
        // The store is never asked here, because callers hold the block manager lock. It may hold blocks this
        // instance never wrote, e.g. blocks of an earlier pipeline on a persistent store; probe() finds them.
        return m_store_hits.find(hash) != m_store_hits.end();
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
        std::vector<bool> in_store(chain.size(), false);
        // The digests are resolved here, once, and pinned with the chain links below, so later requests for these
        // blocks name them by the same key even if the registry has moved on.
        std::vector<std::optional<ContentHash>> digests(chain.size());
        if (m_digest_lease) {
            digests = PrefixDigestRegistry::instance().find_many(chain);
        }
        std::vector<PrefixKey> keys;
        keys.reserve(chain.size());
        for (std::size_t i = 0; i < chain.size(); ++i) {
            keys.push_back(PrefixKey{chain[i], digests[i]});
        }
        try {
            announce_prefix_tokens(keys);
            in_store = m_store->has_blocks(keys);
        } catch (const std::exception& error) {
            GENAI_WARN("%s store probe failed: %s", Traits::kKind, error.what());
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        for (std::size_t i = 0; i < chain.size(); ++i) {
            const bool stored = i < in_store.size() && in_store[i];
            if (stored) {
                remember_store_hit_locked(chain[i]);
            } else {
                // An authoritative miss outweighs any earlier hit, e.g. the store evicted it since the last probe.
                m_store_hits.erase(chain[i]);
            }
            present[i] = stored || find_queued(chain[i]) != nullptr;
            if (m_track_chains) {
                record_link_locked(chain[i], i > 0 ? chain[i - 1] : 0, i > 0, digests[i]);
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
                // Callers hold the block manager lock, so the store is not read here: only prefetch() reads it,
                // without any host lock. A block it did not bring (over its memory budget, or lost) is a miss.
                const auto prefetched = m_prefetched.find(hash);
                if (prefetched == m_prefetched.end()) {
                    GENAI_INFO("[KV_TRACE] %s load hash=%zu source=none (not prefetched)", Traits::kClassName, hash);
                    return false;
                }
                m_staging = std::move(prefetched->second);
                m_prefetched.erase(prefetched);
                GENAI_INFO("[KV_TRACE] %s load hash=%zu source=disk physical_block=%zu", Traits::kClassName, hash, block_index);

                const auto started = std::chrono::steady_clock::now();
                lock.unlock();
                m_cache_manager.write_block(block_index, m_staging);
                const auto write_us = static_cast<std::size_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started)
                        .count());
                lock.lock();
                m_statistics.load_write_us += write_us;
            }
        } catch (const std::exception& error) {
            GENAI_WARN("%s load failed for hash %zu: %s", Traits::kKind, hash, error.what());
            // The write above runs with the lock released, so an exception from it arrives here unlocked.
            if (!lock.owns_lock()) {
                lock.lock();
            }
            ++m_statistics.num_failed;
            return false;
        }

        ++m_statistics.num_loaded;
        // TEMPORARY diagnostics, to be removed once the GPU cost breakdown is settled.
        GENAI_INFO("[KV_TRACE] %s index=%zu", Traits::kLoadBlockTrace, block_index);
        return true;
    }

    /// Reads the blocks of @p hashes the store holds in one batch; load_into() serves only from this memory.
    void prefetch(const std::vector<std::size_t>& hashes) override {
        const std::size_t max_blocks = max_prefetch_blocks();
        std::vector<std::size_t> wanted;
        std::vector<PrefixKey> wanted_keys;
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
                wanted.push_back(hash);
                wanted_keys.push_back(PrefixKey{hash, digest_locked(hash)});
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
            found = m_store->read_blocks(wanted_keys, data);
        } catch (const std::exception& error) {
            // Nothing was read ahead, so load_into reports these blocks as misses.
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
            } else {
                // Not found, or the store gave up on the batch; either way this warm-up cannot use the block.
                m_store_hits.erase(wanted[i]);
            }
        }
    }

    void release_prefetched() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_prefetched.clear();
    }

    /// Bounds the memory one prefetch() holds before the warm-up copies the blocks into the device cache.
    std::size_t max_prefetch_blocks() const override {
        return std::max<std::size_t>(1, m_max_prefetch_bytes / std::max<std::size_t>(1, m_store->block_bytes()));
    }

    void set_max_prefetch_bytes(std::size_t bytes) {
        m_max_prefetch_bytes = bytes;
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
        PrefixKey key{hash, std::nullopt};
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (const QueuedStore* queued = find_queued(hash)) {
                block_data = queued->data;
                return true;
            }
            key.digest = digest_locked(hash);
        }
        std::vector<std::vector<uint8_t>> blocks;
        if (!m_store->read_blocks({key}, blocks).at(0)) {
            return false;
        }
        block_data = std::move(blocks[0]);
        return true;
    }

    /// Hands what a prefix restore found to the store, with no lock held, as a plugin store may be slow.
    /// @return Whether the store reports the reuse metrics itself.
    bool report_restore_stats(const RestoreStats& stats) {
        return m_store->report_restore_stats(stats);
    }

    Statistics get_statistics() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_statistics;
    }

private:
    /// A block whose contents are already copied out but not yet on disk.
    struct QueuedStore {
        std::size_t hash = 0;
        // The 128-bit digest behind hash, when known; a plugin store addresses the block by it.
        std::optional<ContentHash> digest;
        // The prefix chain from block 0 up to this block, probed right before the write. Empty when not tracked.
        std::vector<PrefixKey> chain;
        std::vector<uint8_t> data;
    };

    /// Where a block sits in its chain.
    struct ChainLink {
        std::size_t parent = 0;
        bool has_parent = false;
        std::optional<ContentHash> digest;
    };

    void record_link_locked(std::size_t hash,
                            std::size_t parent,
                            bool has_parent,
                            const std::optional<ContentHash>& digest) {
        const auto [it, inserted] = m_links.try_emplace(hash, ChainLink{parent, has_parent, digest});
        if (!inserted) {
            const std::optional<ContentHash> known = it->second.digest;
            it->second = ChainLink{parent, has_parent, digest ? digest : known};
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

        // A hint that can be stale: the store may have evicted the block since, which only costs the next run a recompute.
        if (m_store_hits.find(hash) != m_store_hits.end() || find_queued(hash) != nullptr) {
            // Contents are content-addressed by `hash`, so the stored copy already holds the same data.
            ++m_statistics.num_already_present;
            return;
        }

        if (m_queued_stores.size() >= m_max_queued_stores) {
            // Waiting here would stall block allocation, and a missing entry only costs recomputation.
            ++m_statistics.num_dropped_no_buffer;
            return;
        }

        QueuedStore queued;
        queued.hash = hash;
        queued.digest = digest_locked(hash);
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
            // While a warm-up pauses reclamation, what it pushes out of memory waits here and is written afterwards.
            m_queued_cv.wait(lock, [this] {
                return m_stopping || (!m_queued_stores.empty() && !m_reclamation_paused);
            });
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
                std::vector<PrefixKey> keys;
                std::vector<const std::vector<uint8_t>*> data;
                std::vector<std::size_t> to_write;
                for (std::size_t i = 0; i < batch.size(); ++i) {
                    if (!batch[i]->chain.empty()) {
                        announce_prefix_tokens(batch[i]->chain);
                        // Every write is preceded by a probe of its chain from block 0, so the store can place the block.
                        const std::vector<bool> present = m_store->has_blocks(batch[i]->chain);
                        // No index here knows what the store holds, so the probe also tells whether to skip.
                        already_present[i] = !present.empty() && present.back();
                    }
                    if (!already_present[i]) {
                        to_write.push_back(i);
                        keys.push_back(PrefixKey{batch[i]->hash, batch[i]->digest});
                        data.push_back(&batch[i]->data);
                    }
                }
                if (!to_write.empty()) {
                    announce_prefix_tokens(keys);
                    const std::vector<bool> written = m_store->write_blocks(keys, data);
                    for (std::size_t i = 0; i < to_write.size(); ++i) {
                        stored[to_write[i]] = written[i];
                        if (!written[i]) {
                            GENAI_INFO("%s store did not take the block with hash %zu", Traits::kKind, keys[i].hash);
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
                m_queued_stores.pop_front();
                if (already_present[i]) {
                    ++m_statistics.num_already_present;
                    remember_store_hit_locked(hash);
                } else if (stored[i]) {
                    ++m_statistics.num_stored;
                    remember_store_hit_locked(hash);
                } else {
                    ++m_statistics.num_failed;
                }
            }
            m_drained_cv.notify_all();
        }
    }

    /// @return The chain from block 0 up to @p hash as far as it is known, in order.
    std::vector<PrefixKey> chain_to_locked(std::size_t hash) const {
        std::vector<PrefixKey> reversed;
        std::size_t current = hash;
        for (std::size_t steps = 0; steps <= m_links.size(); ++steps) {
            reversed.push_back(PrefixKey{current, digest_locked(current)});
            const auto it = m_links.find(current);
            if (it == m_links.end() || !it->second.has_parent) {
                break;
            }
            current = it->second.parent;
        }
        return std::vector<PrefixKey>(reversed.rbegin(), reversed.rend());
    }

    /// @return The digest of @p hash as pinned by a probe or a link, else what Sequence recorded last.
    std::optional<ContentHash> digest_locked(std::size_t hash) const {
        if (!m_digest_lease) {
            return std::nullopt;
        }
        if (const auto link = m_links.find(hash); link != m_links.end() && link->second.digest) {
            return link->second.digest;
        }
        return PrefixDigestRegistry::instance().find(hash);
    }

    void remember_store_hit_locked(std::size_t hash) {
        if (m_store_hits.size() >= m_max_links) {
            m_store_hits.clear();
        }
        m_store_hits.insert(hash);
    }

    /// Tells the store the tokens of the checkpoints in @p keys, when it asked for them. Called with no lock held.
    void announce_prefix_tokens(const std::vector<PrefixKey>& keys) {
        if (!m_token_lease) {
            return;
        }
        std::vector<PrefixTokens> prefixes;
        for (const PrefixKey& key : keys) {
            if (auto token_ids = PrefixTokenRegistry::instance().find(key.hash)) {
                prefixes.push_back(PrefixTokens{key, std::move(token_ids)});
            }
        }
        if (!prefixes.empty()) {
            m_store->announce_prefix_tokens(prefixes);
        }
    }

    const QueuedStore* find_queued(std::size_t hash) const {
        for (const auto& queued : m_queued_stores) {
            if (queued.hash == hash) {
                return &queued;
            }
        }
        return nullptr;
    }

    CacheManagerT& m_cache_manager;
    std::unique_ptr<IKVCacheStore> m_store;
    // Hashes the latest probes and writes found in the store; a hint, since the store evicts by its own policy.
    std::unordered_set<std::size_t> m_store_hits;
    // Chain positions of published blocks, kept only for a plugin store, which wants the chain before a write.
    std::unordered_map<std::size_t, ChainLink> m_links;
    std::deque<std::size_t> m_link_order;
    std::size_t m_max_links = 0;
    std::size_t m_max_prefetch_bytes = 256ull * 1024 * 1024;
    bool m_track_chains = false;
    // Held for a plugin store, so that Sequence records the digests it is keyed by.
    std::optional<PrefixDigestRegistry::Lease> m_digest_lease;
    // Held for a store of state checkpoints that wants their tokens, so that Sequence records them.
    std::optional<PrefixTokenRegistry::Lease> m_token_lease;
    bool m_store_completed_blocks = false;
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
