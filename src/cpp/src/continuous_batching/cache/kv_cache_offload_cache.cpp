// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/kv_cache_offload_cache.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

#include "logger.hpp"

namespace ov::genai {

namespace {

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

}  // namespace

KVCacheOffloadCache::KVCacheOffloadCache(KVCacheManager& cache_manager,
                                         std::unique_ptr<IKVCacheStore> store,
                                         std::size_t max_queued_stores)
    : m_cache_manager(cache_manager),
      m_store(std::move(store)),
      m_max_queued_stores(max_queued_stores) {
    OPENVINO_ASSERT(m_store != nullptr, "KV cache offload store must not be null");
    OPENVINO_ASSERT(m_max_queued_stores > 0, "KV cache offload needs at least one staging buffer");
    OPENVINO_ASSERT(m_store->get_block_size() == m_cache_manager.get_block_layout().get_slot_size(),
                    "KV cache offload store block size does not match the cache block layout");
    m_track_chains = m_store->is_plugin_backed();
    m_max_links = std::max<std::size_t>(65536, 8 * m_store->get_capacity_blocks());
    m_writer = std::thread(&KVCacheOffloadCache::run_writer, this);
}

KVCacheOffloadCache::~KVCacheOffloadCache() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopping = true;
    }
    m_queued_cv.notify_all();
    if (m_writer.joinable()) {
        m_writer.join();
    }
    // TEMPORARY diagnostics, to be removed once the GPU cost breakdown is settled.
    GENAI_INFO("[KV_TRACE] offload_cost stored=%zu loaded=%zu store_read=%zu us store_write=%zu us "
               "load_disk=%zu us load_write=%zu us",
               m_statistics.num_stored,
               m_statistics.num_loaded,
               m_statistics.store_read_us,
               m_statistics.store_write_us,
               m_statistics.load_disk_us,
               m_statistics.load_write_us);
}

void KVCacheOffloadCache::on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) {
    // Offload is only enabled without cache eviction, where one block table is shared by all decoder
    // layers and a single physical block index therefore addresses the whole block set.
    OPENVINO_ASSERT(blocks.size() == 1,
                    "KV cache offload requires the shared block table used when cache eviction is disabled, got ",
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
        if (m_store->get_num_free_blocks() == 0) {
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
        GENAI_WARN("KV cache offload capacity check failed for hash %zu: %s", hash, error.what());
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
        ScopedTimer timer(m_statistics.store_read_us);
        queued.data = m_cache_manager.read_block(static_cast<std::size_t>(block_index));
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache offload store failed for hash %zu: %s", hash, error.what());
        ++m_statistics.num_failed;
        return;
    }

    m_queued_stores.push_back(std::move(queued));
    GENAI_INFO("[KV_TRACE] KVCacheOffloadCache queue_store hash=%zu physical_block=%d queued=%zu",
               hash,
               block_index,
               m_queued_stores.size());
    m_queued_cv.notify_one();
}

void KVCacheOffloadCache::run_writer() {
    std::unique_lock<std::mutex> lock(m_mutex);
    while (true) {
        m_queued_cv.wait(lock, [this] { return m_stopping || !m_queued_stores.empty(); });
        if (m_queued_stores.empty()) {
            if (m_stopping) {
                return;
            }
            continue;
        }

        // Only this thread pops, and deque never invalidates references to existing elements,
        // so the front entry stays readable while the mutex is released for the write.
        const QueuedStore& front = m_queued_stores.front();
        const std::size_t hash = front.hash;
        const bool reclaimed = front.reclaimed;

        bool stored = true;
        const auto write_started = std::chrono::steady_clock::now();
        lock.unlock();
        try {
            if (!front.chain.empty()) {
                // Every write is preceded by a probe of its chain from block 0, so the store can place the block.
                m_store->has_blocks(front.chain);
            }
            stored = m_store->write_block(hash, front.data);
            OPENVINO_ASSERT(stored, "KV cache offload store ran out of capacity after reservation for hash ", hash);
        } catch (const std::exception& error) {
            GENAI_WARN("KV cache offload store failed for hash %zu: %s", hash, error.what());
            stored = false;
        }
        // Stop the clock before re-taking the mutex, so lock contention is not counted as write time.
        const auto write_us = static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - write_started)
                .count());
        lock.lock();
        // Accumulated only under the mutex, which get_statistics() also takes.
        m_statistics.store_write_us += write_us;

        m_queued_stores.pop_front();
        if (stored) {
            publish(hash, reclaimed);
        } else {
            ++m_statistics.num_failed;
        }
        m_drained_cv.notify_all();
    }
}

void KVCacheOffloadCache::publish(std::size_t hash, bool reclaimed) {
    m_insertion_order.push_back(hash);
    m_entries[hash] = Entry{std::prev(m_insertion_order.end())};
    ++m_statistics.num_stored;
    if (reclaimed) {
        ++m_statistics.num_replaced;
    }
    GENAI_INFO("[KV_TRACE] KVCacheOffloadCache publish hash=%zu entries=%zu replaced=%s",
               hash,
               m_entries.size(),
               reclaimed ? "true" : "false");
}

void KVCacheOffloadCache::adopt_locked(std::size_t hash) {
    m_insertion_order.push_back(hash);
    m_entries[hash] = Entry{std::prev(m_insertion_order.end())};
}

void KVCacheOffloadCache::flush() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });
}

void KVCacheOffloadCache::flush_durable() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });
    m_store->flush();
}

const KVCacheOffloadCache::QueuedStore* KVCacheOffloadCache::find_queued(std::size_t hash) const {
    for (const auto& queued : m_queued_stores) {
        if (queued.hash == hash) {
            return &queued;
        }
    }
    return nullptr;
}

KVCacheOffloadCache::ScopedReclamationPause::ScopedReclamationPause(KVCacheOffloadCache& cache) : m_cache(cache) {
    std::lock_guard<std::mutex> lock(m_cache.m_mutex);
    m_cache.m_reclamation_paused = true;
}

KVCacheOffloadCache::ScopedReclamationPause::~ScopedReclamationPause() {
    std::lock_guard<std::mutex> lock(m_cache.m_mutex);
    m_cache.m_reclamation_paused = false;
}

bool KVCacheOffloadCache::reclaim_oldest_entry() {
    for (auto it = m_insertion_order.begin(); it != m_insertion_order.end(); ++it) {
        const std::size_t oldest_hash = *it;
        if (m_pin_counts.find(oldest_hash) != m_pin_counts.end()) {
            continue;
        }
        OPENVINO_ASSERT(m_entries.find(oldest_hash) != m_entries.end(),
                        "KV cache offload index and insertion order are out of sync");
        m_insertion_order.erase(it);
        m_entries.erase(oldest_hash);
        m_store->drop_block(oldest_hash);
        return true;
    }
    return false;
}

bool KVCacheOffloadCache::persist(std::size_t hash, std::size_t block_index) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // Past this wait the mutex stays held, so the writer thread cannot take a slot behind our back.
    m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });

    if (m_entries.find(hash) != m_entries.end()) {
        ++m_pin_counts[hash];
        return true;
    }
    if (m_store_hits.find(hash) != m_store_hits.end()) {
        // Content-addressed: the latest probe found these bytes in the store, e.g. from an earlier pipeline.
        adopt_locked(hash);
        ++m_pin_counts[hash];
        return true;
    }

    bool reclaimed = false;
    if (m_store->get_num_free_blocks() == 0) {
        reclaimed = reclaim_oldest_entry();
        if (!reclaimed) {
            return false;
        }
    }

    const auto data = m_cache_manager.read_block(block_index);
    if (!m_store->write_block(hash, data)) {
        return false;
    }
    publish(hash, reclaimed);
    ++m_pin_counts[hash];
    return true;
}

bool KVCacheOffloadCache::pin(std::size_t hash) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_entries.find(hash) == m_entries.end()) {
        if (m_store_hits.find(hash) == m_store_hits.end()) {
            return false;
        }
        adopt_locked(hash);
    }
    ++m_pin_counts[hash];
    return true;
}

void KVCacheOffloadCache::unpin(std::size_t hash) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_pin_counts.find(hash);
    if (it == m_pin_counts.end()) {
        return;
    }
    if (--it->second == 0) {
        m_pin_counts.erase(it);
    }
}

bool KVCacheOffloadCache::contains(std::size_t hash) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    // The store may hold entries this instance never wrote, e.g. a snapshot taken by an earlier pipeline; probe() finds them.
    return m_entries.find(hash) != m_entries.end() || find_queued(hash) != nullptr ||
           m_store_hits.find(hash) != m_store_hits.end();
}

std::vector<bool> KVCacheOffloadCache::probe(const std::vector<std::size_t>& chain) {
    std::vector<bool> present(chain.size(), false);
    if (chain.empty()) {
        return present;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<bool> in_store(chain.size(), false);
    try {
        in_store = m_store->has_blocks(chain);
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache offload store probe failed: %s", error.what());
    }
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const bool stored = i < in_store.size() && in_store[i];
        if (stored) {
            remember_store_hit_locked(chain[i]);
        }
        present[i] = stored || m_entries.find(chain[i]) != m_entries.end() || find_queued(chain[i]) != nullptr;
        if (m_track_chains) {
            record_link_locked(chain[i], i > 0 ? chain[i - 1] : 0, i > 0);
        }
    }
    return present;
}

void KVCacheOffloadCache::on_prefix_link(std::size_t hash, std::size_t parent_hash, bool has_parent) {
    if (!m_track_chains) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    record_link_locked(hash, parent_hash, has_parent);
}

void KVCacheOffloadCache::record_link_locked(std::size_t hash, std::size_t parent, bool has_parent) {
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

std::vector<std::size_t> KVCacheOffloadCache::chain_to_locked(std::size_t hash) const {
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

void KVCacheOffloadCache::remember_store_hit_locked(std::size_t hash) {
    if (m_store_hits.size() >= m_max_links) {
        m_store_hits.clear();
    }
    m_store_hits.insert(hash);
}

bool KVCacheOffloadCache::load_into(std::size_t hash, std::size_t block_index) {
    std::lock_guard<std::mutex> lock(m_mutex);

    try {
        if (const QueuedStore* queued = find_queued(hash)) {
            // Still on its way to disk, so the staging copy is the closest source.
            GENAI_INFO("[KV_TRACE] KVCacheOffloadCache load hash=%zu source=staging physical_block=%zu",
                       hash,
                       block_index);
            ScopedTimer timer(m_statistics.load_write_us);
            m_cache_manager.write_block(block_index, queued->data);
        } else {
            auto it = m_entries.find(hash);
            if (it == m_entries.end() && m_store_hits.find(hash) == m_store_hits.end()) {
                return false;
            }
            GENAI_INFO("[KV_TRACE] KVCacheOffloadCache load hash=%zu source=disk physical_block=%zu", hash, block_index);
            const auto prefetched = m_prefetched.find(hash);
            if (prefetched != m_prefetched.end()) {
                m_staging = std::move(prefetched->second);
                m_prefetched.erase(prefetched);
            } else {
                ScopedTimer timer(m_statistics.load_disk_us);
                if (!m_store->read_block(hash, m_staging)) {
                    m_store_hits.erase(hash);
                    OPENVINO_ASSERT(it == m_entries.end(),
                                    "KV cache offload index and store are out of sync for hash ",
                                    hash);
                    return false;
                }
            }
            ScopedTimer timer(m_statistics.load_write_us);
            m_cache_manager.write_block(block_index, m_staging);
        }
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache offload load failed for hash %zu: %s", hash, error.what());
        ++m_statistics.num_failed;
        return false;
    }

    ++m_statistics.num_loaded;
    // TEMPORARY diagnostics, to be removed once the GPU cost breakdown is settled.
    GENAI_INFO("[KV_TRACE] offload_load_block index=%zu", block_index);
    return true;
}

void KVCacheOffloadCache::prefetch(const std::vector<std::size_t>& hashes) {
    // Bounds the memory a warm-up can hold before it copies the blocks into the device cache.
    constexpr std::size_t max_prefetch_bytes = 256ull * 1024 * 1024;
    std::lock_guard<std::mutex> lock(m_mutex);
    const std::size_t max_blocks = std::max<std::size_t>(1, max_prefetch_bytes / std::max<std::size_t>(1, m_store->get_block_size()));
    std::vector<std::size_t> wanted;
    for (const std::size_t hash : hashes) {
        if (wanted.size() >= max_blocks) {
            break;
        }
        // A block still queued for writing is served from its staging copy, which is closer than the store.
        if (m_prefetched.find(hash) != m_prefetched.end() || find_queued(hash) != nullptr) {
            continue;
        }
        if (m_entries.find(hash) != m_entries.end() || m_store_hits.find(hash) != m_store_hits.end()) {
            wanted.push_back(hash);
        }
    }
    if (wanted.empty()) {
        return;
    }
    try {
        std::vector<std::vector<uint8_t>> data;
        std::vector<bool> found;
        {
            ScopedTimer timer(m_statistics.load_disk_us);
            found = m_store->read_blocks(wanted, data);
        }
        for (std::size_t i = 0; i < wanted.size(); ++i) {
            if (found[i]) {
                m_prefetched.emplace(wanted[i], std::move(data[i]));
            }
        }
    } catch (const std::exception& error) {
        // The blocks are then read one by one by load_into, which reports its own failures.
        GENAI_WARN("KV cache offload prefetch failed: %s", error.what());
    }
}

void KVCacheOffloadCache::release_prefetched() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_prefetched.clear();
}

bool KVCacheOffloadCache::read(std::size_t hash, std::vector<uint8_t>& block_data) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (const QueuedStore* queued = find_queued(hash)) {
        block_data = queued->data;
        return true;
    }
    return m_store->read_block(hash, block_data);
}

std::size_t KVCacheOffloadCache::get_num_entries() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.size();
}

KVCacheOffloadCache::Statistics KVCacheOffloadCache::get_statistics() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_statistics;
}

}  // namespace ov::genai
