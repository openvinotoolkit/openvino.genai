// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/linear_attention_offload_cache.hpp"

#include <utility>

#include "logger.hpp"

namespace ov::genai {

LinearAttentionOffloadCache::LinearAttentionOffloadCache(LinearAttentionCacheManager& cache_manager,
                                                         std::unique_ptr<IKVCacheStore> store,
                                                         std::size_t max_queued_stores)
    : m_cache_manager(cache_manager),
      m_store(std::move(store)),
      m_max_queued_stores(max_queued_stores) {
    OPENVINO_ASSERT(m_store != nullptr, "Linear attention cache offload store must not be null");
    OPENVINO_ASSERT(m_max_queued_stores > 0, "Linear attention cache offload needs at least one staging buffer");
    OPENVINO_ASSERT(m_store->get_block_size() == m_cache_manager.get_block_layout().get_slot_size(),
                    "Linear attention cache offload store block size does not match the cache block layout");
    m_writer = std::thread(&LinearAttentionOffloadCache::run_writer, this);
}

LinearAttentionOffloadCache::~LinearAttentionOffloadCache() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stopping = true;
    }
    m_queued_cv.notify_all();
    if (m_writer.joinable()) {
        m_writer.join();
    }
    GENAI_INFO("[KV_TRACE] la_offload_cost stored=%zu loaded=%zu", m_statistics.num_stored, m_statistics.num_loaded);
}

void LinearAttentionOffloadCache::on_blocks_overwritten(std::size_t hash, const BlocksPerLayer& blocks) {
    // Offload is only enabled without cache eviction, where one block table is shared by all decoder
    // layers and a single physical block index therefore addresses the whole block set.
    OPENVINO_ASSERT(blocks.size() == 1,
                    "Linear attention cache offload requires a shared block table, got ",
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
        GENAI_WARN("Linear attention cache offload capacity check failed for hash %zu: %s", hash, error.what());
        ++m_statistics.num_failed;
        return;
    }

    QueuedStore queued;
    queued.hash = hash;
    queued.reclaimed = reclaimed;
    try {
        // Copy the contents out now; the block is handed over for overwriting as soon as this returns.
        queued.data = m_cache_manager.read_block(static_cast<std::size_t>(block_index));
    } catch (const std::exception& error) {
        GENAI_WARN("Linear attention cache offload store failed for hash %zu: %s", hash, error.what());
        ++m_statistics.num_failed;
        return;
    }

    m_queued_stores.push_back(std::move(queued));
    GENAI_INFO("[KV_TRACE] LinearAttentionOffloadCache queue_store hash=%zu physical_block=%d queued=%zu",
               hash,
               block_index,
               m_queued_stores.size());
    m_queued_cv.notify_one();
}

void LinearAttentionOffloadCache::run_writer() {
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
        lock.unlock();
        try {
            stored = m_store->write_block(hash, front.data);
            OPENVINO_ASSERT(stored,
                            "Linear attention cache offload store ran out of capacity after reservation for hash ",
                            hash);
        } catch (const std::exception& error) {
            GENAI_WARN("Linear attention cache offload store failed for hash %zu: %s", hash, error.what());
            stored = false;
        }
        lock.lock();

        m_queued_stores.pop_front();
        if (stored) {
            publish(hash, reclaimed);
        } else {
            ++m_statistics.num_failed;
        }
        m_drained_cv.notify_all();
    }
}

void LinearAttentionOffloadCache::publish(std::size_t hash, bool reclaimed) {
    m_insertion_order.push_back(hash);
    m_entries[hash] = Entry{std::prev(m_insertion_order.end())};
    ++m_statistics.num_stored;
    if (reclaimed) {
        ++m_statistics.num_replaced;
    }
    GENAI_INFO("[KV_TRACE] LinearAttentionOffloadCache publish hash=%zu entries=%zu replaced=%s",
               hash,
               m_entries.size(),
               reclaimed ? "true" : "false");
}

void LinearAttentionOffloadCache::flush() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });
}

void LinearAttentionOffloadCache::flush_durable() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });
    m_store->flush();
}

const LinearAttentionOffloadCache::QueuedStore* LinearAttentionOffloadCache::find_queued(std::size_t hash) const {
    for (const auto& queued : m_queued_stores) {
        if (queued.hash == hash) {
            return &queued;
        }
    }
    return nullptr;
}

LinearAttentionOffloadCache::ScopedReclamationPause::ScopedReclamationPause(LinearAttentionOffloadCache& cache)
    : m_cache(cache) {
    std::lock_guard<std::mutex> lock(m_cache.m_mutex);
    m_cache.m_reclamation_paused = true;
}

LinearAttentionOffloadCache::ScopedReclamationPause::~ScopedReclamationPause() {
    std::lock_guard<std::mutex> lock(m_cache.m_mutex);
    m_cache.m_reclamation_paused = false;
}

bool LinearAttentionOffloadCache::reclaim_oldest_entry() {
    for (auto it = m_insertion_order.begin(); it != m_insertion_order.end(); ++it) {
        const std::size_t oldest_hash = *it;
        if (m_pin_counts.find(oldest_hash) != m_pin_counts.end()) {
            continue;
        }
        OPENVINO_ASSERT(m_entries.find(oldest_hash) != m_entries.end(),
                        "Linear attention cache offload index and insertion order are out of sync");
        m_insertion_order.erase(it);
        m_entries.erase(oldest_hash);
        m_store->drop_block(oldest_hash);
        return true;
    }
    return false;
}

bool LinearAttentionOffloadCache::persist(std::size_t hash, std::size_t block_index) {
    std::unique_lock<std::mutex> lock(m_mutex);
    // Past this wait the mutex stays held, so the writer thread cannot take a slot behind our back.
    m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });

    if (m_entries.find(hash) != m_entries.end()) {
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

bool LinearAttentionOffloadCache::pin(std::size_t hash) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_entries.find(hash) == m_entries.end()) {
        return false;
    }
    ++m_pin_counts[hash];
    return true;
}

void LinearAttentionOffloadCache::unpin(std::size_t hash) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_pin_counts.find(hash);
    if (it == m_pin_counts.end()) {
        return;
    }
    if (--it->second == 0) {
        m_pin_counts.erase(it);
    }
}

bool LinearAttentionOffloadCache::contains(std::size_t hash) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.find(hash) != m_entries.end() || find_queued(hash) != nullptr;
}

bool LinearAttentionOffloadCache::load_into(std::size_t hash, std::size_t block_index) {
    std::lock_guard<std::mutex> lock(m_mutex);

    try {
        if (const QueuedStore* queued = find_queued(hash)) {
            // Still on its way to disk, so the staging copy is the closest source.
            GENAI_INFO("[KV_TRACE] LinearAttentionOffloadCache load hash=%zu source=staging physical_block=%zu",
                       hash,
                       block_index);
            m_cache_manager.write_block(block_index, queued->data);
        } else {
            auto it = m_entries.find(hash);
            if (it == m_entries.end()) {
                return false;
            }
            GENAI_INFO("[KV_TRACE] LinearAttentionOffloadCache load hash=%zu source=disk physical_block=%zu",
                       hash,
                       block_index);
            OPENVINO_ASSERT(m_store->read_block(hash, m_staging),
                            "Linear attention cache offload index and store are out of sync for hash ",
                            hash);
            m_cache_manager.write_block(block_index, m_staging);
        }
    } catch (const std::exception& error) {
        GENAI_WARN("Linear attention cache offload load failed for hash %zu: %s", hash, error.what());
        ++m_statistics.num_failed;
        return false;
    }

    ++m_statistics.num_loaded;
    GENAI_INFO("[KV_TRACE] la_offload_load_block index=%zu", block_index);
    return true;
}

bool LinearAttentionOffloadCache::read(std::size_t hash, std::vector<uint8_t>& block_data) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (const QueuedStore* queued = find_queued(hash)) {
        block_data = queued->data;
        return true;
    }
    if (m_entries.find(hash) == m_entries.end()) {
        return false;
    }
    return m_store->read_block(hash, block_data);
}

std::size_t LinearAttentionOffloadCache::get_num_entries() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.size();
}

LinearAttentionOffloadCache::Statistics LinearAttentionOffloadCache::get_statistics() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_statistics;
}

}  // namespace ov::genai
