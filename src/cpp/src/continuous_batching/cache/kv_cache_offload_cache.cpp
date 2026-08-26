// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/kv_cache_offload_cache.hpp"

#include <utility>

#include "logger.hpp"

namespace ov::genai {

KVCacheOffloadCache::KVCacheOffloadCache(KVCacheManager& cache_manager,
                                         std::unique_ptr<KVCacheOffloadManager> backend,
                                         std::size_t max_queued_stores)
    : m_cache_manager(cache_manager),
      m_backend(std::move(backend)),
      m_max_queued_stores(max_queued_stores) {
    OPENVINO_ASSERT(m_backend != nullptr, "KV cache offload backend must not be null");
    OPENVINO_ASSERT(m_max_queued_stores > 0, "KV cache offload needs at least one staging buffer");
    OPENVINO_ASSERT(m_backend->get_slot_size() == m_cache_manager.get_block_layout().get_slot_size(),
                    "KV cache offload backend slot size does not match the cache block layout");
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

    std::optional<std::size_t> slot_id = m_backend->acquire_slot();
    bool reclaimed = false;
    if (!slot_id.has_value()) {
        if (m_reclamation_paused) {
            ++m_statistics.num_failed;
            return;
        }
        slot_id = reclaim_oldest_slot();
        reclaimed = slot_id.has_value();
    }
    if (!slot_id.has_value()) {
        ++m_statistics.num_failed;
        return;
    }

    QueuedStore queued;
    queued.hash = hash;
    queued.slot_id = *slot_id;
    queued.reclaimed = reclaimed;
    try {
        // Copy the contents out now; the block is handed over for overwriting as soon as this returns.
        queued.data = m_cache_manager.read_block(static_cast<std::size_t>(block_index));
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache offload store failed for hash %zu: %s", hash, error.what());
        m_backend->release_slot(*slot_id);
        ++m_statistics.num_failed;
        return;
    }

    m_queued_stores.push_back(std::move(queued));
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
        const std::size_t slot_id = front.slot_id;
        const bool reclaimed = front.reclaimed;

        bool stored = true;
        lock.unlock();
        try {
            m_backend->write_slot(slot_id, front.data);
        } catch (const std::exception& error) {
            GENAI_WARN("KV cache offload store failed for hash %zu: %s", hash, error.what());
            stored = false;
        }
        lock.lock();

        m_queued_stores.pop_front();
        if (stored) {
            publish(hash, slot_id, reclaimed);
        } else {
            m_backend->release_slot(slot_id);
            ++m_statistics.num_failed;
        }
        m_drained_cv.notify_all();
    }
}

void KVCacheOffloadCache::publish(std::size_t hash, std::size_t slot_id, bool reclaimed) {
    m_insertion_order.push_back(hash);
    m_entries[hash] = Entry{slot_id, std::prev(m_insertion_order.end())};
    ++m_statistics.num_stored;
    if (reclaimed) {
        ++m_statistics.num_replaced;
    }
}

void KVCacheOffloadCache::flush() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_drained_cv.wait(lock, [this] { return m_queued_stores.empty(); });
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

std::optional<std::size_t> KVCacheOffloadCache::reclaim_oldest_slot() {
    if (m_insertion_order.empty()) {
        return std::nullopt;
    }
    const std::size_t oldest_hash = m_insertion_order.front();
    auto it = m_entries.find(oldest_hash);
    OPENVINO_ASSERT(it != m_entries.end(), "KV cache offload index and insertion order are out of sync");
    const std::size_t slot_id = it->second.slot_id;
    m_insertion_order.pop_front();
    m_entries.erase(it);
    return slot_id;
}

bool KVCacheOffloadCache::contains(std::size_t hash) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_entries.find(hash) != m_entries.end() || find_queued(hash) != nullptr;
}

bool KVCacheOffloadCache::load_into(std::size_t hash, std::size_t block_index) {
    std::lock_guard<std::mutex> lock(m_mutex);

    try {
        if (const QueuedStore* queued = find_queued(hash)) {
            // Still on its way to disk, so the staging copy is the closest source.
            m_cache_manager.write_block(block_index, queued->data);
        } else {
            auto it = m_entries.find(hash);
            if (it == m_entries.end()) {
                return false;
            }
            m_backend->read_slot(it->second.slot_id, m_staging);
            m_cache_manager.write_block(block_index, m_staging);
        }
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache offload load failed for hash %zu: %s", hash, error.what());
        ++m_statistics.num_failed;
        return false;
    }

    ++m_statistics.num_loaded;
    return true;
}

bool KVCacheOffloadCache::read(std::size_t hash, std::vector<uint8_t>& block_data) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (const QueuedStore* queued = find_queued(hash)) {
        block_data = queued->data;
        return true;
    }
    auto it = m_entries.find(hash);
    if (it == m_entries.end()) {
        return false;
    }
    m_backend->read_slot(it->second.slot_id, block_data);
    return true;
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
