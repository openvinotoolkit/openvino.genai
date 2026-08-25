// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/kv_cache_offload_cache.hpp"

#include <utility>

#include "logger.hpp"

namespace ov::genai {

KVCacheOffloadCache::KVCacheOffloadCache(KVCacheManager& cache_manager,
                                         std::unique_ptr<KVCacheOffloadManager> backend)
    : m_cache_manager(cache_manager),
      m_backend(std::move(backend)) {
    OPENVINO_ASSERT(m_backend != nullptr, "KV cache offload backend must not be null");
    OPENVINO_ASSERT(m_backend->get_slot_size() == m_cache_manager.get_block_layout().get_slot_size(),
                    "KV cache offload backend slot size does not match the cache block layout");
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

    if (m_entries.find(hash) != m_entries.end()) {
        // Contents are content-addressed by `hash`, so the stored copy already holds the same data.
        ++m_statistics.num_already_present;
        return;
    }

    std::optional<std::size_t> slot_id = m_backend->acquire_slot();
    bool reclaimed = false;
    if (!slot_id.has_value()) {
        slot_id = reclaim_oldest_slot();
        reclaimed = slot_id.has_value();
    }
    if (!slot_id.has_value()) {
        ++m_statistics.num_failed;
        return;
    }

    try {
        m_backend->write_slot(*slot_id, m_cache_manager.read_block(static_cast<std::size_t>(block_index)));
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache offload store failed for hash %zu: %s", hash, error.what());
        m_backend->release_slot(*slot_id);
        ++m_statistics.num_failed;
        return;
    }

    // Publish only once the contents are fully on disk.
    m_insertion_order.push_back(hash);
    m_entries[hash] = Entry{*slot_id, std::prev(m_insertion_order.end())};
    ++m_statistics.num_stored;
    if (reclaimed) {
        ++m_statistics.num_replaced;
    }
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
    return m_entries.find(hash) != m_entries.end();
}

bool KVCacheOffloadCache::read(std::size_t hash, std::vector<uint8_t>& block_data) const {
    std::lock_guard<std::mutex> lock(m_mutex);
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
