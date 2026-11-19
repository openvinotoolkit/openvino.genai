// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/file_kv_cache_store.hpp"

#include "openvino/core/except.hpp"

namespace ov::genai {

FileKVCacheStore::FileKVCacheStore(std::unique_ptr<KVCacheOffloadManager> backend) : m_backend(std::move(backend)) {
    OPENVINO_ASSERT(m_backend != nullptr, "KV cache store backend must not be null");
}

bool FileKVCacheStore::has_block(size_t hash) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_hash_to_slot.find(hash) != m_hash_to_slot.end();
}

bool FileKVCacheStore::write_block(size_t hash, const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_hash_to_slot.find(hash);
    if (it != m_hash_to_slot.end()) {
        // Idempotent overwrite of an existing key: the slot stays the same.
        m_backend->write_slot(it->second, data);
        return true;
    }

    std::optional<size_t> slot_id = m_backend->acquire_slot();
    if (!slot_id.has_value()) {
        return false;
    }
    try {
        m_backend->write_slot(*slot_id, data);
    } catch (...) {
        m_backend->release_slot(*slot_id);
        throw;
    }
    m_hash_to_slot.emplace(hash, *slot_id);
    return true;
}

bool FileKVCacheStore::read_block(size_t hash, std::vector<uint8_t>& data) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_hash_to_slot.find(hash);
    if (it == m_hash_to_slot.end()) {
        return false;
    }
    m_backend->read_slot(it->second, data);
    return true;
}

void FileKVCacheStore::drop_block(size_t hash) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_hash_to_slot.find(hash);
    if (it == m_hash_to_slot.end()) {
        return;
    }
    m_backend->release_slot(it->second);
    m_hash_to_slot.erase(it);
}

size_t FileKVCacheStore::get_block_size() const {
    return m_backend->get_slot_size();
}

size_t FileKVCacheStore::get_capacity_blocks() const {
    return m_backend->get_num_slots();
}

size_t FileKVCacheStore::get_num_free_blocks() const {
    return m_backend->get_num_free_slots();
}

}  // namespace ov::genai
