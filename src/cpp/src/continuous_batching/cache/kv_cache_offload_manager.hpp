// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "openvino/genai/cache_offload.hpp"

namespace ov::genai {

/**
 * @brief Fixed-slot disk backend for offloaded KV cache blocks.
 *
 * Owns a run-specific file, hands out slots sized to exactly one block set, and performs
 * length-verified reads and writes. It deliberately knows nothing about block hashes,
 * sequences or eviction policy - those stay in the block manager and the orchestrator.
 */
class KVCacheOffloadManager {
public:
    KVCacheOffloadManager(const KVCacheDiskLayout& layout,
                          const CacheOffloadConfig& config,
                          const std::string& device);
    ~KVCacheOffloadManager();

    KVCacheOffloadManager(const KVCacheOffloadManager&) = delete;
    KVCacheOffloadManager& operator=(const KVCacheOffloadManager&) = delete;

    /// @return Whether KV cache offload is implemented for this inference device.
    static bool is_supported_device(const std::string& device);

    /// @return Size of one slot in bytes, equal to the byte size of one block set across all layers.
    std::size_t get_slot_size() const {
        return m_slot_size;
    }

    /// @return Total number of slots derived from the configured capacity.
    std::size_t get_num_slots() const {
        return m_num_slots;
    }

    std::size_t get_num_free_slots() const;

    /// @return A free slot, or std::nullopt when the offload file is full.
    std::optional<std::size_t> acquire_slot();

    void release_slot(std::size_t slot_id);

    void write_slot(std::size_t slot_id, const std::vector<uint8_t>& block_data);

    void read_slot(std::size_t slot_id, std::vector<uint8_t>& block_data) const;

    const std::filesystem::path& get_file_path() const {
        return m_file_path;
    }

private:
    void write_at(std::size_t offset, const uint8_t* data, std::size_t size) const;
    void read_at(std::size_t offset, uint8_t* data, std::size_t size) const;
    void close_and_remove() noexcept;

    KVCacheDiskLayout m_layout;
    std::filesystem::path m_file_path;
    int m_fd = -1;
    std::size_t m_slot_size = 0;
    std::size_t m_num_slots = 0;
    std::vector<std::size_t> m_free_slots;
    // Serializes slot bookkeeping and the seek/read/write pairs used on platforms without positional I/O.
    mutable std::mutex m_mutex;
};

}  // namespace ov::genai
