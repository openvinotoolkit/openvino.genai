// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <sstream>
#include <string>

namespace ov::genai {

/**
 * @brief Configuration of the KV cache disk offload backend.
 *
 * Offloaded blocks are rediscovered through the prefix-cache block hash, so offload is only
 * meaningful together with `SchedulerConfig::enable_prefix_caching`.
 */
struct CacheOffloadConfig {
    /** Existing directory to place the offload file in. When empty, the system temporary directory is used. */
    std::string path;

    /** Upper bound of the offload file size in bytes. The usable slot count is derived from this. */
    std::size_t capacity_bytes = 0;

    /** Number of staging buffers reserved for transfers. */
    std::size_t buffer_slots = 2;

    /** Whether the offload file may go through the OS page cache. Direct I/O is not implemented yet. */
    bool use_page_cache = true;

    bool operator==(const CacheOffloadConfig& other) const {
        return path == other.path && capacity_bytes == other.capacity_bytes &&
               buffer_slots == other.buffer_slots && use_page_cache == other.use_page_cache;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "  CacheOffloadConfig { \n";
        oss << "    path: " << (path.empty() ? std::string("<temporary directory>") : path) << "\n";
        oss << "    capacity_bytes: " << capacity_bytes << "\n";
        oss << "    buffer_slots: " << buffer_slots << "\n";
        oss << "    use_page_cache: " << std::boolalpha << use_page_cache << "\n";
        oss << "  }";
        return oss.str();
    }
};

}  // namespace ov::genai
