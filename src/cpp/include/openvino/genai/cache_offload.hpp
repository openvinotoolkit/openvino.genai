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

    /**
     * Upper bound of the linear attention state offload file size in bytes, in a separate file from
     * the KV cache offload file above. When 0 (default), linear attention cache offload is disabled
     * even if the model has linear attention layers; `capacity_bytes` alone only offloads KV cache.
     */
    std::size_t linear_attention_capacity_bytes = 0;

    /** Number of staging buffers reserved for transfers. */
    std::size_t buffer_slots = 2;

    /** Whether the offload file may go through the OS page cache. Direct I/O is not implemented yet. */
    bool use_page_cache = true;

    /**
     * Path to a third-party KV cache store plugin shared library implementing the ABI declared in
     * openvino/genai/c/kv_cache_store_provider_abi.h. Empty (default) uses the in-tree file store
     * directly, with no plugin ABI involved. Ignored unless `allow_untrusted_store_plugin` is true.
     */
    std::string store_library_path;

    /** Provider name to select from the plugin module. Empty uses the module's first provider. */
    std::string store_provider_name;

    /**
     * Must be explicitly set to load `store_library_path`. Defaults to false so that no third-party
     * code is ever loaded without an explicit opt-in (see kv_cache_store_provider_abi.h, D39).
     */
    bool allow_untrusted_store_plugin = false;

    bool operator==(const CacheOffloadConfig& other) const {
        return path == other.path && capacity_bytes == other.capacity_bytes &&
               linear_attention_capacity_bytes == other.linear_attention_capacity_bytes &&
               buffer_slots == other.buffer_slots && use_page_cache == other.use_page_cache &&
               store_library_path == other.store_library_path && store_provider_name == other.store_provider_name &&
               allow_untrusted_store_plugin == other.allow_untrusted_store_plugin;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "  CacheOffloadConfig { \n";
        oss << "    path: " << (path.empty() ? std::string("<temporary directory>") : path) << "\n";
        oss << "    capacity_bytes: " << capacity_bytes << "\n";
        oss << "    linear_attention_capacity_bytes: " << linear_attention_capacity_bytes << "\n";
        oss << "    buffer_slots: " << buffer_slots << "\n";
        oss << "    use_page_cache: " << std::boolalpha << use_page_cache << "\n";
        oss << "    store_library_path: " << (store_library_path.empty() ? std::string("<none>") : store_library_path)
            << "\n";
        oss << "    store_provider_name: " << (store_provider_name.empty() ? std::string("<default>") : store_provider_name)
            << "\n";
        oss << "    allow_untrusted_store_plugin: " << std::boolalpha << allow_untrusted_store_plugin << "\n";
        oss << "  }";
        return oss.str();
    }
};

}  // namespace ov::genai
