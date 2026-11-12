// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
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
    /**
     * Existing directory on the SSD where the offloaded KV cache files are stored, which is also the directory a store
     * plugin keeps its files under. When empty, the system temporary directory is used.
     */
    std::string kv_cache_dir;

    /** Upper bound of the offload file size in bytes. The usable slot count is derived from this. */
    std::size_t capacity_bytes = 0;

    /**
     * Upper bound of the SSD space used for the KV cache, in GB. When greater than 0 it takes the place of
     * `capacity_bytes`, which then has to stay 0.
     */
    std::int32_t ssd_cache_capacity_gb = 0;

    /**
     * Upper bound of the DRAM used for the KV cache, in GB. Handed to a store plugin as the `dram_cache_capacity_gb`
     * property; the in-tree file store has no DRAM tier and ignores it.
     */
    std::int32_t dram_cache_capacity_gb = 0;

    /**
     * Whether a store plugin loads the cache that an earlier run left in `kv_cache_dir` for the same model (true) or purges
     * it when opening the store (false). Handed to the plugin as the `reuse_ssd_cache` property; the in-tree file store
     * lives and dies with the pipeline and ignores it.
     */
    bool reuse_ssd_cache = true;

    /** Directory for the debug logs of a store plugin, handed to it as the `debug_log_path` property when not empty. */
    std::string debug_log_path;

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

    /**
     * Plugin-specific key/value settings handed to the store plugin when it opens a store, next to the ones the host
     * always passes (`block_size_bytes`, `device`, `dram_cache_capacity_gb`, `reuse_ssd_cache` and `debug_log_path`;
     * those cannot be overridden). The host does not interpret them.
     */
    std::map<std::string, std::string> store_properties;

    /**
     * Also stores every block as soon as the forward pass that completed it has finished (chunk store), instead of
     * only when the block is about to be overwritten. The block is copied out right after the pass and a
     * background thread writes it, so generation does not wait for the store. A block still stays in the device
     * cache; this makes it reusable from the store without having been evicted first, for example by another
     * process that opens the same persistent store.
     */
    bool store_completed_blocks = false;

    bool operator==(const CacheOffloadConfig& other) const {
        return kv_cache_dir == other.kv_cache_dir && capacity_bytes == other.capacity_bytes &&
               ssd_cache_capacity_gb == other.ssd_cache_capacity_gb &&
               dram_cache_capacity_gb == other.dram_cache_capacity_gb && reuse_ssd_cache == other.reuse_ssd_cache &&
               debug_log_path == other.debug_log_path &&
               linear_attention_capacity_bytes == other.linear_attention_capacity_bytes &&
               buffer_slots == other.buffer_slots && use_page_cache == other.use_page_cache &&
               store_library_path == other.store_library_path && store_provider_name == other.store_provider_name &&
               allow_untrusted_store_plugin == other.allow_untrusted_store_plugin &&
               store_properties == other.store_properties && store_completed_blocks == other.store_completed_blocks;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "  CacheOffloadConfig { \n";
        oss << "    kv_cache_dir: " << (kv_cache_dir.empty() ? std::string("<temporary directory>") : kv_cache_dir)
            << "\n";
        oss << "    capacity_bytes: " << capacity_bytes << "\n";
        oss << "    ssd_cache_capacity_gb: " << ssd_cache_capacity_gb << "\n";
        oss << "    dram_cache_capacity_gb: " << dram_cache_capacity_gb << "\n";
        oss << "    reuse_ssd_cache: " << std::boolalpha << reuse_ssd_cache << "\n";
        oss << "    debug_log_path: " << (debug_log_path.empty() ? std::string("<none>") : debug_log_path) << "\n";
        oss << "    linear_attention_capacity_bytes: " << linear_attention_capacity_bytes << "\n";
        oss << "    buffer_slots: " << buffer_slots << "\n";
        oss << "    use_page_cache: " << std::boolalpha << use_page_cache << "\n";
        oss << "    store_library_path: " << (store_library_path.empty() ? std::string("<none>") : store_library_path)
            << "\n";
        oss << "    store_provider_name: " << (store_provider_name.empty() ? std::string("<default>") : store_provider_name)
            << "\n";
        oss << "    allow_untrusted_store_plugin: " << std::boolalpha << allow_untrusted_store_plugin << "\n";
        oss << "    store_properties: {";
        for (const auto& [key, value] : store_properties) {
            oss << " " << key << "=" << value;
        }
        oss << " }\n";
        oss << "    store_completed_blocks: " << std::boolalpha << store_completed_blocks << "\n";
        oss << "  }";
        return oss.str();
    }
};

}  // namespace ov::genai
