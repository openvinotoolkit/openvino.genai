// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>

#include "openvino/core/except.hpp"

namespace ov::genai {

/**
 * @brief Controls when device KV / linear attention cache blocks are copied into the offload store.
 * Eviction inside the store is always owned by the store itself.
 */
enum class CacheOffloadWritePolicy {
    ON_EVICT,       ///< Only blocks that are about to be overwritten in device memory (victim cache).
    WRITE_THROUGH,  ///< Also copy every completed block right after the forward pass that completed it.
};

/**
 * @brief A third-party KV cache store plugin loaded through the ABI in
 * openvino/genai/c/kv_cache_store_provider_abi.h.
 *
 * Setting `CacheOffloadConfig::plugin` with a non-empty `library_path` is the explicit opt-in to load that library.
 * Any failure to load, handshake, find the provider, open a store or purge it throws when the pipeline is created;
 * the host never falls back to the in-tree store.
 */
struct CacheStorePluginConfig {
    static constexpr std::uint32_t DEFAULT_TIMEOUT_MS = 10000;

    /// Path to the plugin shared library. Must not be empty.
    std::string library_path;

    /// Provider to select from the plugin module. Empty uses the module's first provider.
    std::string provider_name;

    /// Directory for the debug logs of the plugin. Empty: none.
    std::string debug_log_path;

    /**
     * Longest wait, in milliseconds, for one asynchronous batch of the plugin. A batch that is not done by then is
     * cancelled and counted as a cache miss, so a stalled plugin only costs recomputation. Must be greater than 0.
     */
    std::uint32_t timeout_ms = DEFAULT_TIMEOUT_MS;

    /// Plugin-specific settings handed to the plugin when it opens a store. The host does not interpret them.
    std::map<std::string, std::string> properties;

    bool operator==(const CacheStorePluginConfig& other) const {
        return library_path == other.library_path && provider_name == other.provider_name &&
               debug_log_path == other.debug_log_path && timeout_ms == other.timeout_ms &&
               properties == other.properties;
    }
};

/**
 * @brief Configuration of the tiered KV / linear attention cache offload.
 *
 * Offloaded blocks are rediscovered through the prefix-cache block hash, so offload is only meaningful together with
 * `SchedulerConfig::enable_prefix_caching`. It is not supported together with `SchedulerConfig::use_cache_eviction`,
 * `SchedulerConfig::use_sparse_attention`, or models whose sparse attention keeps index caches per block (QSA).
 * Setting `SchedulerConfig::cache_offload` enables offload; a model with linear attention layers offloads those too,
 * out of the same budget.
 *
 * The built-in host DRAM tier is not implemented yet: `dram_cache_capacity_gb` is only honored by a plugin that
 * advertises OV_KVCACHE_CAP_DRAM_TIER, and is otherwise ignored with a warning when an SSD tier is configured.
 */
struct CacheOffloadConfig {
    /// When blocks are written to the store.
    CacheOffloadWritePolicy write_policy = CacheOffloadWritePolicy::ON_EVICT;

    /// Upper bound of the DRAM tier in GiB (2^30 bytes); fractions are allowed. 0 disables the tier.
    double dram_cache_capacity_gb = 0.0;

    /// Upper bound of the SSD tier in GiB (2^30 bytes); fractions are allowed. 0 disables the tier.
    double ssd_cache_capacity_gb = 0.0;

    /// Existing, writable directory of the SSD tier. Required when `ssd_cache_capacity_gb` is greater than 0 and
    /// must be empty otherwise. There is no fallback to the system temporary directory.
    std::string kv_cache_dir;

    /// Whether the cache an earlier run left in `kv_cache_dir` for the same model is loaded (true) or purged when
    /// the store is opened (false).
    bool reuse_ssd_cache = true;

    /// A third-party store plugin. Unset: the in-tree store.
    std::optional<CacheStorePluginConfig> plugin;

    /// Model-independent checks of the values; throws ov::Exception.
    void validate() const {
        OPENVINO_ASSERT(std::isfinite(dram_cache_capacity_gb) && dram_cache_capacity_gb >= 0.0,
                        "CacheOffloadConfig dram_cache_capacity_gb must be finite and not negative");
        OPENVINO_ASSERT(std::isfinite(ssd_cache_capacity_gb) && ssd_cache_capacity_gb >= 0.0,
                        "CacheOffloadConfig ssd_cache_capacity_gb must be finite and not negative");
        OPENVINO_ASSERT(dram_cache_capacity_gb > 0.0 || ssd_cache_capacity_gb > 0.0,
                        "CacheOffloadConfig needs at least one tier: set dram_cache_capacity_gb or ssd_cache_capacity_gb");
        OPENVINO_ASSERT(ssd_cache_capacity_gb == 0.0 || !kv_cache_dir.empty(),
                        "CacheOffloadConfig kv_cache_dir is required when ssd_cache_capacity_gb is greater than 0");
        OPENVINO_ASSERT(ssd_cache_capacity_gb > 0.0 || kv_cache_dir.empty(),
                        "CacheOffloadConfig kv_cache_dir must be empty when ssd_cache_capacity_gb is 0");
        if (plugin.has_value()) {
            OPENVINO_ASSERT(!plugin->library_path.empty(),
                            "CacheOffloadConfig plugin needs a library_path; leave plugin unset to use the in-tree store");
            OPENVINO_ASSERT(plugin->timeout_ms > 0, "CacheOffloadConfig plugin timeout_ms must be greater than 0");
        }
    }

    bool operator==(const CacheOffloadConfig& other) const {
        return write_policy == other.write_policy && dram_cache_capacity_gb == other.dram_cache_capacity_gb &&
               ssd_cache_capacity_gb == other.ssd_cache_capacity_gb && kv_cache_dir == other.kv_cache_dir &&
               reuse_ssd_cache == other.reuse_ssd_cache && plugin == other.plugin;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "  CacheOffloadConfig { \n";
        oss << "    write_policy: "
            << (write_policy == CacheOffloadWritePolicy::ON_EVICT ? "ON_EVICT" : "WRITE_THROUGH") << "\n";
        oss << "    dram_cache_capacity_gb: " << dram_cache_capacity_gb << "\n";
        oss << "    ssd_cache_capacity_gb: " << ssd_cache_capacity_gb << "\n";
        oss << "    kv_cache_dir: " << (kv_cache_dir.empty() ? std::string("<none>") : kv_cache_dir) << "\n";
        oss << "    reuse_ssd_cache: " << std::boolalpha << reuse_ssd_cache << "\n";
        if (plugin.has_value()) {
            oss << "    plugin: { library_path: " << plugin->library_path << ", provider_name: "
                << (plugin->provider_name.empty() ? std::string("<default>") : plugin->provider_name)
                << ", debug_log_path: " << (plugin->debug_log_path.empty() ? std::string("<none>") : plugin->debug_log_path)
                << ", timeout_ms: " << plugin->timeout_ms << ", properties: {";
            for (const auto& [key, value] : plugin->properties) {
                oss << " " << key << "=" << value;
            }
            oss << " } }\n";
        } else {
            oss << "    plugin: <none>\n";
        }
        oss << "  }";
        return oss.str();
    }
};

}  // namespace ov::genai
