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
 * @brief What the offload internals need, resolved from a CacheOffloadConfig and the model by resolve_offload_settings().
 *
 * The user's CacheOffloadConfig is never changed; this is a separate, read-only result.
 */
struct OffloadSettings {
    /// Existing directory of the SSD tier.
    std::string kv_cache_dir;

    /// Byte budget of the KV group of the store, and of the linear attention group (0: no linear attention group).
    std::uint64_t capacity_bytes = 0;
    std::uint64_t linear_attention_capacity_bytes = 0;

    /// Handed to a store plugin as it is configured.
    double dram_cache_capacity_gb = 0.0;
    bool reuse_ssd_cache = true;
    std::string debug_log_path;

    /// Blocks queued for the writer thread per tier before further blocks are skipped. A block that is skipped breaks the
    /// prefix chain behind it, so the queue has to hold the blocks of a long prefill that the writer has not reached yet.
    static constexpr std::size_t DEFAULT_BUFFER_SLOTS = 512;

    /// Plugin to load; empty: the in-tree store. Loading it is the explicit opt-in of the user.
    std::string store_library_path;
    std::string store_provider_name;
    std::map<std::string, std::string> store_properties;
    static constexpr std::uint32_t DEFAULT_STORE_TIMEOUT_MS = 10000;
    std::uint32_t store_timeout_ms = DEFAULT_STORE_TIMEOUT_MS;

    /// Also store every block right after the forward pass that completed it.
    bool store_completed_blocks = false;

    /// What was resolved, for the log; the user's own values are printed by CacheOffloadConfig::to_string().
    std::string to_string() const {
        std::ostringstream oss;
        oss << "owner=" << (store_library_path.empty() ? "default_ssd" : "plugin") << " kv_cache_dir=" << kv_cache_dir
            << " kv_capacity_bytes=" << capacity_bytes << " la_capacity_bytes=" << linear_attention_capacity_bytes
            << " write_policy=" << (store_completed_blocks ? "WRITE_THROUGH" : "ON_EVICT")
            << " reuse_ssd_cache=" << (reuse_ssd_cache ? "true" : "false");
        return oss.str();
    }
};

}  // namespace ov::genai
