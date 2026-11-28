// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "continuous_batching/cache/offload_settings.hpp"
#include "logger.hpp"
#include "openvino/genai/cache_offload.hpp"

namespace ov::genai {

/// What one block of a cache group costs in the store and how many tokens it covers.
struct GroupCost {
    std::size_t block_bytes = 0;   ///< size of a slot in the store, not of the device tensors
    std::size_t block_tokens = 0;  ///< tokens of a KV block, or the checkpoint interval of linear attention
};

/// Blocks of group g needed to reach @p tokens.
inline std::size_t blocks_for_tokens(const GroupCost& group, std::size_t tokens) {
    return (tokens + group.block_tokens - 1) / group.block_tokens;
}

/**
 * @brief The largest token reach t for which sum over groups of ceil(t / block_tokens) * block_bytes fits in
 * @p budget_bytes, i.e. how far into a prompt the store can hold every group.
 */
inline std::size_t max_tokens_within_budget(const std::vector<GroupCost>& groups, std::uint64_t budget_bytes) {
    OPENVINO_ASSERT(!groups.empty(), "A store budget needs at least one cache group");
    const auto bytes_for = [&](std::size_t tokens) {
        std::uint64_t total = 0;
        for (const GroupCost& group : groups) {
            OPENVINO_ASSERT(group.block_bytes > 0 && group.block_tokens > 0, "A cache group needs a block size and a token span");
            total += static_cast<std::uint64_t>(blocks_for_tokens(group, tokens)) * group.block_bytes;
        }
        return total;
    };
    std::size_t max_block_tokens = 1;
    for (const GroupCost& group : groups) {
        max_block_tokens = std::max(max_block_tokens, group.block_tokens);
    }
    // Every group costs at least one byte per block, so no more than budget_bytes blocks of the widest group fit.
    std::size_t low = 0;
    std::size_t high = static_cast<std::size_t>(std::min<std::uint64_t>(
        budget_bytes, std::numeric_limits<std::size_t>::max() / (2 * max_block_tokens))) * max_block_tokens;
    while (low < high) {
        const std::size_t middle = low + (high - low + 1) / 2;
        if (bytes_for(middle) <= budget_bytes) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

/**
 * @brief Resolves what the user configured and the model's cache groups into what the offload internals use.
 *
 * @param groups The cache groups of the model: the KV cache first, then linear attention if the model has it.
 * @return Settings with the SSD budget split over the groups so that all of them reach the same number of tokens.
 * Throws when the budget cannot hold one block of every group.
 */
inline OffloadSettings resolve_offload_settings(const CacheOffloadConfig& config, const std::vector<GroupCost>& groups) {
    config.validate();
    OPENVINO_ASSERT(!groups.empty() && groups.size() <= 2, "Cache offload serves the KV cache and at most one linear attention cache");

    if (config.dram_cache_capacity_gb > 0.0) {
        // The built-in DRAM tier does not exist yet, so the SSD tier alone is used; a plugin may keep a DRAM tier itself.
        OPENVINO_ASSERT(config.ssd_cache_capacity_gb > 0.0,
                        "CacheOffloadConfig dram_cache_capacity_gb alone is not supported yet: the host DRAM tier is not implemented, "
                        "set ssd_cache_capacity_gb as well");
        if (!config.plugin.has_value()) {
            GENAI_WARN("CacheOffloadConfig dram_cache_capacity_gb=%g is ignored: the host DRAM tier is not implemented yet, "
                       "offload runs on the SSD tier only",
                       config.dram_cache_capacity_gb);
        }
    }

    OffloadSettings settings;
    settings.kv_cache_dir = config.kv_cache_dir;
    settings.dram_cache_capacity_gb = config.dram_cache_capacity_gb;
    settings.reuse_ssd_cache = config.reuse_ssd_cache;
    settings.store_completed_blocks = config.write_policy == CacheOffloadWritePolicy::WRITE_THROUGH;
    if (config.plugin.has_value()) {
        settings.store_library_path = config.plugin->library_path;
        settings.store_provider_name = config.plugin->provider_name;
        settings.debug_log_path = config.plugin->debug_log_path;
        settings.store_properties = config.plugin->properties;
        settings.store_timeout_ms = config.plugin->timeout_ms;
    }

    const auto budget_bytes = static_cast<std::uint64_t>(std::llround(config.ssd_cache_capacity_gb * (1ull << 30)));
    const std::size_t tokens = max_tokens_within_budget(groups, budget_bytes);
    std::uint64_t minimum_bytes = 0;
    for (const GroupCost& group : groups) {
        minimum_bytes += group.block_bytes;
    }
    for (const GroupCost& group : groups) {
        OPENVINO_ASSERT(blocks_for_tokens(group, tokens) >= 1,
                        "CacheOffloadConfig ssd_cache_capacity_gb is too small: one block of every cache group needs ",
                        minimum_bytes, " bytes at least");
    }
    settings.capacity_bytes = static_cast<std::uint64_t>(blocks_for_tokens(groups[0], tokens)) * groups[0].block_bytes;
    if (groups.size() > 1) {
        settings.linear_attention_capacity_bytes =
            static_cast<std::uint64_t>(blocks_for_tokens(groups[1], tokens)) * groups[1].block_bytes;
    }
    return settings;
}

}  // namespace ov::genai
