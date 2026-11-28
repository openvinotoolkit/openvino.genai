// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"
#include "continuous_batching/cache/offload_settings.hpp"

namespace ov::genai {

/**
 * @brief Opens the KV cache store backend configured by a OffloadSettings.
 *
 * Without a vendor plugin configured (the default), this returns the in-tree DefaultFileGroupStore
 * directly - no ABI marshaling in the hot path. When config.store_library_path is non-empty the plugin
 * is mandatory: any failure (library not found, missing entry symbol, ABI version/fingerprint
 * mismatch, named provider not found, open() failure, incomplete vtable) throws instead of silently
 * degrading to the file store.
 *
 * @param group Namespaces this store's keys from any other store opened against the same plugin
 * (e.g. "kv" vs "la") - plugins route purely on
 * `OvKVCacheBlockKey`, so distinct callers sharing one plugin/directory MUST use distinct groups
 * to avoid key collisions between unrelated cache types.
 *
 * A successfully loaded plugin library is never unloaded, per the ABI contract.
 *
 * @param block_tokens Tokens one block covers (a linear-attention group: the checkpoint interval),
 * reported to the plugin as `OvKVCacheGroupInfo::block_tokens`; 0 when blocks are not token-addressed.
 */
std::unique_ptr<IKVCacheStore> open_kv_cache_store(const KVCacheDiskLayout& layout,
                                                   const OffloadSettings& config,
                                                   const std::string& device,
                                                   const std::string& group,
                                                   size_t block_tokens = 0);

/// One cache group of a pipeline whose store is opened together with the others.
struct StoreGroupSpec {
    std::string name;           ///< group in the block keys of a store instance shared by all groups ("kv" / "la")
    std::string own_namespace;  ///< namespace (and group) when the group has an instance of its own
    KVCacheDiskLayout layout;
    size_t block_tokens = 0;    ///< see open_kv_cache_store()
    uint64_t capacity_bytes = 0;
};

/**
 * @brief Opens the stores of all cache groups of a pipeline together.
 *
 * A plugin that advertises OV_KVCACHE_CAP_MULTI_GROUP is opened once, under
 * @p shared_namespace, with every group in its open options, and the groups are views of that one instance. Any other
 * backend gets a store per group, as open_kv_cache_store() would open it.
 */
std::unique_ptr<IGroupedKVCacheStore> open_grouped_kv_cache_store(const std::vector<StoreGroupSpec>& groups,
                                                                  const OffloadSettings& config,
                                                                  const std::string& device,
                                                                  const std::string& shared_namespace);

}  // namespace ov::genai
