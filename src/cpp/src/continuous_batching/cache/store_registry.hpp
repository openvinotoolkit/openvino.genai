// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"
#include "openvino/genai/cache_offload.hpp"

namespace ov::genai {

/**
 * @brief Opens the KV cache store backend configured by a CacheOffloadConfig.
 *
 * Without a vendor plugin configured (the default), this returns the in-tree FileKVCacheStore
 * directly - no ABI marshaling in the hot path. A vendor plugin is only even attempted when
 * `config.allow_untrusted_store_plugin` is true and `config.store_library_path` is non-empty; any
 * failure along the way (library not found, missing entry symbol, ABI version/fingerprint
 * mismatch, named provider not found, `open()` failure) is logged and degrades to the in-tree
 * default rather than throwing - offload is best effort, never a hard dependency on a plugin.
 *
 * @param group Namespaces this store's keys from any other store opened against the same plugin
 * (e.g. "kv" vs "la", or a session-snapshot-specific group) - plugins route purely on
 * `OvKVCacheBlockKey`, so distinct callers sharing one plugin/directory MUST use distinct groups
 * to avoid key collisions between unrelated cache types.
 *
 * A successfully loaded plugin library is never unloaded, per the ABI contract.
 */
std::unique_ptr<IKVCacheStore> open_kv_cache_store(const KVCacheDiskLayout& layout,
                                                   const CacheOffloadConfig& config,
                                                   const std::string& device,
                                                   const std::string& group);

/**
 * @brief Opens ONLY a vendor plugin store - never falls back to the in-tree default.
 *
 * For callers where silently degrading to the in-tree (ephemeral, process-lifetime-only) default
 * store would be actively wrong, e.g. session snapshot: a snapshot's durability promise must not
 * quietly depend on whichever store happened to load successfully.
 *
 * @throws ov::Exception if no plugin is configured (`allow_untrusted_store_plugin` is false or
 * `store_library_path` is empty) or the plugin fails to load/handshake/open for any reason.
 */
std::unique_ptr<IKVCacheStore> open_required_kv_cache_store_plugin(const KVCacheDiskLayout& layout,
                                                                    const CacheOffloadConfig& config,
                                                                    const std::string& device,
                                                                    const std::string& group);

}  // namespace ov::genai
