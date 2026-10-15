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
 * A successfully loaded plugin library is never unloaded, per the ABI contract.
 */
std::unique_ptr<IKVCacheStore> open_kv_cache_store(const KVCacheDiskLayout& layout,
                                                   const CacheOffloadConfig& config,
                                                   const std::string& device);

}  // namespace ov::genai
