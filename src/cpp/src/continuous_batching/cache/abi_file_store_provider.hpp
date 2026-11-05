// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/c/kv_cache_store_provider_abi.h"

namespace ov::genai {

/// Returns the single provider descriptor for the in-tree default "file" store provider - the
/// same function a dlopen'd module would export as ov_pipeline_query_kvcache_store_providers(),
/// exposed directly for in-process use. Backs every OvKVCacheStoreVTable/ProviderVTable call with
/// FileKVCacheStore, so this is also the reference proof that kv_cache_store_provider_abi.h is
/// implementable and that round-tripping through it preserves data.
///
/// Recognized OvKVCacheStoreOpenOptions::properties keys:
///   "block_size_bytes" (required): decimal byte size of one opaque block.
///   "device" (optional, default "CPU"): forwarded to KVCacheOffloadManager::is_supported_device.
const OvKVCacheStoreModuleInfo* get_file_store_provider_module_info();

}  // namespace ov::genai
