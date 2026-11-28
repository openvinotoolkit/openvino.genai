// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/c/kv_cache_store_provider_abi.h"

namespace ov::genai {

/// Returns the single provider descriptor for the in-tree default "file" store provider - the
/// same function a dlopen'd module would export as ov_pipeline_query_kvcache_store_providers(),
/// exposed directly for in-process use. Backs every OvKVCacheStoreVTable/ProviderVTable call with
/// a run-private DefaultFileGroupStore, so this is also the reference proof that kv_cache_store_provider_abi.h is
/// implementable and that round-tripping through it preserves data.
///
/// The block size is the one group of OvKVCacheStoreOpenOptions::groups. A host that passes it as the property
/// "block_size_bytes" instead (decimal byte size of one opaque block) is served as well.
const OvKVCacheStoreModuleInfo* get_file_store_provider_module_info();

}  // namespace ov::genai
