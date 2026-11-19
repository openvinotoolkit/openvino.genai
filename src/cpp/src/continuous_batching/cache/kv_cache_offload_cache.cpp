// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// KVCacheOffloadCache is now BlockOffloadTier<KVCacheManager> (see kv_cache_offload_cache.hpp /
// block_offload_tier.hpp); this translation unit exists only so a forced rebuild still notices if
// that header stops compiling for the KV instantiation.
#include "continuous_batching/cache/kv_cache_offload_cache.hpp"
