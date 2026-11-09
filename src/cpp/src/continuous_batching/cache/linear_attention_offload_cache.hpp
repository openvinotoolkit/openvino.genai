// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "continuous_batching/cache/block_offload_tier.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"

namespace ov::genai {

/// Second-level prefix cache holding linear-attention state block contents in a key-addressed store
/// (disk file or third-party plugin). See BlockOffloadTier for the implementation, shared with
/// KVCacheOffloadCache.
using LinearAttentionOffloadCache = BlockOffloadTier<LinearAttentionCacheManager>;

}  // namespace ov::genai
