// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// LinearAttentionOffloadCache is now BlockOffloadTier<LinearAttentionCacheManager> (see
// linear_attention_offload_cache.hpp / block_offload_tier.hpp); this translation unit exists only so
// a forced rebuild still notices if that header stops compiling for the LA instantiation.
#include "continuous_batching/cache/linear_attention_offload_cache.hpp"
