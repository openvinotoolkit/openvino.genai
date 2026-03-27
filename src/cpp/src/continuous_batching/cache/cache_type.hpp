// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ov::genai {

/**
 * @brief Identifies the type of cache managed by a cache manager.
 * New cache types (e.g., SLIDING_WINDOW, LINEAR_ATTENTION) can be added here
 * and handled by implementing a corresponding ICacheManager.
 */
enum class CacheType {
    KV_CACHE,  ///< Standard full-attention KV cache (PagedAttention style)
};

}  // namespace ov::genai
