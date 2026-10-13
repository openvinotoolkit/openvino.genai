// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <vector>

#include "continuous_batching/cache/cache_type.hpp"

namespace ov::genai {

class CacheOrchestrator;

/**
 * @brief Self-contained on-disk snapshot of a single sequence's full cache state (KV blocks in
 * full and, for hybrid models, the latest linear-attention checkpoint), used to resume a
 * conversation across process restarts without re-running prefill.
 *
 * Unlike the opportunistic/LRU disk offload (KVCacheOffloadCache / LinearAttentionOffloadCache),
 * loading a snapshot does not depend on a new request's prompt re-tokenizing to an identical
 * token sequence - the saved token ids are the ground truth, restored directly.
 *
 * V1 scope: single-sequence groups only (no beam search / forked sequences) and
 * `use_cache_eviction` must be disabled - both are validated at save() time.
 */
class SessionSnapshot {
public:
    /**
     * @brief Captures `seq_id`'s current cache contents and full token history to `path`.
     *
     * KV cache is saved in full (one entry per logical block, in token order); linear-attention
     * cache is saved as only its single latest checkpoint block, since the recurrent state is
     * already a compressed summary of all prior history and does not need the full chain.
     *
     * @param orchestrator The cache orchestrator owning `seq_id`'s physical blocks.
     * @param seq_id       The sequence to snapshot. Must be the sole sequence of its group.
     * @param token_ids    The full token history (prompt + generated-so-far) for the sequence.
     * @param path         Destination file path. Overwritten if it already exists.
     */
    static void save(CacheOrchestrator& orchestrator,
                      uint64_t seq_id,
                      const std::vector<int64_t>& token_ids,
                      const std::filesystem::path& path);

    /// Cache payload loaded from a snapshot, keyed by cache type, each a list of blocks in the
    /// order they must be written back (KV: full logical order; linear-attention: single entry).
    struct Loaded {
        std::vector<int64_t> token_ids;
        std::map<CacheType, std::vector<std::vector<uint8_t>>> blocks_per_type;
    };

    /**
     * @brief Reads and validates `path` against the orchestrator's currently loaded model/device.
     * @throws ov::Exception on format, checksum, or model/device compatibility mismatch.
     */
    static Loaded load(const CacheOrchestrator& orchestrator, const std::filesystem::path& path);
};

}  // namespace ov::genai
