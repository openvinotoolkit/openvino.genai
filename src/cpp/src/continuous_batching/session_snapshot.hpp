// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <vector>

#include "continuous_batching/cache/cache_type.hpp"
#include "sequence_group.hpp"

namespace ov::genai {

class CacheOrchestrator;

/**
 * @brief On-disk snapshot of a single sequence, used to resume a conversation without re-running prefill.
 *
 * The file holds only the history (token ids, or for embeddings-routed models the input embeddings, position ids
 * and rope delta) and the prefix hashes of the sequence's cache blocks. The block contents live in the pipeline's
 * prefix-caching offload store (`SchedulerConfig::use_cache_offload`), pinned so that opportunistic eviction does not
 * drop them; a restore re-submits the history as an ordinary prompt and the regular prefix restore brings the blocks
 * back, so only the last position is recomputed. A new pipeline can restore a snapshot only if its store still holds
 * the blocks, i.e. a persistent store plugin.
 *
 * Supports plain token-id sequences (`SequenceGroupType::TOKENS`) and, for text-only requests, embeddings-routed
 * ones (`SequenceGroupType::EMBEDDINGS`, e.g. Qwen3.5). Requests carrying multimodal state are rejected at save().
 *
 * Single-sequence groups only (no beam search / forked sequences); `use_cache_eviction` must be disabled.
 */
class SessionSnapshot {
public:
    /**
     * @brief Persists `sequence`'s cache blocks in the offload store and writes the snapshot file to `path`.
     *
     * The linear-attention cache of a hybrid model is saved as only its latest checkpoint, since the recurrent state
     * already summarizes all prior history.
     *
     * @param orchestrator The cache orchestrator owning `sequence`'s physical blocks.
     * @param seq_group    The (single-sequence) group `sequence` belongs to.
     * @param sequence     The sequence to snapshot.
     * @param path         Destination file path. Overwritten if it already exists.
     * @throws ov::Exception if prefix caching / cache offload are not enabled, the request is still prefilling, or
     * the store cannot hold the blocks.
     */
    static void save(CacheOrchestrator& orchestrator,
                     const SequenceGroup::CPtr& seq_group,
                     const Sequence::CPtr& sequence,
                     const std::filesystem::path& path);

    /// A block of the sequence's prefix chain, addressed the way the prefix-caching offload store keys it.
    struct PrefixEntry {
        uint64_t content_len;
        uint64_t hash;
    };

    /// History and block references loaded from a snapshot.
    struct Loaded {
        bool is_embeddings = false;

        // Populated when !is_embeddings (SequenceGroupType::TOKENS).
        std::vector<int64_t> token_ids;

        // Populated when is_embeddings (SequenceGroupType::EMBEDDINGS).
        size_t hidden_size = 0;
        ov::Tensor input_embeds;   // element type f32, shape [1, num_positions, hidden_size]
        ov::Tensor position_ids;   // element type i64, shape [..., 1, num_positions]
        int64_t rope_delta = 0;

        std::vector<PrefixEntry> kv_entries;
        /// The linear-attention checkpoint at the last computed position, for models that have that cache.
        std::optional<PrefixEntry> la_entry;
    };

    /**
     * @brief Reads and validates `path` against the orchestrator's currently loaded model/device.
     * @throws ov::Exception on format, checksum, or model/device compatibility mismatch.
     */
    static Loaded load(const CacheOrchestrator& orchestrator, const std::filesystem::path& path);
};


}  // namespace ov::genai
