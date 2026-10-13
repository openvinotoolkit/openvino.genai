// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <vector>

#include "continuous_batching/cache/cache_type.hpp"
#include "sequence_group.hpp"

namespace ov::genai {

class CacheOrchestrator;

/**
 * @brief Self-contained on-disk snapshot of a single sequence's full cache state (KV blocks in
 * full and, for hybrid models, the latest linear-attention checkpoint), used to resume a
 * conversation across process restarts without re-running prefill.
 *
 * Unlike the opportunistic/LRU disk offload (KVCacheOffloadCache / LinearAttentionOffloadCache),
 * loading a snapshot does not depend on a new request's prompt re-tokenizing to an identical
 * token sequence - the saved token ids (or, for embeddings-routed models, embeddings) are the
 * ground truth, restored directly.
 *
 * Supports both plain token-id sequences (`SequenceGroupType::TOKENS`) and, for text-only
 * requests, embeddings-routed sequences (`SequenceGroupType::EMBEDDINGS`, e.g. natively
 * multimodal models such as Qwen3.5 that route even text-only prompts through an input
 * embeddings model). Embeddings-routed requests that carry multimodal state (images/video in the
 * history, `token_type_ids`, deepstack visual embeds, etc.) are rejected at save() time - only
 * the plain-text subset of that path is supported.
 *
 * V1 scope: single-sequence groups only (no beam search / forked sequences) and
 * `use_cache_eviction` must be disabled - both are validated at save() time.
 */
class SessionSnapshot {
public:
    /**
     * @brief Captures `sequence`'s current cache contents and full history (token ids, or for
     * embeddings-routed models, embeddings + position ids + rope delta) to `path`.
     *
     * KV cache is saved in full (one entry per logical block, in token order); linear-attention
     * cache is saved as only its single latest checkpoint block, since the recurrent state is
     * already a compressed summary of all prior history and does not need the full chain.
     *
     * @param orchestrator The cache orchestrator owning `sequence`'s physical blocks.
     * @param seq_group    The (single-sequence) group `sequence` belongs to.
     * @param sequence     The sequence to snapshot.
     * @param path         Destination file path. Overwritten if it already exists.
     */
    static void save(CacheOrchestrator& orchestrator,
                      const SequenceGroup::CPtr& seq_group,
                      const Sequence::CPtr& sequence,
                      const std::filesystem::path& path);

    /// Cache payload loaded from a snapshot, keyed by cache type, each a list of blocks in the
    /// order they must be written back (KV: full logical order; linear-attention: single entry).
    struct Loaded {
        bool is_embeddings = false;

        // Populated when !is_embeddings (SequenceGroupType::TOKENS).
        std::vector<int64_t> token_ids;

        // Populated when is_embeddings (SequenceGroupType::EMBEDDINGS).
        size_t hidden_size = 0;
        ov::Tensor input_embeds;   // element type f32, shape [1, num_positions, hidden_size]
        ov::Tensor position_ids;   // element type i64, shape [..., 1, num_positions]
        int64_t rope_delta = 0;

        std::map<CacheType, std::vector<std::vector<uint8_t>>> blocks_per_type;
    };

    /**
     * @brief Reads and validates `path` against the orchestrator's currently loaded model/device.
     * @throws ov::Exception on format, checksum, or model/device compatibility mismatch.
     */
    static Loaded load(const CacheOrchestrator& orchestrator, const std::filesystem::path& path);
};


}  // namespace ov::genai
