// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <vector>

#include "continuous_batching/cache/cache_type.hpp"
#include "openvino/genai/cache_offload.hpp"
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
     * @param store_config When `allow_untrusted_store_plugin` is set and `store_library_path` is
     * non-empty, KV/linear-attention block payloads are written through that third-party plugin
     * (the SAME plugin configuration as `SchedulerConfig::cache_offload_config`, reused here)
     * instead of being embedded inline in `path`; `load()` must then be given a matching config.
     * Left default-constructed (the default), the payload is embedded inline as before - no ABI
     * involved. Only `store_library_path`/`store_provider_name`/`allow_untrusted_store_plugin`/
     * `path` are meaningful here; `capacity_bytes` is derived from the snapshot's own block
     * count and the rest (`buffer_slots`, `use_page_cache`, `linear_attention_capacity_bytes`)
     * are ignored, since this is a single synchronous save, not a queued eviction cache.
     */
    static void save(CacheOrchestrator& orchestrator,
                      const SequenceGroup::CPtr& seq_group,
                      const Sequence::CPtr& sequence,
                      const std::filesystem::path& path,
                      const CacheOffloadConfig& store_config = {});

    /**
     * @brief Whether save() writes the experimental v4 format: the snapshot file keeps only the token
     * history and the prefix hashes of the sequence's blocks, whose contents live in the pipeline's
     * prefix-caching offload store. Enabled by OV_GENAI_SESSION_SNAPSHOT_FORMAT=4 and only effective when
     * the orchestrator has a KV offload cache; load() dispatches on the file's own version instead.
     */
    static bool prefix_offload_format_requested(const CacheOrchestrator& orchestrator);

    /// A block of the sequence's prefix chain, addressed the way the prefix-caching offload store keys it.
    struct PrefixEntry {
        uint64_t content_len;
        uint64_t hash;
    };

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

        /// Set for the v4 format: `blocks_per_type` is empty and the cache contents must be brought back
        /// from the offload store through these entries (the restored request is a normal prompt).
        bool prefix_offload_format = false;
        std::vector<PrefixEntry> kv_entries;
        /// The linear-attention checkpoint at the last computed position, for models that have that cache.
        std::optional<PrefixEntry> la_entry;
    };

    /**
     * @brief Reads and validates `path` against the orchestrator's currently loaded model/device.
     * @param store_config Must match what `save()` was called with - if the snapshot was saved
     * with block payloads delegated to a store plugin, the same plugin configuration is required
     * to read them back; it is never recorded inside the snapshot file itself (a data file must
     * never be able to steer which shared library gets loaded).
     * @throws ov::Exception on format, checksum, or model/device compatibility mismatch.
     */
    static Loaded load(const CacheOrchestrator& orchestrator,
                        const std::filesystem::path& path,
                        const CacheOffloadConfig& store_config = {});
};


}  // namespace ov::genai
