// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "content_hash.hpp"

namespace ov::genai {

/**
 * @brief Identity of one block for a store: the prefix-cache hash plus, when known, the 128-bit digest behind it.
 *
 * A store that keeps blocks beyond this process keys them by the digest, which is wide and reproducible, and takes no
 * block without one. The size_t hash is only the identity inside this process, for example of a block whose digest is
 * unknown because it was hashed by a hash provider.
 */
struct PrefixKey {
    size_t hash = 0;
    std::optional<ContentHash> digest;
};

/**
 * @brief What one prefix restore of one request found for one cache group, in tokens.
 *
 * For a group of state checkpoints (linear attention) a count is the position of the checkpoint, not the length of a
 * contiguous run of blocks.
 */
struct RestoreStats {
    size_t prompt_tokens = 0;      ///< length of the request's prompt
    size_t device_hit_tokens = 0;  ///< already in the device cache when the restore began
    size_t store_hit_tokens = 0;   ///< read from the store and written to the device cache, all rounds together
    size_t reused_tokens = 0;      ///< prefix length the request finally uses
};

/// The tokens a state checkpoint (linear attention) covers: the whole prompt prefix from position 0 up to it.
struct PrefixTokens {
    PrefixKey key;
    std::shared_ptr<const std::vector<int64_t>> token_ids;
};

/**
 * @brief Host-internal, key-addressed interface to a KV/linear-attention cache block store.
 *
 * A block is identified purely by its PrefixKey; implementations own whatever indexing, placement and eviction scheme
 * they like internally, and callers never see or manage a physical storage location. Every store evicts by its own
 * policy when it is full: a write never fails for lack of capacity, and any block written earlier may be gone by the
 * time it is asked for again, in which case the caller recomputes it.
 *
 * This is the in-tree C++ seam behind KVCacheOffloadCache / LinearAttentionOffloadCache. It mirrors the addressing
 * model of the stable third-party plugin ABI in kv_cache_store_provider_abi.h but is a plain C++ interface, not a
 * frozen binary contract.
 */
class IKVCacheStore {
public:
    virtual ~IKVCacheStore() = default;

    /**
     * @brief Probes a whole prefix chain in one call.
     * @param chain The keys of the blocks of one prompt from block 0, in order. A plugin numbers the blocks by their
     * place in this list, so callers must always pass a complete chain and never a bare block.
     * @return Whether each block of @p chain is stored.
     */
    virtual std::vector<bool> has_blocks(const std::vector<PrefixKey>& chain) const = 0;

    /**
     * @brief Reads several blocks in one call.
     * @param data Resized to @p keys; entry i holds block i when it was found and is empty otherwise.
     * @return Whether each block of @p keys was found.
     */
    virtual std::vector<bool> read_blocks(const std::vector<PrefixKey>& keys,
                                          std::vector<std::vector<uint8_t>>& data) const = 0;

    /**
     * @brief Persists several blocks in one call, replacing any existing entry for the same key.
     * @param data One buffer per entry of @p keys.
     * @return Per block whether it was stored. False means an I/O error or a rejected block (a persistent store takes
     * no block without a digest); never a full store.
     */
    virtual std::vector<bool> write_blocks(const std::vector<PrefixKey>& keys,
                                           const std::vector<const std::vector<uint8_t>*>& data) = 0;

    /// Quarantines the entry for @p key, e.g. a block that failed verification. Never used to make room. Not an error
    /// when missing.
    virtual void invalidate(const PrefixKey& key) = 0;

    /// @return Byte size of one block, as persisted by this store.
    virtual size_t block_bytes() const = 0;

    /// @return How many blocks this store aims to hold. A hint for diagnostics and for sizing tables; the store may hold
    /// more or fewer.
    virtual size_t capacity_blocks_hint() const = 0;

    /// Whether blocks stay in the store after this pipeline, so that they are keyed by the 128-bit digest.
    virtual bool is_persistent() const = 0;

    /// Whether the store wants a block written only after the chain of blocks before it was probed (see has_blocks()).
    virtual bool wants_chain_context() const = 0;

    /// Makes everything written so far durable. A synchronous write does not imply persistence for a buffering
    /// backend, so callers that report data as saved must flush first. Throws on failure.
    virtual void flush() {}

    /**
     * @brief Tells the store what a prefix restore found, once per request after the restore is decided.
     * @return Whether the store took the report and so reports the reuse metrics itself; when false the caller
     * reports them.
     */
    virtual bool report_restore_stats(const RestoreStats& /*stats*/) {
        return false;
    }

    /// Whether the store wants to know which tokens each linear-attention checkpoint covers. The caller then calls
    /// announce_prefix_tokens() before it probes or writes checkpoints.
    virtual bool wants_prefix_tokens() const {
        return false;
    }

    /// Tells the store the tokens of the checkpoints of the call that follows. A checkpoint whose tokens are no longer
    /// known is left out, and the store has to cope with that.
    virtual void announce_prefix_tokens(const std::vector<PrefixTokens>& /*prefixes*/) {}
};

/**
 * @brief The stores of all cache groups of one pipeline (KV, and linear attention for a hybrid model), opened together.
 *
 * Each group is used through an IKVCacheStore view. The views of a store that serves all groups from one instance
 * (a plugin with OV_KVCACHE_CAP_MULTI_GROUP) share that instance, so everything the pipeline writes reaches it in one
 * place; other backends hand out independent stores.
 */
class IGroupedKVCacheStore {
public:
    virtual ~IGroupedKVCacheStore() = default;

    virtual size_t group_count() const = 0;

    /// Hands out the view of group @p index; each view can be taken once.
    virtual std::unique_ptr<IKVCacheStore> group(size_t index) = 0;
};

/// A set of ready views, one per group.
class ViewSetKVCacheStore final : public IGroupedKVCacheStore {
public:
    explicit ViewSetKVCacheStore(std::vector<std::unique_ptr<IKVCacheStore>> views) : m_views(std::move(views)) {}

    size_t group_count() const override {
        return m_views.size();
    }

    std::unique_ptr<IKVCacheStore> group(size_t index) override {
        return std::move(m_views.at(index));
    }

private:
    std::vector<std::unique_ptr<IKVCacheStore>> m_views;
};

}  // namespace ov::genai
