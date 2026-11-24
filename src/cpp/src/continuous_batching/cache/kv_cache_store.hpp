// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "content_hash.hpp"

namespace ov::genai {

/**
 * @brief Identity of one block for a store: the prefix-cache hash plus, when known, the 128-bit digest behind it.
 *
 * A store that keeps blocks beyond this process keys them by the digest, which is wide and reproducible. The size_t
 * hash is the fallback identity for a block whose digest is unknown, for example one hashed by a hash provider.
 */
struct PrefixKey {
    size_t hash = 0;
    std::optional<ContentHash> digest;

    /// @return The 128 bits a persistent store addresses the block by.
    ContentHash wire_id() const {
        if (digest) {
            return *digest;
        }
        return ContentHash{static_cast<uint64_t>(hash), static_cast<uint64_t>(hash)};
    }
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

/**
 * @brief Host-internal, key-addressed interface to a KV/linear-attention cache block store.
 *
 * A block is identified purely by its content hash; implementations own whatever indexing,
 * placement and eviction scheme they like internally, and callers never see or manage a physical
 * storage location. This is the in-tree C++ seam sitting behind KVCacheOffloadCache /
 * LinearAttentionOffloadCache. It mirrors, in spirit, the addressing model of the stable
 * third-party plugin ABI in kv_cache_store_provider_abi.h (key-addressed, store owns its own
 * index) but is itself a plain C++ interface, not a frozen binary contract.
 */
class IKVCacheStore {
public:
    virtual ~IKVCacheStore() = default;

    /// @return Whether a block with this hash is already stored.
    virtual bool has_block(size_t hash) const = 0;

    /**
     * @brief Probes a whole prefix chain in one call.
     * @param chain The hashes of the blocks of one prompt from block 0, in order. A plugin numbers the blocks by their
     * place in this list, so callers must always pass a complete chain and never a bare block.
     * @return Whether each block of @p chain is stored.
     */
    virtual std::vector<bool> has_blocks(const std::vector<size_t>& chain) const {
        std::vector<bool> present;
        present.reserve(chain.size());
        for (const size_t hash : chain) {
            present.push_back(has_block(hash));
        }
        return present;
    }

    /**
     * @brief Persists @p data under @p hash, replacing any existing entry for the same hash.
     * @return false when the store has no free capacity; the caller must `drop_block` an
     * existing entry and retry.
     */
    virtual bool write_block(size_t hash, const std::vector<uint8_t>& data) = 0;

    /**
     * @brief Persists several blocks in one call.
     * @param data One buffer per entry of @p hashes.
     * @return Per block, what write_block would have returned.
     */
    virtual std::vector<bool> write_blocks(const std::vector<size_t>& hashes,
                                           const std::vector<const std::vector<uint8_t>*>& data) {
        std::vector<bool> written;
        written.reserve(hashes.size());
        for (size_t i = 0; i < hashes.size(); ++i) {
            written.push_back(write_block(hashes[i], *data[i]));
        }
        return written;
    }

    /// @return false if @p hash has no entry, in which case @p data is left untouched.
    virtual bool read_block(size_t hash, std::vector<uint8_t>& data) const = 0;

    /**
     * @brief Reads several blocks in one call.
     * @param data Resized to @p hashes; entry i holds block i when it was found and is empty otherwise.
     * @return Whether each block of @p hashes was found.
     */
    virtual std::vector<bool> read_blocks(const std::vector<size_t>& hashes, std::vector<std::vector<uint8_t>>& data) const {
        std::vector<bool> found;
        found.reserve(hashes.size());
        data.assign(hashes.size(), {});
        for (size_t i = 0; i < hashes.size(); ++i) {
            found.push_back(read_block(hashes[i], data[i]));
        }
        return found;
    }

    /// Drops the entry for @p hash, if any. Not an error when missing.
    virtual void drop_block(size_t hash) = 0;

    /// @name Keyed variants
    /// The same operations, addressing blocks by PrefixKey. A store that cannot use the digest keeps these defaults,
    /// which address by PrefixKey::hash.
    /// @{
    virtual std::vector<bool> has_blocks_by_key(const std::vector<PrefixKey>& chain) const {
        return has_blocks(hashes_of(chain));
    }

    virtual std::vector<bool> write_blocks_by_key(const std::vector<PrefixKey>& keys,
                                                  const std::vector<const std::vector<uint8_t>*>& data) {
        return write_blocks(hashes_of(keys), data);
    }

    virtual std::vector<bool> read_blocks_by_key(const std::vector<PrefixKey>& keys,
                                                 std::vector<std::vector<uint8_t>>& data) const {
        return read_blocks(hashes_of(keys), data);
    }

    virtual void drop_block_by_key(const PrefixKey& key) {
        drop_block(key.hash);
    }
    /// @}

    /// @return Byte size of one block, as persisted by this store.
    virtual size_t get_block_size() const = 0;

    /// @return Maximum number of blocks this store can hold at once.
    virtual size_t get_capacity_blocks() const = 0;

    /// @return Number of additional blocks that can be stored without dropping an existing one.
    virtual size_t get_num_free_blocks() const = 0;

    /// Makes everything written so far durable. A synchronous write does not imply persistence for
    /// a buffering backend, so callers that report data as saved must flush first. Throws on failure.
    virtual void flush() {}

    /// Whether blocks live in a third-party plugin that may outlive this pipeline, as opposed to the in-tree
    /// run-scoped file store.
    virtual bool is_plugin_backed() const {
        return false;
    }

    /// Whether the store decides by itself which blocks to evict when it is full. The caller then never drops blocks
    /// to make room, and any block it stored earlier may be gone by the time it asks for it again.
    virtual bool evicts_itself() const {
        return false;
    }

    /**
     * @brief Tells the store what a prefix restore found, once per request after the restore is decided.
     * @return Whether the store took the report and so reports the reuse metrics itself; when false the caller
     * reports them.
     */
    virtual bool report_restore_stats(const RestoreStats& /*stats*/) {
        return false;
    }

protected:
    static std::vector<size_t> hashes_of(const std::vector<PrefixKey>& keys) {
        std::vector<size_t> hashes;
        hashes.reserve(keys.size());
        for (const PrefixKey& key : keys) {
            hashes.push_back(key.hash);
        }
        return hashes;
    }
};

}  // namespace ov::genai
