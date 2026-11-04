// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ov::genai {

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
};

}  // namespace ov::genai
