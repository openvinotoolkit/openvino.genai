// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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
     * @brief Persists @p data under @p hash, replacing any existing entry for the same hash.
     * @return false when the store has no free capacity; the caller must `drop_block` an
     * existing entry and retry.
     */
    virtual bool write_block(size_t hash, const std::vector<uint8_t>& data) = 0;

    /// @return false if @p hash has no entry, in which case @p data is left untouched.
    virtual bool read_block(size_t hash, std::vector<uint8_t>& data) const = 0;

    /// Drops the entry for @p hash, if any. Not an error when missing.
    virtual void drop_block(size_t hash) = 0;

    /// @return Byte size of one block, as persisted by this store.
    virtual size_t get_block_size() const = 0;

    /// @return Maximum number of blocks this store can hold at once.
    virtual size_t get_capacity_blocks() const = 0;

    /// @return Number of additional blocks that can be stored without dropping an existing one.
    virtual size_t get_num_free_blocks() const = 0;
};

}  // namespace ov::genai
