// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "continuous_batching/cache/kv_cache_store.hpp"
#include "openvino/genai/c/kv_cache_store_provider_abi.h"

namespace ov::genai {

/**
 * @brief IKVCacheStore adapter around an ABI store instance.
 *
 * Wraps an (OvKVCacheStoreVTable*, void* store) pair obtained from a provider's `open()` - whether
 * the in-tree default (AbiFileStoreProvider) or, eventually, a dynamically loaded third-party
 * plugin - so it can be used anywhere an IKVCacheStore is expected. Computes and verifies its own
 * witness (a content hash of the payload bytes, independent of the caller-supplied key) on every
 * write/read: the ABI's witness is host-defined and host-verified, never trusted from the plugin
 * (see kv_cache_store_provider_abi.h).
 */
class AbiKVCacheStore final : public IKVCacheStore {
public:
    AbiKVCacheStore(const OvKVCacheStoreVTable* vtable,
                    void* store,
                    std::string group,
                    size_t block_size,
                    size_t capacity_blocks);
    ~AbiKVCacheStore() override;

    AbiKVCacheStore(const AbiKVCacheStore&) = delete;
    AbiKVCacheStore& operator=(const AbiKVCacheStore&) = delete;

    bool has_block(size_t hash) const override;
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override;
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override;
    void drop_block(size_t hash) override;

    size_t get_block_size() const override;
    size_t get_capacity_blocks() const override;
    size_t get_num_free_blocks() const override;

private:
    OvKVCacheBlockKey make_key(size_t hash) const;

    const OvKVCacheStoreVTable* m_vtable;
    void* m_store;
    std::string m_group;
    size_t m_block_size;
    size_t m_capacity_blocks;
};

}  // namespace ov::genai
