// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "continuous_batching/cache/kv_cache_store.hpp"
#include "openvino/genai/cache_offload.hpp"
#include "openvino/genai/c/kv_cache_store_provider_abi.h"

namespace ov::genai {

/**
 * @brief IKVCacheStore adapter around an ABI store instance.
 *
 * Wraps an (OvKVCacheStoreVTable*, void* store) pair obtained from a provider's `open()` - whether
 * the in-tree default (AbiFileStoreProvider) or, eventually, a dynamically loaded third-party
 * plugin - so it can be used anywhere an IKVCacheStore is expected. Computes and verifies its own
 * witness (a content hash of the payload bytes seeded with the requested key) on every write/read,
 * so a store that returns a different but internally-consistent block for the requested key is
 * caught, not just bit-level corruption: the ABI's witness is host-defined and host-verified,
 * never trusted from the plugin (see kv_cache_store_provider_abi.h).
 */
class AbiKVCacheStore final : public IKVCacheStore {
public:
    /// What the provider and its `open` told the host about the store.
    struct StoreInfo {
        OvKVCacheCapabilities capabilities = 0;
        std::optional<OvKVCacheContentHash> semantics_fingerprint;
    };

    AbiKVCacheStore(const OvKVCacheStoreVTable* vtable,
                    void* store,
                    std::string group,
                    size_t block_size,
                    size_t capacity_blocks,
                    StoreInfo info = StoreInfo{0, std::nullopt},
                    std::uint32_t io_timeout_ms = CacheOffloadConfig::DEFAULT_STORE_TIMEOUT_MS);
    ~AbiKVCacheStore() override;

    AbiKVCacheStore(const AbiKVCacheStore&) = delete;
    AbiKVCacheStore& operator=(const AbiKVCacheStore&) = delete;

    bool has_block(size_t hash) const override;
    std::vector<bool> has_blocks(const std::vector<size_t>& chain) const override;
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override;
    std::vector<bool> write_blocks(const std::vector<size_t>& hashes,
                                   const std::vector<const std::vector<uint8_t>*>& data) override;
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override;
    std::vector<bool> read_blocks(const std::vector<size_t>& hashes, std::vector<std::vector<uint8_t>>& data) const override;
    void drop_block(size_t hash) override;

    size_t get_block_size() const override;
    size_t get_capacity_blocks() const override;
    size_t get_num_free_blocks() const override;
    void flush() override;

    bool is_plugin_backed() const override {
        return true;
    }

    bool evicts_itself() const override {
        return (m_info.capabilities & OV_KVCACHE_CAP_SELF_EVICT) != 0;
    }

    /// Whether reads and writes go through the plugin's async entry points.
    bool uses_async_io() const {
        return m_use_async;
    }

    /// Whether the plugin never finished a cancelled batch, after which every request to the store is a miss.
    bool is_given_up() const {
        return m_abandoned;
    }

    const StoreInfo& get_info() const {
        return m_info;
    }

private:
    struct ReadBatch;
    struct WriteBatch;

    static OvKVCacheBlockKey make_key(const std::string& group, size_t hash);

    /// Runs one batch to completion. Through the async entry points the batch is waited on for the I/O timeout, then
    /// cancelled and waited on for one more timeout; the status is that of the batch, and CANCELLED means that it
    /// ended without results. A batch that is still not terminal then is kept alive for good, the store is given up,
    /// and the status is PENDING.
    OvKVCacheStatus run_reads(const std::shared_ptr<ReadBatch>& batch) const;
    OvKVCacheStatus run_writes(const std::shared_ptr<WriteBatch>& batch) const;
    OvKVCacheStatus await_token(OvKVCacheAsyncToken token, const std::shared_ptr<void>& batch) const;
    void give_up(const std::shared_ptr<void>& batch) const;

    const OvKVCacheStoreVTable* m_vtable;
    void* m_store;
    std::string m_group;
    size_t m_block_size;
    size_t m_capacity_blocks;
    StoreInfo m_info;
    std::uint32_t m_io_timeout_ms;
    bool m_use_async = false;
    mutable std::atomic<bool> m_abandoned{false};
};

}  // namespace ov::genai
