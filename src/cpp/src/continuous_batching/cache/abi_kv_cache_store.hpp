// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "continuous_batching/cache/kv_cache_store.hpp"
#include "continuous_batching/cache/offload_settings.hpp"
#include "openvino/genai/c/kv_cache_store_provider_abi.h"

namespace ov::genai {

/**
 * @brief A plugin store instance, shared by the AbiKVCacheStore views of its cache groups.
 *
 * Flushes and destroys the instance when the last view is gone, unless the plugin never finished a cancelled batch.
 */
struct AbiStoreInstance {
    AbiStoreInstance(const OvKVCacheStoreVTable* vtable, void* store) : vtable(vtable), store(store) {}
    ~AbiStoreInstance();

    AbiStoreInstance(const AbiStoreInstance&) = delete;
    AbiStoreInstance& operator=(const AbiStoreInstance&) = delete;

    /// Makes what was written durable. Throws on failure.
    void flush();

    const OvKVCacheStoreVTable* vtable;
    void* store;
    /// Set once the plugin holds buffers of a batch it never finished: the instance is then never touched again.
    std::atomic<bool> abandoned{false};
};

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
 *
 * One instance can serve several cache groups (OV_KVCACHE_CAP_MULTI_GROUP): each group is a view with its own group
 * name, block size and capacity over the shared AbiStoreInstance.
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
                    std::uint32_t io_timeout_ms = OffloadSettings::DEFAULT_STORE_TIMEOUT_MS);

    /// A view of @p group over an instance that other views share.
    AbiKVCacheStore(std::shared_ptr<AbiStoreInstance> instance,
                    std::string group,
                    size_t block_size,
                    size_t capacity_blocks,
                    StoreInfo info,
                    std::uint32_t io_timeout_ms = OffloadSettings::DEFAULT_STORE_TIMEOUT_MS);
    ~AbiKVCacheStore() override = default;

    AbiKVCacheStore(const AbiKVCacheStore&) = delete;
    AbiKVCacheStore& operator=(const AbiKVCacheStore&) = delete;

    /// The plugin addresses a block by its 128-bit digest, so a block without one is never stored or found.
    std::vector<bool> has_blocks(const std::vector<PrefixKey>& chain) const override;
    std::vector<bool> write_blocks(const std::vector<PrefixKey>& keys,
                                   const std::vector<const std::vector<uint8_t>*>& data) override;
    std::vector<bool> read_blocks(const std::vector<PrefixKey>& keys,
                                  std::vector<std::vector<uint8_t>>& data) const override;
    void invalidate(const PrefixKey& key) override;

    size_t block_bytes() const override;
    size_t capacity_blocks_hint() const override;
    void flush() override;

    /// A plugin store is assumed to outlive the pipeline and wants the chain before a write (ABI contract).
    bool is_persistent() const override {
        return true;
    }

    bool wants_chain_context() const override {
        return true;
    }

    /// How many more blocks the plugin reports room for; informational.
    size_t get_num_free_blocks() const;

    /// Calls the plugin's `on_restore_finish`. @return false when the plugin has no such entry.
    bool report_restore_stats(const RestoreStats& stats) override;

    bool wants_prefix_tokens() const override {
        return m_takes_prefix_tokens;
    }

    /// Calls the plugin's `announce_prefix_tokens`; a failure is logged and otherwise ignored.
    void announce_prefix_tokens(const std::vector<PrefixTokens>& prefixes) override;

    /// Whether the plugin takes restore reports.
    bool takes_restore_stats() const {
        return m_takes_restore_stats;
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

    static OvKVCacheBlockKey make_key(const std::string& group, const PrefixKey& key);

    /// Runs one batch to completion. Through the async entry points the batch is waited on for the I/O timeout, then
    /// cancelled and waited on for one more timeout; the status is that of the batch, and CANCELLED means that it
    /// ended without results. A batch that is still not terminal then is kept alive for good, the store is given up,
    /// and the status is PENDING.
    OvKVCacheStatus run_reads(const std::shared_ptr<ReadBatch>& batch) const;
    OvKVCacheStatus run_writes(const std::shared_ptr<WriteBatch>& batch) const;
    OvKVCacheStatus await_token(OvKVCacheAsyncToken token, const std::shared_ptr<void>& batch) const;
    void give_up(const std::shared_ptr<void>& batch) const;

    std::shared_ptr<AbiStoreInstance> m_instance;
    const OvKVCacheStoreVTable* m_vtable;
    void* m_store;
    std::string m_group;
    size_t m_block_size;
    size_t m_capacity_blocks;
    StoreInfo m_info;
    std::uint32_t m_io_timeout_ms;
    bool m_use_async = false;
    bool m_takes_restore_stats = false;
    bool m_takes_prefix_tokens = false;
    std::atomic<bool>& m_abandoned;
};

}  // namespace ov::genai
