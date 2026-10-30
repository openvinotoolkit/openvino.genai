// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "continuous_batching/cache/kv_cache_offload_manager.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"

namespace ov::genai {

/**
 * @brief Default, in-tree IKVCacheStore backend.
 *
 * Fronts a fixed-slot local file (KVCacheOffloadManager) with a hash -> slot index, so that
 * callers only ever deal in content hashes. This index is owned entirely by this class, exactly
 * as a third-party store plugin would own its own index behind kv_cache_store_provider_abi.h: it
 * is the built-in equivalent of that ABI's default "file" provider.
 */
class FileKVCacheStore final : public IKVCacheStore {
public:
    explicit FileKVCacheStore(std::unique_ptr<KVCacheOffloadManager> backend);

    bool has_block(size_t hash) const override;
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override;
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override;
    void drop_block(size_t hash) override;

    size_t get_block_size() const override;
    size_t get_capacity_blocks() const override;
    size_t get_num_free_blocks() const override;

private:
    std::unique_ptr<KVCacheOffloadManager> m_backend;
    std::unordered_map<size_t, size_t> m_hash_to_slot;
    mutable std::mutex m_mutex;
};

}  // namespace ov::genai
