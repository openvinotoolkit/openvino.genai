// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/abi_kv_cache_store.hpp"

#include <cstddef>
#include <cstring>
#include <utility>

#include "content_hash.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {
OvKVCacheContentHash to_abi_hash(size_t hash) {
    // size_t keys at this layer are already a single folded 64-bit value (see content_hash.hpp);
    // mirroring it into both halves keeps the 128-bit wire key well-defined without a second hash.
    return OvKVCacheContentHash{static_cast<uint64_t>(hash), static_cast<uint64_t>(hash)};
}
}  // namespace

AbiKVCacheStore::AbiKVCacheStore(const OvKVCacheStoreVTable* vtable,
                                 void* store,
                                 std::string group,
                                 size_t block_size,
                                 size_t capacity_blocks)
    : m_vtable(vtable),
      m_store(store),
      m_group(std::move(group)),
      m_block_size(block_size),
      m_capacity_blocks(capacity_blocks) {
    OPENVINO_ASSERT(m_vtable != nullptr && m_store != nullptr, "ABI KV cache store must not be null");
    OPENVINO_ASSERT(m_vtable->struct_bytes >= offsetof(OvKVCacheStoreVTable, read_blocks_async),
                    "ABI store vtable is missing required synchronous entries");
}

AbiKVCacheStore::~AbiKVCacheStore() {
    if (m_vtable->destroy != nullptr) {
        m_vtable->destroy(m_store);
    }
}

OvKVCacheBlockKey AbiKVCacheStore::make_key(size_t hash) const {
    OvKVCacheBlockKey key{};
    key.group = m_group.c_str();
    key.group_len = static_cast<uint32_t>(m_group.size());
    key.hash = to_abi_hash(hash);
    return key;
}

bool AbiKVCacheStore::has_block(size_t hash) const {
    const OvKVCacheBlockKey key = make_key(hash);
    uint8_t present = 0;
    const OvKVCacheStatus status = m_vtable->has_blocks(m_store, &key, &present, 1);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store has_blocks failed with status ", status);
    return present != 0;
}

bool AbiKVCacheStore::write_block(size_t hash, const std::vector<uint8_t>& data) {
    OPENVINO_ASSERT(data.size() == m_block_size, "ABI store block size mismatch");
    const ContentHash digest = compute_content_hash(data.data(), data.size());

    OvKVCacheStoreWriteRequest request{};
    request.key = make_key(hash);
    request.source = data.data();
    request.bytes = data.size();
    request.witness.data = &digest;
    request.witness.len = sizeof(digest);

    OvKVCacheStoreWriteResult result{};
    const OvKVCacheStatus status = m_vtable->write_blocks(m_store, &request, &result, 1);
    if (result.status == OV_KVCACHE_ERR_CAPACITY) {
        return false;
    }
    OPENVINO_ASSERT(status == OV_KVCACHE_OK && result.status == OV_KVCACHE_OK,
                    "ABI store write_blocks failed with status ",
                    result.status);
    return true;
}

bool AbiKVCacheStore::read_block(size_t hash, std::vector<uint8_t>& data) const {
    data.assign(m_block_size, 0);
    ContentHash returned_witness{};

    OvKVCacheStoreReadRequest request{};
    request.key = make_key(hash);
    request.target.kind = OV_KVCACHE_TARGET_HOST;
    request.target.base = data.data();
    request.target.bytes = data.size();
    request.witness_out.data = &returned_witness;
    request.witness_out.capacity = sizeof(returned_witness);

    OvKVCacheStoreReadResult result{};
    const OvKVCacheStatus status = m_vtable->read_blocks(m_store, &request, &result, 1);
    if (result.status == OV_KVCACHE_ERR_NOT_FOUND) {
        return false;
    }
    OPENVINO_ASSERT(status == OV_KVCACHE_OK && result.status == OV_KVCACHE_OK,
                    "ABI store read_blocks failed with status ",
                    result.status);
    OPENVINO_ASSERT(result.bytes_read == data.size(), "ABI store returned a short read");

    const ContentHash expected_witness = compute_content_hash(data.data(), data.size());
    OPENVINO_ASSERT(result.witness.len == sizeof(expected_witness) &&
                        std::memcmp(&expected_witness, &returned_witness, sizeof(expected_witness)) == 0,
                    "ABI store witness mismatch for a read block - payload may be corrupted");
    return true;
}

void AbiKVCacheStore::drop_block(size_t hash) {
    const OvKVCacheBlockKey key = make_key(hash);
    const OvKVCacheStatus status = m_vtable->drop_blocks(m_store, &key, 1);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store drop_blocks failed with status ", status);
}

size_t AbiKVCacheStore::get_block_size() const {
    return m_block_size;
}

size_t AbiKVCacheStore::get_capacity_blocks() const {
    return m_capacity_blocks;
}

size_t AbiKVCacheStore::get_num_free_blocks() const {
    OvKVCacheStoreStats stats{};
    const OvKVCacheStatus status = m_vtable->stats(m_store, &stats);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store stats failed with status ", status);
    OPENVINO_ASSERT(stats.blocks <= m_capacity_blocks, "ABI store reported more used blocks than its capacity");
    return m_capacity_blocks - stats.blocks;
}

}  // namespace ov::genai
