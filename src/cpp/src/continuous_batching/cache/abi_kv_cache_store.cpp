// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/abi_kv_cache_store.hpp"

#include <cstddef>
#include <cstring>
#include <utility>

#include "content_hash.hpp"
#include "logger.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {
// KNOWN LIMITATION: Sequence::_make_hash() (sequence_group.cpp) already folds the full 128-bit
// ContentHash digest down to a single size_t via XOR before it ever reaches this layer, so the
// value mirrored into both halves below carries only 64 bits of real identity, not the 128 the
// wire struct's width implies; distinct digests that fold to the same size_t alias here too.
// Fixing this requires widening Sequence::get_hash()/the prefix-cache hash maps and the snapshot
// file format to carry ContentHash end to end, which is out of scope for this change.
OvKVCacheContentHash to_abi_hash(size_t hash) {
    return OvKVCacheContentHash{static_cast<uint64_t>(hash), static_cast<uint64_t>(hash)};
}
}  // namespace

AbiKVCacheStore::AbiKVCacheStore(const OvKVCacheStoreVTable* vtable,
                                 void* store,
                                 std::string group,
                                 size_t block_size,
                                 size_t capacity_blocks,
                                 StoreInfo info)
    : m_vtable(vtable),
      m_store(store),
      m_group(std::move(group)),
      m_block_size(block_size),
      m_capacity_blocks(capacity_blocks),
      m_info(info) {
    OPENVINO_ASSERT(m_vtable != nullptr && m_store != nullptr, "ABI KV cache store must not be null");
    OPENVINO_ASSERT(m_vtable->struct_bytes >= offsetof(OvKVCacheStoreVTable, read_blocks_async),
                    "ABI store vtable is missing required synchronous entries");

    if ((m_info.capabilities & OV_KVCACHE_CAP_ASYNC_IO) != 0) {
        // submit and wait are enough for the host; cancel is optional and never needed because every token is waited on.
        constexpr size_t async_end = offsetof(OvKVCacheStoreVTable, wait) + sizeof(OvKVCacheStoreVTable::wait);
        m_use_async = m_vtable->struct_bytes >= async_end && m_vtable->read_blocks_async != nullptr &&
                      m_vtable->write_blocks_async != nullptr && m_vtable->wait != nullptr;
        if (!m_use_async) {
            GENAI_WARN("KV cache store plugin advertises async I/O but its vtable is incomplete; using synchronous I/O");
        }
    }
}

AbiKVCacheStore::~AbiKVCacheStore() {
    // Teardown order of the plugin contract: make everything written durable, then destroy the instance.
    try {
        flush();
    } catch (const std::exception& error) {
        GENAI_WARN("KV cache store plugin flush before destroy failed: %s", error.what());
    }
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

std::vector<bool> AbiKVCacheStore::has_blocks(const std::vector<size_t>& chain) const {
    if (chain.empty()) {
        return {};
    }
    std::vector<OvKVCacheBlockKey> keys;
    keys.reserve(chain.size());
    for (const size_t hash : chain) {
        keys.push_back(make_key(hash));
    }
    std::vector<uint8_t> present(chain.size(), 0);
    const OvKVCacheStatus status = m_vtable->has_blocks(m_store, keys.data(), present.data(), keys.size());
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store has_blocks failed with status ", status);
    return std::vector<bool>(present.begin(), present.end());
}

bool AbiKVCacheStore::write_block(size_t hash, const std::vector<uint8_t>& data) {
    return write_blocks({hash}, {&data})[0];
}

std::vector<bool> AbiKVCacheStore::write_blocks(const std::vector<size_t>& hashes,
                                                const std::vector<const std::vector<uint8_t>*>& data) {
    OPENVINO_ASSERT(hashes.size() == data.size(), "ABI store write_blocks needs one buffer per hash");
    const size_t count = hashes.size();
    std::vector<bool> written(count, false);
    if (count == 0) {
        return written;
    }

    std::vector<ContentHash> digests(count);
    std::vector<OvKVCacheStoreWriteRequest> requests(count);
    std::vector<OvKVCacheStoreWriteResult> results(count);
    for (size_t i = 0; i < count; ++i) {
        OPENVINO_ASSERT(data[i]->size() == m_block_size, "ABI store block size mismatch");
        // Seeding with the key's hash binds the witness to the key it was written under, so a store that
        // returns a different (but internally valid) block for a read of this key is still caught below.
        digests[i] = compute_content_hash(data[i]->data(), data[i]->size(), static_cast<uint64_t>(hashes[i]));
        requests[i].key = make_key(hashes[i]);
        requests[i].source = data[i]->data();
        requests[i].bytes = data[i]->size();
        requests[i].witness.data = &digests[i];
        requests[i].witness.len = sizeof(ContentHash);
    }

    const OvKVCacheStatus status = run_writes(requests.data(), results.data(), count);
    const bool batch_ok = status == OV_KVCACHE_OK || status == OV_KVCACHE_PARTIAL;
    for (size_t i = 0; i < count; ++i) {
        if (results[i].status == OV_KVCACHE_ERR_CAPACITY) {
            continue;
        }
        OPENVINO_ASSERT(batch_ok && results[i].status == OV_KVCACHE_OK,
                        "ABI store write_blocks failed with status ",
                        results[i].status);
        written[i] = true;
    }
    return written;
}

bool AbiKVCacheStore::read_block(size_t hash, std::vector<uint8_t>& data) const {
    std::vector<std::vector<uint8_t>> blocks;
    if (!read_blocks({hash}, blocks)[0]) {
        return false;
    }
    data = std::move(blocks[0]);
    return true;
}

std::vector<bool> AbiKVCacheStore::read_blocks(const std::vector<size_t>& hashes,
                                               std::vector<std::vector<uint8_t>>& data) const {
    const size_t count = hashes.size();
    std::vector<bool> found(count, false);
    data.assign(count, {});
    if (count == 0) {
        return found;
    }

    // Read into scratch buffers so a not-found or failed read leaves nothing half filled.
    std::vector<std::vector<uint8_t>> scratch(count, std::vector<uint8_t>(m_block_size, 0));
    std::vector<ContentHash> returned_witnesses(count);
    std::vector<OvKVCacheStoreReadRequest> requests(count);
    std::vector<OvKVCacheStoreReadResult> results(count);
    for (size_t i = 0; i < count; ++i) {
        requests[i].key = make_key(hashes[i]);
        requests[i].target.kind = OV_KVCACHE_TARGET_HOST;
        requests[i].target.base = scratch[i].data();
        requests[i].target.bytes = scratch[i].size();
        requests[i].witness_out.data = &returned_witnesses[i];
        requests[i].witness_out.capacity = sizeof(ContentHash);
    }

    const OvKVCacheStatus status = run_reads(requests.data(), results.data(), count);
    const bool batch_ok = status == OV_KVCACHE_OK || status == OV_KVCACHE_PARTIAL;
    for (size_t i = 0; i < count; ++i) {
        if (results[i].status == OV_KVCACHE_ERR_NOT_FOUND) {
            continue;
        }
        OPENVINO_ASSERT(batch_ok && results[i].status == OV_KVCACHE_OK,
                        "ABI store read_blocks failed with status ",
                        results[i].status);
        OPENVINO_ASSERT(results[i].bytes_read == scratch[i].size(), "ABI store returned a short read");

        const ContentHash expected_witness =
            compute_content_hash(scratch[i].data(), scratch[i].size(), static_cast<uint64_t>(hashes[i]));
        OPENVINO_ASSERT(results[i].witness.len == sizeof(expected_witness) &&
                            std::memcmp(&expected_witness, &returned_witnesses[i], sizeof(expected_witness)) == 0,
                        "ABI store witness mismatch for a read block - payload may be corrupted");
        data[i] = std::move(scratch[i]);
        found[i] = true;
    }
    return found;
}

OvKVCacheStatus AbiKVCacheStore::await_token(OvKVCacheAsyncToken token) const {
    OvKVCacheStatus status = OV_KVCACHE_PENDING;
    while (status == OV_KVCACHE_PENDING) {
        status = m_vtable->wait(m_store, token, OV_KVCACHE_WAIT_INFINITE);
    }
    return status;
}

OvKVCacheStatus AbiKVCacheStore::run_reads(const OvKVCacheStoreReadRequest* reqs,
                                           OvKVCacheStoreReadResult* out,
                                           size_t count) const {
    if (!m_use_async) {
        return m_vtable->read_blocks(m_store, reqs, out, count);
    }
    OvKVCacheAsyncToken token = 0;
    const OvKVCacheStatus submitted = m_vtable->read_blocks_async(m_store, reqs, out, count, &token);
    OPENVINO_ASSERT(submitted == OV_KVCACHE_OK && token != 0, "ABI store read_blocks_async failed with status ", submitted);
    // The buffers stay untouched until the token is terminal; a failed or cancelled batch has no usable items.
    const OvKVCacheStatus finished = await_token(token);
    OPENVINO_ASSERT(finished == OV_KVCACHE_OK || finished == OV_KVCACHE_PARTIAL,
                    "ABI store async read ended with status ",
                    finished);
    return finished;
}

OvKVCacheStatus AbiKVCacheStore::run_writes(const OvKVCacheStoreWriteRequest* reqs,
                                            OvKVCacheStoreWriteResult* out,
                                            size_t count) const {
    if (!m_use_async) {
        return m_vtable->write_blocks(m_store, reqs, out, count);
    }
    OvKVCacheAsyncToken token = 0;
    const OvKVCacheStatus submitted = m_vtable->write_blocks_async(m_store, reqs, out, count, &token);
    OPENVINO_ASSERT(submitted == OV_KVCACHE_OK && token != 0, "ABI store write_blocks_async failed with status ", submitted);
    const OvKVCacheStatus finished = await_token(token);
    OPENVINO_ASSERT(finished == OV_KVCACHE_OK || finished == OV_KVCACHE_PARTIAL,
                    "ABI store async write ended with status ",
                    finished);
    return finished;
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
    if (evicts_itself()) {
        // The plugin sizes itself; the host's capacity estimate is only a hint and may be exceeded.
        return stats.blocks >= m_capacity_blocks ? 0 : m_capacity_blocks - stats.blocks;
    }
    OPENVINO_ASSERT(stats.blocks <= m_capacity_blocks, "ABI store reported more used blocks than its capacity");
    return m_capacity_blocks - stats.blocks;
}

void AbiKVCacheStore::flush() {
    // flush is a tail entry of the vtable: only call it when struct_bytes says the plugin provides it.
    constexpr size_t flush_end = offsetof(OvKVCacheStoreVTable, flush) + sizeof(OvKVCacheStoreVTable::flush);
    if (m_vtable->struct_bytes < flush_end || m_vtable->flush == nullptr) {
        return;
    }
    const OvKVCacheStatus status = m_vtable->flush(m_store);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store flush failed with status ", status);
}

}  // namespace ov::genai
