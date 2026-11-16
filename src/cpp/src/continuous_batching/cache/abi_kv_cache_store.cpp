// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/abi_kv_cache_store.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

#include "content_hash.hpp"
#include "logger.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

/// What a read batch hands to the plugin. Owned by a shared pointer so that a batch the plugin never finishes can be
/// kept alive for good while the caller goes on.
struct AbiKVCacheStore::ReadBatch {
    ReadBatch(const std::string& group, size_t count, size_t block_size)
        : group(group),
          scratch(count, std::vector<uint8_t>(block_size, 0)),
          witnesses(count),
          requests(count),
          results(count) {}

    std::string group;
    std::vector<std::vector<uint8_t>> scratch;
    std::vector<ContentHash> witnesses;
    std::vector<OvKVCacheStoreReadRequest> requests;
    std::vector<OvKVCacheStoreReadResult> results;
};

struct AbiKVCacheStore::WriteBatch {
    WriteBatch(const std::string& group, size_t count)
        : group(group), digests(count), requests(count), results(count) {}

    std::string group;
    // Copies of the caller's blocks; empty for a synchronous call, which reads the caller's buffers directly.
    std::vector<std::vector<uint8_t>> owned_data;
    std::vector<ContentHash> digests;
    std::vector<OvKVCacheStoreWriteRequest> requests;
    std::vector<OvKVCacheStoreWriteResult> results;
};

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
                                 StoreInfo info,
                                 std::uint32_t io_timeout_ms)
    : m_vtable(vtable),
      m_store(store),
      m_group(std::move(group)),
      m_block_size(block_size),
      m_capacity_blocks(capacity_blocks),
      m_info(info),
      m_io_timeout_ms(io_timeout_ms) {
    OPENVINO_ASSERT(m_vtable != nullptr && m_store != nullptr, "ABI KV cache store must not be null");
    OPENVINO_ASSERT(m_io_timeout_ms > 0, "ABI KV cache store I/O timeout must be greater than 0");
    OPENVINO_ASSERT(m_vtable->struct_bytes >= offsetof(OvKVCacheStoreVTable, read_blocks_async),
                    "ABI store vtable is missing required synchronous entries");

    if ((m_info.capabilities & OV_KVCACHE_CAP_ASYNC_IO) != 0) {
        // A batch that outlives the I/O timeout has to be cancelled, so cancel belongs to the async entries here.
        constexpr size_t async_end = offsetof(OvKVCacheStoreVTable, cancel) + sizeof(OvKVCacheStoreVTable::cancel);
        m_use_async = m_vtable->struct_bytes >= async_end && m_vtable->read_blocks_async != nullptr &&
                      m_vtable->write_blocks_async != nullptr && m_vtable->wait != nullptr &&
                      m_vtable->cancel != nullptr;
        if (!m_use_async) {
            GENAI_WARN("KV cache store plugin advertises async I/O but its vtable lacks a submit, wait or cancel entry; "
                       "using synchronous I/O");
        }
    }
}

AbiKVCacheStore::~AbiKVCacheStore() {
    if (m_abandoned) {
        // The plugin still holds buffers of a batch it never finished, so the instance cannot be flushed or destroyed.
        GENAI_WARN("KV cache store plugin instance is left undestroyed because it never finished a cancelled batch");
        return;
    }
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

OvKVCacheBlockKey AbiKVCacheStore::make_key(const std::string& group, size_t hash) {
    OvKVCacheBlockKey key{};
    key.group = group.c_str();
    key.group_len = static_cast<uint32_t>(group.size());
    key.hash = to_abi_hash(hash);
    return key;
}

bool AbiKVCacheStore::has_block(size_t hash) const {
    if (m_abandoned) {
        return false;
    }
    const OvKVCacheBlockKey key = make_key(m_group, hash);
    uint8_t present = 0;
    const OvKVCacheStatus status = m_vtable->has_blocks(m_store, &key, &present, 1);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store has_blocks failed with status ", status);
    return present != 0;
}

std::vector<bool> AbiKVCacheStore::has_blocks(const std::vector<size_t>& chain) const {
    if (chain.empty() || m_abandoned) {
        return std::vector<bool>(chain.size(), false);
    }
    std::vector<OvKVCacheBlockKey> keys;
    keys.reserve(chain.size());
    for (const size_t hash : chain) {
        keys.push_back(make_key(m_group, hash));
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
    if (count == 0 || m_abandoned) {
        return written;
    }

    auto batch = std::make_shared<WriteBatch>(m_group, count);
    if (m_use_async) {
        // The plugin may still read the blocks after the host has given up waiting, so it gets copies the host never frees.
        batch->owned_data.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            batch->owned_data.push_back(*data[i]);
        }
    }
    for (size_t i = 0; i < count; ++i) {
        OPENVINO_ASSERT(data[i]->size() == m_block_size, "ABI store block size mismatch");
        const std::vector<uint8_t>& source = m_use_async ? batch->owned_data[i] : *data[i];
        // Seeding with the key's hash binds the witness to the key it was written under, so a store that
        // returns a different (but internally valid) block for a read of this key is still caught below.
        batch->digests[i] = compute_content_hash(source.data(), source.size(), static_cast<uint64_t>(hashes[i]));
        batch->requests[i].key = make_key(batch->group, hashes[i]);
        batch->requests[i].source = source.data();
        batch->requests[i].bytes = source.size();
        batch->requests[i].witness.data = &batch->digests[i];
        batch->requests[i].witness.len = sizeof(ContentHash);
    }

    const OvKVCacheStatus status = run_writes(batch);
    if (status == OV_KVCACHE_PENDING || status == OV_KVCACHE_ERR_CANCELLED) {
        return written;
    }
    const bool batch_ok = status == OV_KVCACHE_OK || status == OV_KVCACHE_PARTIAL;
    for (size_t i = 0; i < count; ++i) {
        if (batch->results[i].status == OV_KVCACHE_ERR_CAPACITY) {
            continue;
        }
        OPENVINO_ASSERT(batch_ok && batch->results[i].status == OV_KVCACHE_OK,
                        "ABI store write_blocks failed with status ",
                        batch->results[i].status);
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
    if (count == 0 || m_abandoned) {
        return found;
    }

    // Read into scratch buffers so a not-found or failed read leaves nothing half filled.
    auto batch = std::make_shared<ReadBatch>(m_group, count, m_block_size);
    for (size_t i = 0; i < count; ++i) {
        batch->requests[i].key = make_key(batch->group, hashes[i]);
        batch->requests[i].target.kind = OV_KVCACHE_TARGET_HOST;
        batch->requests[i].target.base = batch->scratch[i].data();
        batch->requests[i].target.bytes = batch->scratch[i].size();
        batch->requests[i].witness_out.data = &batch->witnesses[i];
        batch->requests[i].witness_out.capacity = sizeof(ContentHash);
    }

    const OvKVCacheStatus status = run_reads(batch);
    if (status == OV_KVCACHE_PENDING || status == OV_KVCACHE_ERR_CANCELLED) {
        return found;
    }
    const bool batch_ok = status == OV_KVCACHE_OK || status == OV_KVCACHE_PARTIAL;
    for (size_t i = 0; i < count; ++i) {
        std::vector<uint8_t>& scratch = batch->scratch[i];
        if (batch->results[i].status == OV_KVCACHE_ERR_NOT_FOUND) {
            continue;
        }
        OPENVINO_ASSERT(batch_ok && batch->results[i].status == OV_KVCACHE_OK,
                        "ABI store read_blocks failed with status ",
                        batch->results[i].status);
        OPENVINO_ASSERT(batch->results[i].bytes_read == scratch.size(), "ABI store returned a short read");

        const ContentHash expected_witness =
            compute_content_hash(scratch.data(), scratch.size(), static_cast<uint64_t>(hashes[i]));
        OPENVINO_ASSERT(batch->results[i].witness.len == sizeof(expected_witness) &&
                            std::memcmp(&expected_witness, &batch->witnesses[i], sizeof(expected_witness)) == 0,
                        "ABI store witness mismatch for a read block - payload may be corrupted");
        data[i] = std::move(scratch);
        found[i] = true;
    }
    return found;
}

OvKVCacheStatus AbiKVCacheStore::await_token(OvKVCacheAsyncToken token, const std::shared_ptr<void>& batch) const {
    OvKVCacheStatus status = m_vtable->wait(m_store, token, m_io_timeout_ms);
    if (status != OV_KVCACHE_PENDING) {
        return status;
    }

    GENAI_WARN("KV cache store plugin did not finish a batch within %u ms; cancelling it", m_io_timeout_ms);
    m_vtable->cancel(m_store, token);
    // A cancelled batch is owed a terminal status within one more timeout; the plugin may keep working on a batch
    // that is not terminal, so its buffers are never freed once the host stops waiting.
    status = m_vtable->wait(m_store, token, m_io_timeout_ms);
    if (status == OV_KVCACHE_PENDING) {
        GENAI_WARN("KV cache store plugin did not end a cancelled batch within %u ms; the store is given up and every "
                   "later request to it is a miss",
                   m_io_timeout_ms);
        give_up(batch);
    }
    return status;
}

void AbiKVCacheStore::give_up(const std::shared_ptr<void>& batch) const {
    // Never destroyed, because the plugin may touch the batch buffers at any later time.
    static std::mutex mutex;
    static auto* const kept_batches = new std::vector<std::shared_ptr<void>>();
    {
        const std::lock_guard<std::mutex> lock(mutex);
        kept_batches->push_back(batch);
    }
    m_abandoned = true;
}

OvKVCacheStatus AbiKVCacheStore::run_reads(const std::shared_ptr<ReadBatch>& batch) const {
    const size_t count = batch->requests.size();
    if (!m_use_async) {
        return m_vtable->read_blocks(m_store, batch->requests.data(), batch->results.data(), count);
    }
    OvKVCacheAsyncToken token = 0;
    const OvKVCacheStatus submitted =
        m_vtable->read_blocks_async(m_store, batch->requests.data(), batch->results.data(), count, &token);
    OPENVINO_ASSERT(submitted == OV_KVCACHE_OK && token != 0, "ABI store read_blocks_async failed with status ", submitted);
    // The buffers stay untouched until the token is terminal; a failed or cancelled batch has no usable items.
    const OvKVCacheStatus finished = await_token(token, batch);
    if (finished == OV_KVCACHE_PENDING || finished == OV_KVCACHE_ERR_CANCELLED) {
        return finished;
    }
    OPENVINO_ASSERT(finished == OV_KVCACHE_OK || finished == OV_KVCACHE_PARTIAL,
                    "ABI store async read ended with status ",
                    finished);
    return finished;
}

OvKVCacheStatus AbiKVCacheStore::run_writes(const std::shared_ptr<WriteBatch>& batch) const {
    const size_t count = batch->requests.size();
    if (!m_use_async) {
        return m_vtable->write_blocks(m_store, batch->requests.data(), batch->results.data(), count);
    }
    OvKVCacheAsyncToken token = 0;
    const OvKVCacheStatus submitted =
        m_vtable->write_blocks_async(m_store, batch->requests.data(), batch->results.data(), count, &token);
    OPENVINO_ASSERT(submitted == OV_KVCACHE_OK && token != 0, "ABI store write_blocks_async failed with status ", submitted);
    const OvKVCacheStatus finished = await_token(token, batch);
    if (finished == OV_KVCACHE_PENDING || finished == OV_KVCACHE_ERR_CANCELLED) {
        return finished;
    }
    OPENVINO_ASSERT(finished == OV_KVCACHE_OK || finished == OV_KVCACHE_PARTIAL,
                    "ABI store async write ended with status ",
                    finished);
    return finished;
}

void AbiKVCacheStore::drop_block(size_t hash) {
    if (m_abandoned) {
        return;
    }
    const OvKVCacheBlockKey key = make_key(m_group, hash);
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
    if (m_abandoned) {
        return 0;
    }
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
    if (m_abandoned || m_vtable->struct_bytes < flush_end || m_vtable->flush == nullptr) {
        return;
    }
    const OvKVCacheStatus status = m_vtable->flush(m_store);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store flush failed with status ", status);
}

}  // namespace ov::genai
