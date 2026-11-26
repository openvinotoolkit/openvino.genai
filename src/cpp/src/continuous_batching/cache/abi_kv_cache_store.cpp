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

// The plugin is addressed by the 128-bit digest only; a block without one is never sent to it.
OvKVCacheContentHash to_abi_hash(const PrefixKey& key) {
    return OvKVCacheContentHash{key.digest->lo, key.digest->hi};
}

// Seed of the witness of a block: both halves of the key, so a block returned for another key fails the check.
uint64_t witness_seed(const PrefixKey& key) {
    return key.digest->lo ^ (key.digest->hi * 0x9E3779B97F4A7C15ull);
}

std::shared_ptr<AbiStoreInstance> make_instance(const OvKVCacheStoreVTable* vtable, void* store) {
    OPENVINO_ASSERT(vtable != nullptr && store != nullptr, "ABI KV cache store must not be null");
    return std::make_shared<AbiStoreInstance>(vtable, store);
}

}  // namespace

AbiStoreInstance::~AbiStoreInstance() {
    if (abandoned) {
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
    if (vtable->destroy != nullptr) {
        vtable->destroy(store);
    }
}

void AbiStoreInstance::flush() {
    // flush is a tail entry of the vtable: only call it when struct_bytes says the plugin provides it.
    constexpr size_t flush_end = offsetof(OvKVCacheStoreVTable, flush) + sizeof(OvKVCacheStoreVTable::flush);
    if (abandoned || vtable->struct_bytes < flush_end || vtable->flush == nullptr) {
        return;
    }
    const OvKVCacheStatus status = vtable->flush(store);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store flush failed with status ", status);
}

AbiKVCacheStore::AbiKVCacheStore(const OvKVCacheStoreVTable* vtable,
                                 void* store,
                                 std::string group,
                                 size_t block_size,
                                 size_t capacity_blocks,
                                 StoreInfo info,
                                 std::uint32_t io_timeout_ms)
    : AbiKVCacheStore(make_instance(vtable, store),
                      std::move(group),
                      block_size,
                      capacity_blocks,
                      info,
                      io_timeout_ms) {}

AbiKVCacheStore::AbiKVCacheStore(std::shared_ptr<AbiStoreInstance> instance,
                                 std::string group,
                                 size_t block_size,
                                 size_t capacity_blocks,
                                 StoreInfo info,
                                 std::uint32_t io_timeout_ms)
    : m_instance(std::move(instance)),
      m_vtable(m_instance->vtable),
      m_store(m_instance->store),
      m_group(std::move(group)),
      m_block_size(block_size),
      m_capacity_blocks(capacity_blocks),
      m_info(info),
      m_io_timeout_ms(io_timeout_ms),
      m_abandoned(m_instance->abandoned) {
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

    if ((m_info.capabilities & OV_KVCACHE_CAP_RESTORE_STATS) != 0) {
        constexpr size_t restore_end =
            offsetof(OvKVCacheStoreVTable, on_restore_finish) + sizeof(OvKVCacheStoreVTable::on_restore_finish);
        m_takes_restore_stats = m_vtable->struct_bytes >= restore_end && m_vtable->on_restore_finish != nullptr;
        if (!m_takes_restore_stats) {
            GENAI_WARN("KV cache store plugin advertises restore stats but its vtable lacks on_restore_finish; "
                       "the host reports the reuse metrics itself");
        }
    }
    if ((m_info.capabilities & OV_KVCACHE_CAP_PREFIX_TOKENS) != 0) {
        constexpr size_t announce_end = offsetof(OvKVCacheStoreVTable, announce_prefix_tokens) +
                                        sizeof(OvKVCacheStoreVTable::announce_prefix_tokens);
        m_takes_prefix_tokens = m_vtable->struct_bytes >= announce_end && m_vtable->announce_prefix_tokens != nullptr;
        if (!m_takes_prefix_tokens) {
            GENAI_WARN("KV cache store plugin advertises prefix tokens but its vtable lacks announce_prefix_tokens; "
                       "no token ids are sent");
        }
    }
}

bool AbiKVCacheStore::report_restore_stats(const RestoreStats& stats) {
    if (!m_takes_restore_stats || m_abandoned) {
        return false;
    }
    OvKVCacheRestoreStats abi_stats{};
    abi_stats.struct_bytes = sizeof(abi_stats);
    abi_stats.prompt_tokens = stats.prompt_tokens;
    abi_stats.device_hit_tokens = stats.device_hit_tokens;
    abi_stats.store_hit_tokens = stats.store_hit_tokens;
    abi_stats.reused_tokens = stats.reused_tokens;
    abi_stats.group = m_group.c_str();
    abi_stats.group_len = static_cast<uint32_t>(m_group.size());
    m_vtable->on_restore_finish(m_store, &abi_stats);
    return true;
}

void AbiKVCacheStore::announce_prefix_tokens(const std::vector<PrefixTokens>& prefixes) {
    if (!m_takes_prefix_tokens || m_abandoned || prefixes.empty()) {
        return;
    }
    std::vector<OvKVCachePrefixTokens> items;
    items.reserve(prefixes.size());
    for (const PrefixTokens& prefix : prefixes) {
        if (prefix.token_ids == nullptr || !prefix.key.digest) {
            continue;
        }
        items.push_back(OvKVCachePrefixTokens{to_abi_hash(prefix.key),
                                              prefix.token_ids->data(),
                                              static_cast<uint64_t>(prefix.token_ids->size())});
    }
    if (items.empty()) {
        return;
    }
    const OvKVCacheStatus status = m_vtable->announce_prefix_tokens(m_store, items.data(), items.size());
    if (status != OV_KVCACHE_OK) {
        GENAI_WARN("KV cache store plugin rejected the prefix tokens of group '%s' (status %d)",
                   m_group.c_str(),
                   static_cast<int>(status));
    }
}

OvKVCacheBlockKey AbiKVCacheStore::make_key(const std::string& group, const PrefixKey& prefix_key) {
    OvKVCacheBlockKey key{};
    key.group = group.c_str();
    key.group_len = static_cast<uint32_t>(group.size());
    key.hash = to_abi_hash(prefix_key);
    return key;
}

std::vector<bool> AbiKVCacheStore::has_blocks(const std::vector<PrefixKey>& chain) const {
    std::vector<bool> present(chain.size(), false);
    // The plugin numbers the blocks by their place in the chain, so the chain it is asked about ends where a digest is
    // missing; whatever follows cannot be told to be stored.
    size_t known = 0;
    while (known < chain.size() && chain[known].digest) {
        ++known;
    }
    if (known == 0 || m_abandoned) {
        return present;
    }
    std::vector<OvKVCacheBlockKey> keys;
    keys.reserve(known);
    for (size_t i = 0; i < known; ++i) {
        keys.push_back(make_key(m_group, chain[i]));
    }
    std::vector<uint8_t> stored(known, 0);
    const OvKVCacheStatus status = m_vtable->has_blocks(m_store, keys.data(), stored.data(), keys.size());
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store has_blocks failed with status ", status);
    for (size_t i = 0; i < known; ++i) {
        present[i] = stored[i] != 0;
    }
    return present;
}

std::vector<bool> AbiKVCacheStore::write_blocks(const std::vector<PrefixKey>& all_keys,
                                                const std::vector<const std::vector<uint8_t>*>& all_data) {
    OPENVINO_ASSERT(all_keys.size() == all_data.size(), "ABI store write_blocks needs one buffer per key");
    std::vector<bool> written(all_keys.size(), false);
    // A block without a digest cannot be addressed in the plugin.
    std::vector<size_t> usable;
    for (size_t i = 0; i < all_keys.size(); ++i) {
        if (all_keys[i].digest) {
            usable.push_back(i);
        }
    }
    const size_t count = usable.size();
    if (count == 0 || m_abandoned) {
        return written;
    }

    auto batch = std::make_shared<WriteBatch>(m_group, count);
    if (m_use_async) {
        // The plugin may still read the blocks after the host has given up waiting, so it gets copies the host never frees.
        batch->owned_data.reserve(count);
        for (const size_t index : usable) {
            batch->owned_data.push_back(*all_data[index]);
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const PrefixKey& key = all_keys[usable[i]];
        const std::vector<uint8_t>& block = *all_data[usable[i]];
        OPENVINO_ASSERT(block.size() == m_block_size, "ABI store block size mismatch");
        const std::vector<uint8_t>& source = m_use_async ? batch->owned_data[i] : block;
        // Seeding with the key binds the witness to the key it was written under, so a store that
        // returns a different (but internally valid) block for a read of this key is still caught below.
        batch->digests[i] = compute_content_hash(source.data(), source.size(), witness_seed(key));
        batch->requests[i].key = make_key(batch->group, key);
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
        written[usable[i]] = true;
    }
    return written;
}

std::vector<bool> AbiKVCacheStore::read_blocks(const std::vector<PrefixKey>& all_keys,
                                               std::vector<std::vector<uint8_t>>& data) const {
    std::vector<bool> found(all_keys.size(), false);
    data.assign(all_keys.size(), {});
    std::vector<size_t> usable;
    for (size_t i = 0; i < all_keys.size(); ++i) {
        if (all_keys[i].digest) {
            usable.push_back(i);
        }
    }
    const size_t count = usable.size();
    if (count == 0 || m_abandoned) {
        return found;
    }

    // Read into scratch buffers so a not-found or failed read leaves nothing half filled.
    auto batch = std::make_shared<ReadBatch>(m_group, count, m_block_size);
    for (size_t i = 0; i < count; ++i) {
        batch->requests[i].key = make_key(batch->group, all_keys[usable[i]]);
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
        const PrefixKey& key = all_keys[usable[i]];
        std::vector<uint8_t>& scratch = batch->scratch[i];
        if (batch->results[i].status == OV_KVCACHE_ERR_NOT_FOUND) {
            continue;
        }
        OPENVINO_ASSERT(batch_ok && batch->results[i].status == OV_KVCACHE_OK,
                        "ABI store read_blocks failed with status ",
                        batch->results[i].status);
        OPENVINO_ASSERT(batch->results[i].bytes_read == scratch.size(), "ABI store returned a short read");

        const ContentHash expected_witness = compute_content_hash(scratch.data(), scratch.size(), witness_seed(key));
        if (batch->results[i].witness.len != sizeof(expected_witness) ||
            std::memcmp(&expected_witness, &batch->witnesses[i], sizeof(expected_witness)) != 0) {
            // A corrupted or mismatched block is quarantined: dropped from the store and served as a miss, so the
            // request recomputes it instead of failing.
            GENAI_WARN("KV cache store plugin returned a block whose witness does not match; dropping it and "
                       "treating it as a miss");
            const OvKVCacheBlockKey block_key = make_key(m_group, key);
            const OvKVCacheStatus drop_status = m_vtable->drop_blocks(m_store, &block_key, 1);
            if (drop_status != OV_KVCACHE_OK) {
                GENAI_WARN("KV cache store plugin could not drop a block that failed its witness check (status %d)",
                           static_cast<int>(drop_status));
            }
            continue;
        }
        data[usable[i]] = std::move(scratch);
        found[usable[i]] = true;
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

void AbiKVCacheStore::invalidate(const PrefixKey& prefix_key) {
    if (m_abandoned || !prefix_key.digest) {
        return;
    }
    const OvKVCacheBlockKey key = make_key(m_group, prefix_key);
    const OvKVCacheStatus status = m_vtable->drop_blocks(m_store, &key, 1);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store drop_blocks failed with status ", status);
}

size_t AbiKVCacheStore::block_bytes() const {
    return m_block_size;
}

size_t AbiKVCacheStore::capacity_blocks_hint() const {
    return m_capacity_blocks;
}

size_t AbiKVCacheStore::get_num_free_blocks() const {
    if (m_abandoned) {
        return 0;
    }
    OvKVCacheStoreStats stats{};
    const OvKVCacheStatus status = m_vtable->stats(m_store, &stats);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK, "ABI store stats failed with status ", status);
    // The plugin sizes itself; the host's capacity estimate is only a hint and may be exceeded.
    return stats.blocks >= m_capacity_blocks ? 0 : m_capacity_blocks - stats.blocks;
}

void AbiKVCacheStore::flush() {
    m_instance->flush();
}

}  // namespace ov::genai
