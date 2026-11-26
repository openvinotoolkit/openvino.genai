// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/abi_file_store_provider.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include "continuous_batching/cache/abi_sdk_fingerprint.hpp"
#include "continuous_batching/cache/default_file_store.hpp"
#include "continuous_batching/cache/offload_settings.hpp"

namespace ov::genai {

namespace {

/// Identifies how this provider stores blocks (one run-private file, a block and its witness in one slot); bump the low
/// half whenever that changes so blocks written under other semantics are never mistaken for these.
constexpr OvKVCacheContentHash kFileStoreSemanticsFingerprint{/*lo=*/2, /*hi=*/0x66696c6553746f72ull};

/// Room the slot of a block keeps for its witness, after the payload: the length, then the bytes.
constexpr size_t kWitnessLengthBytes = sizeof(uint64_t);

/// One store instance per `open()` call: a run-private DefaultFileGroupStore that evicts by itself. A witness is
/// host-defined and host-supplied (see kv_cache_store_provider_abi.h), so each slot holds it next to the payload and
/// the two leave the store together.
struct AbiFileStoreInstance {
    AbiFileStoreInstance(std::unique_ptr<DefaultFileStore> files_,
                         std::unique_ptr<IKVCacheStore> store_,
                         size_t block_size_,
                         size_t witness_bytes_max_)
        : files(std::move(files_)),
          store(std::move(store_)),
          block_size(block_size_),
          witness_bytes_max(witness_bytes_max_) {}

    std::unique_ptr<DefaultFileStore> files;  // owns the directory the store lives in
    std::unique_ptr<IKVCacheStore> store;
    size_t block_size;
    size_t witness_bytes_max;
    /// Serializes whole store operations.
    std::mutex op_mutex;
    /// Guards `finished_batches` and `next_token`.
    std::mutex token_mutex;
    /// Async batches run to completion inside submit, so a token only carries the final batch status until it is waited on.
    std::unordered_map<OvKVCacheAsyncToken, OvKVCacheStatus> finished_batches;
    OvKVCacheAsyncToken next_token = 1;
};

// OvKVCacheBlockKey.group is not used by this single-group default provider.
PrefixKey store_key(const OvKVCacheBlockKey& key) {
    return PrefixKey{static_cast<size_t>(key.hash.lo), ContentHash{key.hash.lo, key.hash.hi}};
}

void destroy_store(void* self) {
    delete static_cast<AbiFileStoreInstance*>(self);
}

// C++ exceptions must not unwind through the C vtable, so every callback below converts them to a status.
OvKVCacheStatus store_has_blocks(void* self, const OvKVCacheBlockKey* keys, uint8_t* out_present, size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    try {
        std::lock_guard<std::mutex> operation_lock(instance->op_mutex);
        std::vector<PrefixKey> chain;
        chain.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            chain.push_back(store_key(keys[i]));
        }
        const std::vector<bool> present = instance->store->has_blocks(chain);
        for (size_t i = 0; i < count; ++i) {
            out_present[i] = present[i] ? 1 : 0;
        }
    } catch (...) {
        return OV_KVCACHE_ERR_IO;
    }
    return OV_KVCACHE_OK;
}

OvKVCacheStatus store_read_blocks(void* self,
                                  const OvKVCacheStoreReadRequest* reqs,
                                  OvKVCacheStoreReadResult* out,
                                  size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    std::lock_guard<std::mutex> operation_lock(instance->op_mutex);
    bool any_failed = false;
    for (size_t i = 0; i < count; ++i) {
        const OvKVCacheStoreReadRequest& req = reqs[i];
        OvKVCacheStoreReadResult& result = out[i];
        result = OvKVCacheStoreReadResult{};

        if (req.target.kind != OV_KVCACHE_TARGET_HOST || req.target.base == nullptr) {
            result.status = OV_KVCACHE_ERR_UNSUPPORTED;
            any_failed = true;
            continue;
        }

        std::vector<std::vector<uint8_t>> slot;
        bool found = false;
        try {
            found = instance->store->read_blocks({store_key(req.key)}, slot)[0];
        } catch (...) {
            result.status = OV_KVCACHE_ERR_IO;
            any_failed = true;
            continue;
        }
        if (!found) {
            result.status = OV_KVCACHE_ERR_NOT_FOUND;
            any_failed = true;
            continue;
        }
        const std::vector<uint8_t>& payload = slot[0];
        if (req.target.bytes != instance->block_size) {
            result.status = OV_KVCACHE_ERR_INTERNAL;
            any_failed = true;
            continue;
        }
        std::memcpy(req.target.base, payload.data(), instance->block_size);

        uint64_t witness_len = 0;
        std::memcpy(&witness_len, payload.data() + instance->block_size + instance->witness_bytes_max, sizeof(witness_len));
        if (witness_len > 0 && req.witness_out.data != nullptr) {
            if (witness_len > req.witness_out.capacity) {
                result.status = OV_KVCACHE_ERR_BUFFER_TOO_SMALL;
                any_failed = true;
                continue;
            }
            std::memcpy(req.witness_out.data, payload.data() + instance->block_size, static_cast<size_t>(witness_len));
            result.witness.data = req.witness_out.data;
            result.witness.len = witness_len;
        }
        result.status = OV_KVCACHE_OK;
        result.bytes_read = instance->block_size;
    }
    return any_failed ? OV_KVCACHE_PARTIAL : OV_KVCACHE_OK;
}

OvKVCacheStatus store_write_blocks(void* self,
                                   const OvKVCacheStoreWriteRequest* reqs,
                                   OvKVCacheStoreWriteResult* out,
                                   size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    std::lock_guard<std::mutex> operation_lock(instance->op_mutex);
    bool any_failed = false;
    for (size_t i = 0; i < count; ++i) {
        const OvKVCacheStoreWriteRequest& req = reqs[i];
        OvKVCacheStoreWriteResult& result = out[i];
        result = OvKVCacheStoreWriteResult{};

        if (req.bytes != instance->block_size || req.witness.len > instance->witness_bytes_max) {
            result.status = OV_KVCACHE_ERR_INVALID_ARG;
            any_failed = true;
            continue;
        }

        try {
            std::vector<uint8_t> slot(instance->block_size + instance->witness_bytes_max + kWitnessLengthBytes, 0);
            std::memcpy(slot.data(), req.source, req.bytes);
            if (req.witness.data != nullptr && req.witness.len > 0) {
                std::memcpy(slot.data() + instance->block_size, req.witness.data, static_cast<size_t>(req.witness.len));
            }
            const uint64_t witness_len = req.witness.data != nullptr ? req.witness.len : 0;
            std::memcpy(slot.data() + instance->block_size + instance->witness_bytes_max, &witness_len, sizeof(witness_len));
            if (!instance->store->write_blocks({store_key(req.key)}, {&slot})[0]) {
                result.status = OV_KVCACHE_ERR_IO;
                any_failed = true;
                continue;
            }
        } catch (...) {
            result.status = OV_KVCACHE_ERR_IO;
            any_failed = true;
            continue;
        }
        result.status = OV_KVCACHE_OK;
        result.bytes_written = req.bytes;
    }
    return any_failed ? OV_KVCACHE_PARTIAL : OV_KVCACHE_OK;
}

OvKVCacheStatus store_drop_blocks(void* self, const OvKVCacheBlockKey* keys, size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    try {
        std::lock_guard<std::mutex> operation_lock(instance->op_mutex);
        for (size_t i = 0; i < count; ++i) {
            instance->store->invalidate(store_key(keys[i]));
        }
    } catch (...) {
        return OV_KVCACHE_ERR_IO;
    }
    return OV_KVCACHE_OK;
}

OvKVCacheStatus store_flush(void* self) {
    try {
        static_cast<AbiFileStoreInstance*>(self)->store->flush();
    } catch (...) {
        return OV_KVCACHE_ERR_IO;
    }
    return OV_KVCACHE_OK;
}

OvKVCacheStatus store_stats(void* self, OvKVCacheStoreStats* out) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    auto* group = static_cast<DefaultFileGroupStore*>(instance->store.get());
    out->blocks = group->capacity_blocks_hint() - group->get_num_free_blocks();
    out->bytes = out->blocks * instance->block_size;
    out->chunks = 0;
    return OV_KVCACHE_OK;
}

OvKVCacheStatus issue_token(AbiFileStoreInstance& instance, OvKVCacheStatus status, OvKVCacheAsyncToken* out_token) {
    std::lock_guard<std::mutex> lock(instance.token_mutex);
    const OvKVCacheAsyncToken token = instance.next_token++;
    instance.finished_batches.emplace(token, status);
    *out_token = token;
    return OV_KVCACHE_OK;
}

OvKVCacheStatus store_read_blocks_async(void* self,
                                        const OvKVCacheStoreReadRequest* reqs,
                                        OvKVCacheStoreReadResult* out,
                                        size_t count,
                                        OvKVCacheAsyncToken* out_token) {
    *out_token = 0;
    return issue_token(*static_cast<AbiFileStoreInstance*>(self), store_read_blocks(self, reqs, out, count), out_token);
}

OvKVCacheStatus store_write_blocks_async(void* self,
                                         const OvKVCacheStoreWriteRequest* reqs,
                                         OvKVCacheStoreWriteResult* out,
                                         size_t count,
                                         OvKVCacheAsyncToken* out_token) {
    *out_token = 0;
    return issue_token(*static_cast<AbiFileStoreInstance*>(self), store_write_blocks(self, reqs, out, count), out_token);
}

OvKVCacheStatus store_wait(void* self, OvKVCacheAsyncToken token, uint32_t /*timeout_ms*/) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    std::lock_guard<std::mutex> lock(instance->token_mutex);
    const auto it = instance->finished_batches.find(token);
    if (it == instance->finished_batches.end()) {
        return OV_KVCACHE_ERR_INVALID_ARG;
    }
    const OvKVCacheStatus status = it->second;
    instance->finished_batches.erase(it);
    return status;
}

OvKVCacheStatus store_cancel(void* /*self*/, OvKVCacheAsyncToken /*token*/) {
    // Nothing is ever in flight: the batch has already finished when its token is returned.
    return OV_KVCACHE_OK;
}

const OvKVCacheStoreVTable g_store_vtable = {
    sizeof(OvKVCacheStoreVTable),
    0,
    &destroy_store,
    &store_has_blocks,
    &store_read_blocks,
    &store_write_blocks,
    &store_drop_blocks,
    &store_flush,
    &store_stats,
    &store_read_blocks_async,
    &store_write_blocks_async,
    &store_wait,
    &store_cancel,
    nullptr,
    nullptr,
};

void destroy_provider(void* /*self*/) {}

const char* provider_name(void* /*self*/) {
    return "file";
}

OvKVCacheStatus provider_open(void* /*self*/,
                              const OvKVCacheNamespace* ns,
                              const OvKVCacheStoreOpenOptions* opts,
                              OvKVCacheStoreOpenResult* out) {
    if (out == nullptr) {
        return OV_KVCACHE_ERR_INVALID_ARG;
    }
    *out = OvKVCacheStoreOpenResult{};
    if (opts == nullptr) {
        out->status = OV_KVCACHE_ERR_INVALID_ARG;
        return out->status;
    }

    std::optional<size_t> block_size_bytes;
    std::string device = opts->device != nullptr ? opts->device : "CPU";
    if (opts->groups != nullptr && opts->group_count == 1) {
        block_size_bytes = static_cast<size_t>(opts->groups[0].block_bytes);
    }
    // A host built before the typed fields passes these as properties.
    for (uint64_t i = 0; i < opts->property_count; ++i) {
        const OvKVCacheProperty& property = opts->properties[i];
        if (property.key == nullptr || property.value == nullptr) {
            continue;
        }
        if (std::strcmp(property.key, "block_size_bytes") == 0) {
            try {
                block_size_bytes = static_cast<size_t>(std::stoull(property.value));
            } catch (const std::exception&) {
                out->status = OV_KVCACHE_ERR_INVALID_ARG;
                return out->status;
            }
        } else if (std::strcmp(property.key, "device") == 0) {
            device = property.value;
        }
    }
    if (!block_size_bytes.has_value() || *block_size_bytes == 0) {
        out->status = OV_KVCACHE_ERR_INVALID_ARG;
        return out->status;
    }

    // The group descriptor for this namespace must agree with the block size the store is built for.
    if (ns != nullptr && ns->bytes != nullptr && opts->groups != nullptr) {
        for (uint64_t i = 0; i < opts->group_count; ++i) {
            const OvKVCacheGroupInfo& info = opts->groups[i];
            if (info.group != nullptr && info.group_len == ns->len &&
                std::memcmp(info.group, ns->bytes, ns->len) == 0 && info.block_bytes != *block_size_bytes) {
                if (opts->log != nullptr) {
                    opts->log(OV_KVCACHE_LOG_ERROR, "file store: group block_bytes disagrees with block_size_bytes");
                }
                out->status = OV_KVCACHE_ERR_INVALID_ARG;
                return out->status;
            }
        }
    }
    if (opts->log != nullptr) {
        opts->log(OV_KVCACHE_LOG_DEBUG, "file store: opening store instance");
    }

    try {
        const size_t witness_bytes_max = opts->witness_bytes_max > 0 ? static_cast<size_t>(opts->witness_bytes_max) : 16;
        const size_t slot_bytes = *block_size_bytes + witness_bytes_max + kWitnessLengthBytes;
        // The budget counts the blocks themselves, as it does for every store.
        const size_t capacity_blocks = static_cast<size_t>(opts->capacity_bytes / *block_size_bytes);
        if (capacity_blocks == 0) {
            out->status = OV_KVCACHE_ERR_INVALID_ARG;
            return out->status;
        }
        // A namespace without ':' gives a directory that is deleted when the instance is destroyed.
        const std::filesystem::path root =
            opts->directory != nullptr && opts->directory[0] != '\0' ? std::filesystem::path(opts->directory)
                                                                       : std::filesystem::temp_directory_path();
        auto files = std::make_unique<DefaultFileStore>(
            root, "file", std::vector<DefaultFileStore::Group>{{"file", slot_bytes, capacity_blocks}}, false);
        auto store = files->group(0);
        out->store = new AbiFileStoreInstance(std::move(files), std::move(store), *block_size_bytes, witness_bytes_max);
        out->vtable = &g_store_vtable;
        out->store_semantics_fingerprint = kFileStoreSemanticsFingerprint;
        out->has_semantics_fingerprint = 1;
        out->status = OV_KVCACHE_OK;
    } catch (const std::exception&) {
        out->status = OV_KVCACHE_ERR_IO;
    }
    return out->status;
}

OvKVCacheStatus provider_purge(void* /*self*/, const OvKVCacheNamespace* /*ns*/) {
    // Each store instance is a single run-private file, so no block of an earlier run exists.
    return OV_KVCACHE_OK;
}

OvKVCacheStatus provider_list(void* /*self*/,
                              OvKVCacheStoredNamespaceInfo* /*out*/,
                              size_t /*capacity*/,
                              size_t* out_count) {
    if (out_count != nullptr) {
        *out_count = 0;
    }
    return OV_KVCACHE_OK;
}

const OvKVCacheStoreProviderVTable g_provider_vtable = {
    sizeof(OvKVCacheStoreProviderVTable),
    0,
    &destroy_provider,
    &provider_name,
    &provider_open,
    &provider_purge,
    &provider_list,
    nullptr,  // register_host_memory: OV_KVCACHE_CAP_REGISTER_HOST_MEMORY not advertised
};

const OvKVCacheStoreProviderDescriptor g_provider_descriptor = {
    &g_provider_vtable,
    nullptr,
    OV_KVCACHE_CAP_KEY_IO | OV_KVCACHE_CAP_EXISTS | OV_KVCACHE_CAP_SEMANTICS_FINGERPRINT | OV_KVCACHE_CAP_ASYNC_IO,
    "file",
};

const OvKVCacheStoreProviderDescriptor* get_provider(uint32_t index) {
    return index == 0 ? &g_provider_descriptor : nullptr;
}

const OvKVCacheStoreModuleInfo g_module_info = {
    OV_KVCACHE_STORE_ABI_MAJOR,
    OV_KVCACHE_STORE_ABI_MINOR,
    get_host_sdk_build_fingerprint(),
    "ov-genai-file-store",
    "1.0",
    1,
    0,
    &get_provider,
};

}  // namespace

const OvKVCacheStoreModuleInfo* get_file_store_provider_module_info() {
    return &g_module_info;
}

}  // namespace ov::genai

extern "C" OV_KVCACHE_STORE_PLUGIN_EXPORT const OvKVCacheStoreModuleInfo* ov_pipeline_query_kvcache_store_providers(void) {
    return ov::genai::get_file_store_provider_module_info();
}
