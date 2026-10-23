// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/abi_file_store_provider.hpp"

#include <cstddef>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "continuous_batching/cache/abi_sdk_fingerprint.hpp"
#include "continuous_batching/cache/file_kv_cache_store.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/kv_cache_offload_manager.hpp"
#include "openvino/genai/cache_offload.hpp"

namespace ov::genai {

namespace {

/// One store instance per `open()` call. Holds the default FileKVCacheStore plus a side table for
/// witnesses, since witnesses are host-defined/host-supplied (see kv_cache_store_provider_abi.h)
/// and the payload-only FileKVCacheStore has no slot for them.
struct AbiFileStoreInstance {
    AbiFileStoreInstance(std::unique_ptr<FileKVCacheStore> store_, size_t block_size_)
        : store(std::move(store_)),
          block_size(block_size_) {}

    std::unique_ptr<FileKVCacheStore> store;
    size_t block_size;
    mutable std::mutex witness_mutex;
    std::unordered_map<size_t, std::vector<uint8_t>> witnesses;
};

// size_t keys at this layer are already a single folded 64-bit value, mirrored into both halves
// of the 128-bit wire key by AbiKVCacheStore::to_abi_hash(); read it back from either half (NOT
// their XOR, which would always cancel out to 0 since both halves are identical).
// OvKVCacheBlockKey.group is not used by this single-group default provider.
size_t fold_key(const OvKVCacheBlockKey& key) {
    return static_cast<size_t>(key.hash.lo);
}

void destroy_store(void* self) {
    delete static_cast<AbiFileStoreInstance*>(self);
}

OvKVCacheStatus store_has_blocks(void* self, const OvKVCacheBlockKey* keys, uint8_t* out_present, size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    for (size_t i = 0; i < count; ++i) {
        out_present[i] = instance->store->has_block(fold_key(keys[i])) ? 1 : 0;
    }
    return OV_KVCACHE_OK;
}

OvKVCacheStatus store_read_blocks(void* self,
                                  const OvKVCacheStoreReadRequest* reqs,
                                  OvKVCacheStoreReadResult* out,
                                  size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
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

        const size_t hash = fold_key(req.key);
        std::vector<uint8_t> data;
        if (!instance->store->read_block(hash, data)) {
            result.status = OV_KVCACHE_ERR_NOT_FOUND;
            any_failed = true;
            continue;
        }
        if (data.size() != req.target.bytes) {
            result.status = OV_KVCACHE_ERR_INTERNAL;
            any_failed = true;
            continue;
        }
        std::memcpy(req.target.base, data.data(), data.size());

        std::lock_guard<std::mutex> lock(instance->witness_mutex);
        auto it = instance->witnesses.find(hash);
        if (it != instance->witnesses.end() && req.witness_out.data != nullptr) {
            if (it->second.size() > req.witness_out.capacity) {
                result.status = OV_KVCACHE_ERR_BUFFER_TOO_SMALL;
                any_failed = true;
                continue;
            }
            std::memcpy(req.witness_out.data, it->second.data(), it->second.size());
            result.witness.data = req.witness_out.data;
            result.witness.len = it->second.size();
        }
        result.status = OV_KVCACHE_OK;
        result.bytes_read = data.size();
    }
    return any_failed ? OV_KVCACHE_PARTIAL : OV_KVCACHE_OK;
}

OvKVCacheStatus store_write_blocks(void* self,
                                   const OvKVCacheStoreWriteRequest* reqs,
                                   OvKVCacheStoreWriteResult* out,
                                   size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    bool any_failed = false;
    for (size_t i = 0; i < count; ++i) {
        const OvKVCacheStoreWriteRequest& req = reqs[i];
        OvKVCacheStoreWriteResult& result = out[i];
        result = OvKVCacheStoreWriteResult{};

        if (req.bytes != instance->block_size) {
            result.status = OV_KVCACHE_ERR_INVALID_ARG;
            any_failed = true;
            continue;
        }

        const size_t hash = fold_key(req.key);
        const auto* source_bytes = static_cast<const uint8_t*>(req.source);
        std::vector<uint8_t> data(source_bytes, source_bytes + req.bytes);
        if (!instance->store->write_block(hash, data)) {
            result.status = OV_KVCACHE_ERR_CAPACITY;
            any_failed = true;
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(instance->witness_mutex);
            if (req.witness.data != nullptr && req.witness.len > 0) {
                const auto* witness_bytes = static_cast<const uint8_t*>(req.witness.data);
                instance->witnesses[hash].assign(witness_bytes, witness_bytes + req.witness.len);
            } else {
                instance->witnesses.erase(hash);
            }
        }
        result.status = OV_KVCACHE_OK;
        result.bytes_written = req.bytes;
    }
    return any_failed ? OV_KVCACHE_PARTIAL : OV_KVCACHE_OK;
}

OvKVCacheStatus store_drop_blocks(void* self, const OvKVCacheBlockKey* keys, size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    std::lock_guard<std::mutex> lock(instance->witness_mutex);
    for (size_t i = 0; i < count; ++i) {
        const size_t hash = fold_key(keys[i]);
        instance->store->drop_block(hash);
        instance->witnesses.erase(hash);
    }
    return OV_KVCACHE_OK;
}

OvKVCacheStatus store_flush(void* /*self*/) {
    // KVCacheOffloadManager's pwrite/pread are already synchronous.
    return OV_KVCACHE_OK;
}

OvKVCacheStatus store_stats(void* self, OvKVCacheStoreStats* out) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    out->blocks = instance->store->get_capacity_blocks() - instance->store->get_num_free_blocks();
    out->bytes = out->blocks * instance->block_size;
    out->chunks = 0;
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
    nullptr,  // read_blocks_async: OV_KVCACHE_CAP_ASYNC_IO not advertised
    nullptr,  // write_blocks_async
    nullptr,  // wait
    nullptr,  // cancel
};

void destroy_provider(void* /*self*/) {}

const char* provider_name(void* /*self*/) {
    return "file";
}

OvKVCacheStatus provider_open(void* /*self*/,
                              const OvKVCacheNamespace* /*ns*/,
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
    std::string device = "CPU";
    for (uint64_t i = 0; i < opts->property_count; ++i) {
        const OvKVCacheProperty& property = opts->properties[i];
        if (property.key == nullptr || property.value == nullptr) {
            continue;
        }
        if (std::strcmp(property.key, "block_size_bytes") == 0) {
            block_size_bytes = static_cast<size_t>(std::stoull(property.value));
        } else if (std::strcmp(property.key, "device") == 0) {
            device = property.value;
        }
    }
    if (!block_size_bytes.has_value() || *block_size_bytes == 0) {
        out->status = OV_KVCACHE_ERR_INVALID_ARG;
        return out->status;
    }

    try {
        CacheOffloadConfig config;
        config.path = opts->directory != nullptr ? opts->directory : "";
        config.capacity_bytes = opts->capacity_bytes;
        const KVCacheDiskLayout layout({*block_size_bytes}, {0});
        auto backend = std::make_unique<KVCacheOffloadManager>(layout, config, device);
        auto store = std::make_unique<FileKVCacheStore>(std::move(backend));
        out->store = new AbiFileStoreInstance(std::move(store), *block_size_bytes);
        out->vtable = &g_store_vtable;
        out->status = OV_KVCACHE_OK;
    } catch (const std::exception&) {
        out->status = OV_KVCACHE_ERR_IO;
    }
    return out->status;
}

OvKVCacheStatus provider_purge(void* /*self*/, const OvKVCacheNamespace* /*ns*/) {
    // Each store instance is a single run-scoped file (see KVCacheOffloadManager); there is no
    // persisted namespace to purge independently of destroying the instance that opened it.
    return OV_KVCACHE_ERR_UNSUPPORTED;
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
    OV_KVCACHE_CAP_KEY_IO | OV_KVCACHE_CAP_EXISTS,
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
