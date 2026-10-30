// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/abi_file_store_provider.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

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
    /// Guards `key_ids`, `next_id` and `witnesses`.
    mutable std::mutex witness_mutex;
    /// Full 128-bit ABI key -> the id FileKVCacheStore addresses the block by. Keys that differ only
    /// in `hi` must stay distinct, so the id is allocated here instead of being derived from `lo`.
    std::map<std::pair<uint64_t, uint64_t>, size_t> key_ids;
    size_t next_id = 0;
    std::unordered_map<size_t, std::vector<uint8_t>> witnesses;
};

// OvKVCacheBlockKey.group is not used by this single-group default provider.
std::pair<uint64_t, uint64_t> full_key(const OvKVCacheBlockKey& key) {
    return {key.hash.lo, key.hash.hi};
}

std::optional<size_t> find_id(AbiFileStoreInstance& instance, const OvKVCacheBlockKey& key) {
    std::lock_guard<std::mutex> lock(instance.witness_mutex);
    const auto it = instance.key_ids.find(full_key(key));
    return it == instance.key_ids.end() ? std::nullopt : std::optional<size_t>(it->second);
}

/// @return The id for `key` and whether it was newly allocated by this call.
std::pair<size_t, bool> get_or_create_id(AbiFileStoreInstance& instance, const OvKVCacheBlockKey& key) {
    std::lock_guard<std::mutex> lock(instance.witness_mutex);
    const auto [it, inserted] = instance.key_ids.emplace(full_key(key), instance.next_id);
    if (inserted) {
        ++instance.next_id;
    }
    return {it->second, inserted};
}

void destroy_store(void* self) {
    delete static_cast<AbiFileStoreInstance*>(self);
}

// C++ exceptions must not unwind through the C vtable, so every callback below converts them to a status.
OvKVCacheStatus store_has_blocks(void* self, const OvKVCacheBlockKey* keys, uint8_t* out_present, size_t count) {
    auto* instance = static_cast<AbiFileStoreInstance*>(self);
    try {
        for (size_t i = 0; i < count; ++i) {
            const auto id = find_id(*instance, keys[i]);
            out_present[i] = id.has_value() && instance->store->has_block(*id) ? 1 : 0;
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

        const auto id = find_id(*instance, req.key);
        std::vector<uint8_t> data;
        bool found = false;
        try {
            found = id.has_value() && instance->store->read_block(*id, data);
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
        if (data.size() != req.target.bytes) {
            result.status = OV_KVCACHE_ERR_INTERNAL;
            any_failed = true;
            continue;
        }
        std::memcpy(req.target.base, data.data(), data.size());

        std::lock_guard<std::mutex> lock(instance->witness_mutex);
        auto it = instance->witnesses.find(*id);
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

        const auto source_bytes = static_cast<const uint8_t*>(req.source);
        const auto [id, created] = get_or_create_id(*instance, req.key);
        try {
            std::vector<uint8_t> data(source_bytes, source_bytes + req.bytes);
            if (!instance->store->write_block(id, data)) {
                if (created) {
                    std::lock_guard<std::mutex> lock(instance->witness_mutex);
                    instance->key_ids.erase(full_key(req.key));
                }
                result.status = OV_KVCACHE_ERR_CAPACITY;
                any_failed = true;
                continue;
            }

            std::lock_guard<std::mutex> lock(instance->witness_mutex);
            if (req.witness.data != nullptr && req.witness.len > 0) {
                const auto* witness_bytes = static_cast<const uint8_t*>(req.witness.data);
                instance->witnesses[id].assign(witness_bytes, witness_bytes + req.witness.len);
            } else {
                instance->witnesses.erase(id);
            }
        } catch (...) {
            if (created) {
                std::lock_guard<std::mutex> lock(instance->witness_mutex);
                instance->key_ids.erase(full_key(req.key));
            }
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
        std::lock_guard<std::mutex> lock(instance->witness_mutex);
        for (size_t i = 0; i < count; ++i) {
            const auto it = instance->key_ids.find(full_key(keys[i]));
            if (it == instance->key_ids.end()) {
                continue;
            }
            instance->store->drop_block(it->second);
            instance->witnesses.erase(it->second);
            instance->key_ids.erase(it);
        }
    } catch (...) {
        return OV_KVCACHE_ERR_IO;
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
    // A session snapshot must outlive the store that wrote it, but this provider's data is a
    // run-scoped file that disappears with the instance, so refuse instead of reporting a save
    // that can never be restored.
    constexpr char snapshot_prefix[] = "session_snapshot_";
    constexpr size_t snapshot_prefix_len = sizeof(snapshot_prefix) - 1;
    if (ns != nullptr && ns->bytes != nullptr && ns->len >= snapshot_prefix_len &&
        std::memcmp(ns->bytes, snapshot_prefix, snapshot_prefix_len) == 0) {
        out->status = OV_KVCACHE_ERR_UNSUPPORTED;
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
