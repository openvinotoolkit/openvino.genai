// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/store_registry.hpp"

#include <cstring>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

#include "continuous_batching/cache/abi_kv_cache_store.hpp"
#include "continuous_batching/cache/abi_sdk_fingerprint.hpp"
#include "continuous_batching/cache/file_kv_cache_store.hpp"
#include "continuous_batching/cache/kv_cache_offload_manager.hpp"
#include "logger.hpp"
#include "openvino/genai/c/kv_cache_store_provider_abi.h"

namespace ov::genai {

namespace {

using QueryProvidersFn = const OvKVCacheStoreModuleInfo* (*)();

// The loaded module is intentionally never unloaded (see kv_cache_store_provider_abi.h:
// "The plugin library is never dlclose'd by the host."), so there is nothing to track beyond the
// handle itself staying valid for the process lifetime - no registry bookkeeping is needed.
void* load_library(const std::string& path) {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* resolve_symbol(void* handle, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

bool handshake_ok(const std::string& library_path, const OvKVCacheStoreModuleInfo& info) {
    if (info.abi_major != OV_KVCACHE_STORE_ABI_MAJOR) {
        GENAI_WARN("KV cache store plugin '%s' ABI major version %u does not match host %u",
                   library_path.c_str(),
                   info.abi_major,
                   static_cast<uint32_t>(OV_KVCACHE_STORE_ABI_MAJOR));
        return false;
    }
    if (info.abi_minor > OV_KVCACHE_STORE_ABI_MINOR) {
        GENAI_WARN("KV cache store plugin '%s' ABI minor version %u is newer than host %u",
                   library_path.c_str(),
                   info.abi_minor,
                   static_cast<uint32_t>(OV_KVCACHE_STORE_ABI_MINOR));
        return false;
    }
    if (info.sdk_build_fingerprint == nullptr || std::strcmp(info.sdk_build_fingerprint, get_host_sdk_build_fingerprint()) != 0) {
        GENAI_WARN("KV cache store plugin '%s' SDK build fingerprint does not match the host", library_path.c_str());
        return false;
    }
    return true;
}

const OvKVCacheStoreProviderDescriptor* find_provider(const OvKVCacheStoreModuleInfo& info, const std::string& name) {
    for (uint32_t i = 0; i < info.provider_count; ++i) {
        if (info.get_provider == nullptr) {
            break;
        }
        const OvKVCacheStoreProviderDescriptor* provider = info.get_provider(i);
        if (provider == nullptr) {
            continue;
        }
        if (name.empty() || (provider->name != nullptr && name == provider->name)) {
            return provider;
        }
    }
    return nullptr;
}

std::unique_ptr<IKVCacheStore> make_default_store(const KVCacheDiskLayout& layout,
                                                  const CacheOffloadConfig& config,
                                                  const std::string& device) {
    auto backend = std::make_unique<KVCacheOffloadManager>(layout, config, device);
    return std::make_unique<FileKVCacheStore>(std::move(backend));
}

/// Returns nullptr (after logging why) on any failure - never throws, never falls back itself.
std::unique_ptr<IKVCacheStore> try_open_plugin_store(const KVCacheDiskLayout& layout,
                                                     const CacheOffloadConfig& config,
                                                     const std::string& device,
                                                     const std::string& group) {
    void* handle = load_library(config.store_library_path);
    if (handle == nullptr) {
        GENAI_WARN("Failed to load KV cache store plugin '%s'", config.store_library_path.c_str());
        return nullptr;
    }

    void* symbol = resolve_symbol(handle, OV_KVCACHE_STORE_ENTRY_SYMBOL);
    if (symbol == nullptr) {
        GENAI_WARN("KV cache store plugin '%s' does not export '%s'",
                   config.store_library_path.c_str(),
                   OV_KVCACHE_STORE_ENTRY_SYMBOL);
        return nullptr;
    }

    const OvKVCacheStoreModuleInfo* module_info = reinterpret_cast<QueryProvidersFn>(symbol)();
    if (module_info == nullptr || !handshake_ok(config.store_library_path, *module_info)) {
        GENAI_WARN("KV cache store plugin '%s' failed the ABI handshake", config.store_library_path.c_str());
        return nullptr;
    }

    const OvKVCacheStoreProviderDescriptor* provider = find_provider(*module_info, config.store_provider_name);
    if (provider == nullptr || provider->vtable == nullptr || provider->vtable->open == nullptr) {
        GENAI_WARN("KV cache store plugin '%s' has no usable provider named '%s'",
                   config.store_library_path.c_str(),
                   config.store_provider_name.c_str());
        return nullptr;
    }

    const std::string block_size_str = std::to_string(layout.get_slot_size());
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}, {"device", device.c_str()}};

    OvKVCacheStoreOpenOptions options{};
    options.directory = config.path.empty() ? nullptr : config.path.c_str();
    options.capacity_bytes = config.capacity_bytes;
    options.properties = properties;
    options.property_count = 2;

    // The ABI opens one store instance per namespace; the group (kv / la / snapshot path) is the namespace.
    const OvKVCacheNamespace ns{reinterpret_cast<const uint8_t*>(group.data()), static_cast<uint64_t>(group.size())};

    OvKVCacheStoreOpenResult result{};
    const OvKVCacheStatus status = provider->vtable->open(provider->self, &ns, &options, &result);
    if (status != OV_KVCACHE_OK || result.status != OV_KVCACHE_OK || result.vtable == nullptr || result.store == nullptr) {
        GENAI_WARN("KV cache store plugin '%s' failed to open a store instance (status %d)",
                   config.store_library_path.c_str(),
                   static_cast<int>(result.status));
        return nullptr;
    }

    const size_t capacity_blocks = layout.get_slot_size() > 0 ? config.capacity_bytes / layout.get_slot_size() : 0;
    GENAI_INFO("[KV_TRACE] kv_cache_store plugin='%s' provider='%s' group='%s' capacity_blocks=%zu",
               config.store_library_path.c_str(),
               provider->name != nullptr ? provider->name : "",
               group.c_str(),
               capacity_blocks);
    return std::make_unique<AbiKVCacheStore>(result.vtable, result.store, group, layout.get_slot_size(), capacity_blocks);
}

}  // namespace

std::unique_ptr<IKVCacheStore> open_kv_cache_store(const KVCacheDiskLayout& layout,
                                                   const CacheOffloadConfig& config,
                                                   const std::string& device,
                                                   const std::string& group) {
    if (config.allow_untrusted_store_plugin && !config.store_library_path.empty()) {
        if (auto store = try_open_plugin_store(layout, config, device, group)) {
            return store;
        }
        GENAI_WARN("Falling back to the default file store for group '%s'", group.c_str());
    }
    return make_default_store(layout, config, device);
}

std::unique_ptr<IKVCacheStore> open_required_kv_cache_store_plugin(const KVCacheDiskLayout& layout,
                                                                    const CacheOffloadConfig& config,
                                                                    const std::string& device,
                                                                    const std::string& group) {
    OPENVINO_ASSERT(config.allow_untrusted_store_plugin && !config.store_library_path.empty(),
                    "A KV cache store plugin is required here (group '", group,
                    "') but none is configured: set both allow_untrusted_store_plugin and store_library_path");
    auto store = try_open_plugin_store(layout, config, device, group);
    OPENVINO_ASSERT(store != nullptr,
                    "Failed to open the required KV cache store plugin '", config.store_library_path,
                    "' for group '", group, "'");
    return store;
}

}  // namespace ov::genai
