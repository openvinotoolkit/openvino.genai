// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/store_registry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

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
#include "continuous_batching/cache/default_file_store.hpp"
#include "content_hash.hpp"
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

void plugin_log(OvKVCacheLogLevel level, const char* message) {
    switch (level) {
    case OV_KVCACHE_LOG_ERROR:
        GENAI_ERR("[kv_cache_store plugin] %s", message != nullptr ? message : "");
        break;
    case OV_KVCACHE_LOG_WARNING:
        GENAI_WARN("[kv_cache_store plugin] %s", message != nullptr ? message : "");
        break;
    case OV_KVCACHE_LOG_INFO:
        GENAI_INFO("[kv_cache_store plugin] %s", message != nullptr ? message : "");
        break;
    default:
        GENAI_DEBUG("[kv_cache_store plugin] %s", message != nullptr ? message : "");
        break;
    }
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

std::string file_stem_of(const std::string& group) {
    std::string stem = group.substr(0, group.find(':'));
    std::replace_if(stem.begin(), stem.end(), [](char c) { return !std::isalnum(static_cast<unsigned char>(c)) && c != '_'; }, '_');
    return stem.empty() ? "group" : stem;
}

/// The in-tree store of @p groups in the directory of @p name_space; see DefaultFileGroupStore.
std::unique_ptr<IGroupedKVCacheStore> open_default_store(const std::vector<StoreGroupSpec>& groups,
                                                         const OffloadSettings& config,
                                                         const std::string& device,
                                                         const std::string& name_space) {
    // Device caches are staged through host tensors, so anything the KV cache manager can copy a block out of works;
    // NPU keeps its cache outside of that manager and is therefore excluded.
    OPENVINO_ASSERT(device.find("CPU") != std::string::npos || device.find("GPU") != std::string::npos,
                    "KV cache offload is implemented for CPU and GPU, but the inference device is '", device, "'");
    std::vector<DefaultFileStore::Group> files;
    for (const StoreGroupSpec& group : groups) {
        const size_t block_bytes = group.layout.get_slot_size();
        OPENVINO_ASSERT(block_bytes > 0, "KV cache offload slot size must be greater than 0");
        files.push_back({file_stem_of(group.name), block_bytes, static_cast<size_t>(group.capacity_bytes / block_bytes)});
    }
    return std::make_unique<DefaultFileStore>(config.kv_cache_dir, name_space, files, config.reuse_ssd_cache);
}

/// The provider of the configured plugin. Throws, naming the library, the provider and the stage, when the plugin cannot
/// be used; the host never falls back to the in-tree store.
const OvKVCacheStoreProviderDescriptor* load_plugin_provider(const OffloadSettings& config) {
    const std::string& library = config.store_library_path;
    void* handle = load_library(library);
    OPENVINO_ASSERT(handle != nullptr, "Failed to load KV cache store plugin '", library, "'");

    void* symbol = resolve_symbol(handle, OV_KVCACHE_STORE_ENTRY_SYMBOL);
    OPENVINO_ASSERT(symbol != nullptr,
                    "KV cache store plugin '", library, "' does not export '", OV_KVCACHE_STORE_ENTRY_SYMBOL, "'");

    const OvKVCacheStoreModuleInfo* module_info = reinterpret_cast<QueryProvidersFn>(symbol)();
    OPENVINO_ASSERT(module_info != nullptr && handshake_ok(library, *module_info),
                    "KV cache store plugin '", library, "' failed the ABI handshake");
    if (!config.debug_log_path.empty() && module_info->abi_minor < 3) {
        GENAI_WARN("KV cache store plugin '%s' was built against ABI 1.%u, which has no debug_log_path, so it is ignored",
                   library.c_str(),
                   module_info->abi_minor);
    }

    const OvKVCacheStoreProviderDescriptor* provider = find_provider(*module_info, config.store_provider_name);
    // struct_bytes bounds what the host may read; open() is a tail entry, so check it covers the pointer.
    constexpr size_t provider_open_end =
        offsetof(OvKVCacheStoreProviderVTable, open) + sizeof(OvKVCacheStoreProviderVTable::open);
    OPENVINO_ASSERT(provider != nullptr && provider->vtable != nullptr &&
                        provider->vtable->struct_bytes >= provider_open_end && provider->vtable->open != nullptr,
                    "KV cache store plugin '", library, "' has no usable provider named '", config.store_provider_name, "'");
    // The host calls has_blocks before every read and write, so a provider without it cannot be used.
    constexpr OvKVCacheCapabilities required_capabilities = OV_KVCACHE_CAP_KEY_IO | OV_KVCACHE_CAP_EXISTS;
    OPENVINO_ASSERT((provider->capabilities & required_capabilities) == required_capabilities,
                    "KV cache store plugin '", library, "' provider '", provider->name != nullptr ? provider->name : "",
                    "' does not advertise key I/O and has_blocks");

    return provider;
}

/// The cache group of a store instance, as the plugin is told about it.
struct PluginGroup {
    std::string name;
    size_t block_bytes = 0;
    size_t block_tokens = 0;
};

struct PluginInstance {
    std::shared_ptr<AbiStoreInstance> instance;
    AbiKVCacheStore::StoreInfo info;
};

/// Opens one store instance of @p provider for @p groups. Throws when it cannot be used.
PluginInstance open_plugin_instance(const OvKVCacheStoreProviderDescriptor* provider,
                                                   const OffloadSettings& config,
                                                   const std::string& device,
                                                   const std::string& ns_string,
                                                   const std::vector<PluginGroup>& groups,
                                                   uint64_t capacity_bytes) {
    // The settings of the host reach the plugin in typed fields; the properties are only the ones the user configured.
    std::vector<OvKVCacheProperty> properties;
    for (const auto& [key, value] : config.store_properties) {
        properties.push_back({key.c_str(), value.c_str()});
    }

    OvKVCacheStoreOpenOptions options{};
    options.directory = config.kv_cache_dir.empty() ? nullptr : config.kv_cache_dir.c_str();
    options.capacity_bytes = capacity_bytes;
    options.properties = properties.data();
    options.property_count = properties.size();
    options.device = device.c_str();
    options.debug_log_path = config.debug_log_path.empty() ? nullptr : config.debug_log_path.c_str();
    if ((provider->capabilities & OV_KVCACHE_CAP_DRAM_TIER) != 0) {
        options.dram_capacity_bytes = static_cast<uint64_t>(std::llround(config.dram_cache_capacity_gb * (1ull << 30)));
    } else if (config.dram_cache_capacity_gb > 0.0) {
        GENAI_WARN("KV cache store plugin '%s' has no DRAM tier, so dram_cache_capacity_gb=%g is ignored",
                   config.store_library_path.c_str(),
                   config.dram_cache_capacity_gb);
    }
    std::vector<OvKVCacheGroupInfo> group_infos;
    group_infos.reserve(groups.size());
    for (const PluginGroup& group : groups) {
        group_infos.push_back(OvKVCacheGroupInfo{group.name.c_str(),
                                                 static_cast<uint32_t>(group.name.size()),
                                                 static_cast<uint32_t>(group.block_tokens),
                                                 static_cast<uint64_t>(group.block_bytes)});
    }
    options.groups = group_infos.data();
    options.group_count = group_infos.size();
    options.witness_bytes_max = sizeof(ContentHash);
    options.log = &plugin_log;
    const OvKVCacheNamespace ns{reinterpret_cast<const uint8_t*>(ns_string.data()), static_cast<uint64_t>(ns_string.size())};

    OvKVCacheStoreOpenResult result{};
    const OvKVCacheStatus status = provider->vtable->open(provider->self, &ns, &options, &result);
    OPENVINO_ASSERT(status == OV_KVCACHE_OK && result.status == OV_KVCACHE_OK && result.vtable != nullptr &&
                        result.store != nullptr,
                    "KV cache store plugin '", config.store_library_path, "' provider '",
                    provider->name != nullptr ? provider->name : "", "' failed to open a store instance (status ",
                    static_cast<int>(status != OV_KVCACHE_OK ? status : result.status), ")");

    constexpr size_t store_sync_end = offsetof(OvKVCacheStoreVTable, read_blocks_async);
    // A vtable long enough to hold the entries can still leave them null; reject it here instead of crashing on the
    // first call.
    if (result.vtable->struct_bytes < store_sync_end || result.vtable->destroy == nullptr ||
        result.vtable->has_blocks == nullptr || result.vtable->read_blocks == nullptr ||
        result.vtable->write_blocks == nullptr || result.vtable->drop_blocks == nullptr ||
        result.vtable->flush == nullptr || result.vtable->stats == nullptr) {
        // destroy itself may sit past struct_bytes in a truncated vtable; only call it when that is not the case.
        constexpr size_t destroy_end = offsetof(OvKVCacheStoreVTable, destroy) + sizeof(OvKVCacheStoreVTable::destroy);
        if (result.vtable->struct_bytes >= destroy_end && result.vtable->destroy != nullptr) {
            result.vtable->destroy(result.store);
        }
        OPENVINO_THROW("KV cache store plugin '", config.store_library_path,
                       "' returned a store vtable missing required synchronous entries");
    }

    PluginInstance opened;
    opened.instance = std::make_shared<AbiStoreInstance>(result.vtable, result.store);
    if (!config.reuse_ssd_cache) {
        constexpr size_t provider_purge_end =
            offsetof(OvKVCacheStoreProviderVTable, purge) + sizeof(OvKVCacheStoreProviderVTable::purge);
        const OvKVCacheStatus purged = provider->vtable->struct_bytes >= provider_purge_end && provider->vtable->purge != nullptr
                                           ? provider->vtable->purge(provider->self, &ns)
                                           : OV_KVCACHE_ERR_UNSUPPORTED;
        OPENVINO_ASSERT(purged == OV_KVCACHE_OK,
                        "KV cache store plugin '", config.store_library_path,
                        "' could not purge the cache of an earlier run (status ", static_cast<int>(purged),
                        "), which reuse_ssd_cache=false asks for");
    }
    opened.info.capabilities = provider->capabilities;
    const bool fingerprint_advertised = (provider->capabilities & OV_KVCACHE_CAP_SEMANTICS_FINGERPRINT) != 0;
    if (fingerprint_advertised && result.has_semantics_fingerprint != 0) {
        opened.info.semantics_fingerprint = result.store_semantics_fingerprint;
    } else if (fingerprint_advertised != (result.has_semantics_fingerprint != 0)) {
        GENAI_WARN("KV cache store plugin '%s' advertises a semantics fingerprint inconsistently with what open() reported; ignoring it",
                   config.store_library_path.c_str());
    }
    GENAI_INFO("[KV_TRACE] kv_cache_store plugin='%s' provider='%s' groups=%zu capacity_bytes=%llu async_io=%d semantics_fingerprint=%016llx%016llx",
               config.store_library_path.c_str(),
               provider->name != nullptr ? provider->name : "",
               groups.size(),
               static_cast<unsigned long long>(capacity_bytes),
               (provider->capabilities & OV_KVCACHE_CAP_ASYNC_IO) != 0 ? 1 : 0,
               static_cast<unsigned long long>(opened.info.semantics_fingerprint ? opened.info.semantics_fingerprint->hi : 0),
               static_cast<unsigned long long>(opened.info.semantics_fingerprint ? opened.info.semantics_fingerprint->lo : 0));
    return opened;
}

std::unique_ptr<AbiKVCacheStore> make_plugin_view(const PluginInstance& opened,
                                                  const std::string& group,
                                                  size_t block_bytes,
                                                  uint64_t capacity_bytes,
                                                  std::uint32_t timeout_ms) {
    const size_t capacity_blocks = block_bytes > 0 ? static_cast<size_t>(capacity_bytes / block_bytes) : 0;
    GENAI_INFO("[KV_TRACE] kv_cache_store group='%s' capacity_blocks=%zu", group.c_str(), capacity_blocks);
    return std::make_unique<AbiKVCacheStore>(opened.instance, group, block_bytes, capacity_blocks, opened.info, timeout_ms);
}

/// One store instance for the group, opened under its own namespace. Throws when the plugin cannot be used.
std::unique_ptr<IKVCacheStore> open_plugin_store(const OvKVCacheStoreProviderDescriptor* provider,
                                                 const KVCacheDiskLayout& layout,
                                                 const OffloadSettings& config,
                                                 const std::string& device,
                                                 const std::string& group,
                                                 size_t block_tokens,
                                                 uint64_t capacity_bytes) {
    const size_t block_bytes = layout.get_slot_size();
    // The ABI opens one store instance per namespace; the group (kv / la, plus the model digest) is the namespace.
    const auto opened = open_plugin_instance(provider, config, device, group, {{group, block_bytes, block_tokens}}, capacity_bytes);
    return make_plugin_view(opened, group, block_bytes, capacity_bytes, config.store_timeout_ms);
}

OffloadSettings config_with_capacity(const OffloadSettings& config, uint64_t capacity_bytes) {
    OffloadSettings group_config = config;
    group_config.capacity_bytes = capacity_bytes;
    return group_config;
}

}  // namespace

std::unique_ptr<IKVCacheStore> open_kv_cache_store(const KVCacheDiskLayout& layout,
                                                   const OffloadSettings& config,
                                                   const std::string& device,
                                                   const std::string& group,
                                                   size_t block_tokens) {
    if (!config.store_library_path.empty()) {
        return open_plugin_store(load_plugin_provider(config), layout, config, device, group, block_tokens, config.capacity_bytes);
    }
    return open_default_store({StoreGroupSpec{group, group, layout, block_tokens, config.capacity_bytes}}, config, device, group)->group(0);
}

std::unique_ptr<IGroupedKVCacheStore> open_grouped_kv_cache_store(const std::vector<StoreGroupSpec>& groups,
                                                                  const OffloadSettings& config,
                                                                  const std::string& device,
                                                                  const std::string& shared_namespace) {
    OPENVINO_ASSERT(!groups.empty(), "A grouped KV cache store needs at least one group");
    if (config.store_library_path.empty()) {
        return open_default_store(groups, config, device, shared_namespace);
    }
    const OvKVCacheStoreProviderDescriptor* provider = load_plugin_provider(config);

    if ((provider->capabilities & OV_KVCACHE_CAP_MULTI_GROUP) != 0) {
        std::vector<PluginGroup> plugin_groups;
        uint64_t capacity_bytes = 0;
        for (const StoreGroupSpec& group : groups) {
            plugin_groups.push_back({group.name, group.layout.get_slot_size(), group.block_tokens});
            capacity_bytes += group.capacity_bytes;
        }
        const auto opened = open_plugin_instance(provider, config, device, shared_namespace, plugin_groups, capacity_bytes);
        std::vector<std::unique_ptr<IKVCacheStore>> views;
        for (const StoreGroupSpec& group : groups) {
            views.push_back(make_plugin_view(opened,
                                             group.name,
                                             group.layout.get_slot_size(),
                                             group.capacity_bytes,
                                             config.store_timeout_ms));
        }
        return std::make_unique<ViewSetKVCacheStore>(std::move(views));
    }

    std::vector<std::unique_ptr<IKVCacheStore>> views;
    for (const StoreGroupSpec& group : groups) {
        const OffloadSettings group_config = config_with_capacity(config, group.capacity_bytes);
        views.push_back(open_plugin_store(provider,
                                          group.layout,
                                          group_config,
                                          device,
                                          group.own_namespace,
                                          group.block_tokens,
                                          group.capacity_bytes));
    }
    return std::make_unique<ViewSetKVCacheStore>(std::move(views));
}

}  // namespace ov::genai
