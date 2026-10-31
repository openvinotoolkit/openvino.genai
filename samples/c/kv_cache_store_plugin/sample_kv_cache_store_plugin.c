// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Minimal, illustrative third-party KV cache store plugin.
//
// Implements the full ABI surface declared in kv_cache_store_provider_abi.h using nothing but a
// process-local, in-memory linked list - NOT meant for production use (no persistence, no real
// I/O, naive O(n) key lookup, unbounded memory growth) but enough to show exactly what a vendor
// plugin needs to implement and how the host drives it. A real plugin would replace the storage
// in store_read_blocks/store_write_blocks/store_drop_blocks with actual SSD I/O and would track
// its own capacity/eviction instead of growing forever.
//
// Build as a shared library (see CMakeLists.txt in this directory) and point a host's
// `store_library_path` configuration at the resulting .dll/.so; the host discovers it purely
// through the single exported entry symbol at the bottom of this file. This file has no
// dependency beyond the one ABI header - that is the point.

#include "openvino/genai/c/kv_cache_store_provider_abi.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#    include <windows.h>
typedef CRITICAL_SECTION sample_mutex_t;
static void sample_mutex_init(sample_mutex_t* m) {
    InitializeCriticalSection(m);
}
static void sample_mutex_lock(sample_mutex_t* m) {
    EnterCriticalSection(m);
}
static void sample_mutex_unlock(sample_mutex_t* m) {
    LeaveCriticalSection(m);
}
#else
#    include <pthread.h>
typedef pthread_mutex_t sample_mutex_t;
static void sample_mutex_init(sample_mutex_t* m) {
    pthread_mutex_init(m, NULL);
}
static void sample_mutex_lock(sample_mutex_t* m) {
    pthread_mutex_lock(m);
}
static void sample_mutex_unlock(sample_mutex_t* m) {
    pthread_mutex_unlock(m);
}
#endif

/* --------------------------------------------------------------------------
 * A single stored block: payload + the witness the host asked us to keep verbatim, plus a copy of
 * the key's group so unrelated callers sharing one store instance (e.g. KV cache offload vs.
 * linear-attention offload vs. session snapshot) never collide on a bare hash match.
 * ----------------------------------------------------------------------- */
typedef struct SampleEntry {
    OvKVCacheContentHash hash;
    char* group;
    uint32_t group_len;
    void* data;
    uint64_t bytes;
    void* witness;
    uint64_t witness_len;
    struct SampleEntry* next;
} SampleEntry;

typedef struct SampleStore {
    SampleEntry* head;
    sample_mutex_t mutex;
} SampleStore;

static int hash_equal(const OvKVCacheContentHash* a, const OvKVCacheContentHash* b) {
    return a->lo == b->lo && a->hi == b->hi;
}

static int key_matches_entry(const SampleEntry* entry, const OvKVCacheBlockKey* key) {
    return hash_equal(&entry->hash, &key->hash) && entry->group_len == key->group_len &&
           (key->group_len == 0 || memcmp(entry->group, key->group, key->group_len) == 0);
}

/* Caller must hold store->mutex. */
static SampleEntry* find_locked(SampleStore* store, const OvKVCacheBlockKey* key) {
    SampleEntry* entry;
    for (entry = store->head; entry != NULL; entry = entry->next) {
        if (key_matches_entry(entry, key)) {
            return entry;
        }
    }
    return NULL;
}

/* Caller must hold store->mutex. Not an error if the key is not present. */
static void remove_locked(SampleStore* store, const OvKVCacheBlockKey* key) {
    SampleEntry** link = &store->head;
    while (*link != NULL) {
        if (key_matches_entry(*link, key)) {
            SampleEntry* dead = *link;
            *link = dead->next;
            free(dead->group);
            free(dead->data);
            free(dead->witness);
            free(dead);
            return;
        }
        link = &(*link)->next;
    }
}

/* --------------------------------------------------------------------------
 * Store instance vtable
 * ----------------------------------------------------------------------- */

static void store_destroy(void* self) {
    /* Store instances are shared across open() calls for the same directory (see provider_open),
     * so destroying one handle must not free data a sibling handle might still read - this sample
     * simply leaks the shared instance for the lifetime of the process, same as a real plugin
     * would leave its on-disk files alone when a caller merely closes its handle. */
    (void)self;
}

static OvKVCacheStatus store_has_blocks(void* self,
                                        const OvKVCacheBlockKey* keys,
                                        uint8_t* out_present,
                                        size_t count) {
    SampleStore* store = (SampleStore*)self;
    size_t i;
    sample_mutex_lock(&store->mutex);
    for (i = 0; i < count; ++i) {
        out_present[i] = find_locked(store, &keys[i]) != NULL ? 1 : 0;
    }
    sample_mutex_unlock(&store->mutex);
    return OV_KVCACHE_OK;
}

static OvKVCacheStatus store_read_blocks(void* self,
                                         const OvKVCacheStoreReadRequest* reqs,
                                         OvKVCacheStoreReadResult* out,
                                         size_t count) {
    SampleStore* store = (SampleStore*)self;
    int any_failed = 0;
    size_t i;
    sample_mutex_lock(&store->mutex);
    for (i = 0; i < count; ++i) {
        const OvKVCacheStoreReadRequest* req = &reqs[i];
        OvKVCacheStoreReadResult* result = &out[i];
        SampleEntry* entry;
        memset(result, 0, sizeof(*result));

        if (req->target.kind != OV_KVCACHE_TARGET_HOST || req->target.base == NULL) {
            result->status = OV_KVCACHE_ERR_UNSUPPORTED;
            any_failed = 1;
            continue;
        }
        entry = find_locked(store, &req->key);
        if (entry == NULL) {
            result->status = OV_KVCACHE_ERR_NOT_FOUND;
            any_failed = 1;
            continue;
        }
        if (entry->bytes != req->target.bytes) {
            result->status = OV_KVCACHE_ERR_INTERNAL;
            any_failed = 1;
            continue;
        }
        memcpy(req->target.base, entry->data, (size_t)entry->bytes);

        if (req->witness_out.data != NULL && entry->witness_len > 0) {
            if (entry->witness_len > req->witness_out.capacity) {
                result->status = OV_KVCACHE_ERR_BUFFER_TOO_SMALL;
                any_failed = 1;
                continue;
            }
            memcpy(req->witness_out.data, entry->witness, (size_t)entry->witness_len);
            result->witness.data = req->witness_out.data;
            result->witness.len = entry->witness_len;
        }
        result->status = OV_KVCACHE_OK;
        result->bytes_read = entry->bytes;
    }
    sample_mutex_unlock(&store->mutex);
    return any_failed ? OV_KVCACHE_PARTIAL : OV_KVCACHE_OK;
}

static OvKVCacheStatus store_write_blocks(void* self,
                                          const OvKVCacheStoreWriteRequest* reqs,
                                          OvKVCacheStoreWriteResult* out,
                                          size_t count) {
    SampleStore* store = (SampleStore*)self;
    int any_failed = 0;
    size_t i;
    sample_mutex_lock(&store->mutex);
    for (i = 0; i < count; ++i) {
        const OvKVCacheStoreWriteRequest* req = &reqs[i];
        OvKVCacheStoreWriteResult* result = &out[i];
        void* data_copy;
        void* witness_copy;
        char* group_copy;
        SampleEntry* entry;
        memset(result, 0, sizeof(*result));

        data_copy = malloc((size_t)req->bytes);
        witness_copy = req->witness.len > 0 ? malloc((size_t)req->witness.len) : NULL;
        group_copy = req->key.group_len > 0 ? malloc(req->key.group_len) : NULL;
        if (data_copy == NULL || (req->witness.len > 0 && witness_copy == NULL) ||
            (req->key.group_len > 0 && group_copy == NULL)) {
            free(data_copy);
            free(witness_copy);
            free(group_copy);
            result->status = OV_KVCACHE_ERR_CAPACITY;
            any_failed = 1;
            continue;
        }
        memcpy(data_copy, req->source, (size_t)req->bytes);
        if (witness_copy != NULL) {
            memcpy(witness_copy, req->witness.data, (size_t)req->witness.len);
        }
        if (group_copy != NULL) {
            memcpy(group_copy, req->key.group, req->key.group_len);
        }

        entry = (SampleEntry*)malloc(sizeof(SampleEntry));
        if (entry == NULL) {
            free(data_copy);
            free(witness_copy);
            free(group_copy);
            result->status = OV_KVCACHE_ERR_CAPACITY;
            any_failed = 1;
            continue;
        }

        /* Idempotent overwrite: drop any existing entry for this key first. */
        remove_locked(store, &req->key);

        entry->hash = req->key.hash;
        entry->group = group_copy;
        entry->group_len = req->key.group_len;
        entry->data = data_copy;
        entry->bytes = req->bytes;
        entry->witness = witness_copy;
        entry->witness_len = req->witness.len;
        entry->next = store->head;
        store->head = entry;

        result->status = OV_KVCACHE_OK;
        result->bytes_written = req->bytes;
    }
    sample_mutex_unlock(&store->mutex);
    return any_failed ? OV_KVCACHE_PARTIAL : OV_KVCACHE_OK;
}

static OvKVCacheStatus store_drop_blocks(void* self, const OvKVCacheBlockKey* keys, size_t count) {
    SampleStore* store = (SampleStore*)self;
    size_t i;
    sample_mutex_lock(&store->mutex);
    for (i = 0; i < count; ++i) {
        remove_locked(store, &keys[i]);
    }
    sample_mutex_unlock(&store->mutex);
    return OV_KVCACHE_OK;
}

static OvKVCacheStatus store_flush(void* self) {
    (void)self; /* Everything lives in memory; nothing to flush. */
    return OV_KVCACHE_OK;
}

static OvKVCacheStatus store_stats(void* self, OvKVCacheStoreStats* out) {
    SampleStore* store = (SampleStore*)self;
    uint64_t blocks = 0;
    uint64_t bytes = 0;
    SampleEntry* entry;
    sample_mutex_lock(&store->mutex);
    for (entry = store->head; entry != NULL; entry = entry->next) {
        ++blocks;
        bytes += entry->bytes;
    }
    sample_mutex_unlock(&store->mutex);
    out->blocks = blocks;
    out->bytes = bytes;
    out->chunks = 0;
    return OV_KVCACHE_OK;
}

static const OvKVCacheStoreVTable g_store_vtable = {
    sizeof(OvKVCacheStoreVTable),
    0,
    &store_destroy,
    &store_has_blocks,
    &store_read_blocks,
    &store_write_blocks,
    &store_drop_blocks,
    &store_flush,
    &store_stats,
    NULL, /* read_blocks_async: this sample does not advertise OV_KVCACHE_CAP_ASYNC_IO */
    NULL, /* write_blocks_async */
    NULL, /* wait */
    NULL, /* cancel */
};

/* --------------------------------------------------------------------------
 * Provider vtable
 * ----------------------------------------------------------------------- */

static void provider_destroy(void* self) {
    (void)self; /* Stateless: nothing was allocated for the provider itself. */
}

static const char* provider_name(void* self) {
    (void)self;
    return "sample-in-memory";
}

/* --------------------------------------------------------------------------
 * Directory registry: shares one SampleStore per (`directory`, namespace) across open() calls, so
 * a caller that closes its handle and later reopens the same directory still finds what was
 * written - what any real on-disk backend would do. A NULL/empty directory shares one
 * process-wide default instance. The namespace is part of the key because the host sizes each
 * store against its own capacity and checks stats() against it: KV blocks and linear-attention
 * blocks (different namespaces) must not be counted in one store. Entries are never removed (same
 * "nothing is ever deleted behind the host's back" spirit as store_destroy being a no-op); this is
 * a reference sample, not a long-running service.
 * ----------------------------------------------------------------------- */
typedef struct DirectoryStore {
    char* directory;
    uint8_t* ns_bytes;
    uint64_t ns_len;
    SampleStore* store;
    struct DirectoryStore* next;
} DirectoryStore;

static DirectoryStore* g_directories = NULL;
static sample_mutex_t g_registry_mutex;
static int g_registry_mutex_ready = 0;

/* Illustrative sample: this check is not safe against two threads racing on the very first
 * open() call (same simplification as the rest of this file). A real plugin would use a portable
 * static initializer or an explicit one-time init hook instead. */
static void ensure_registry_mutex(void) {
    if (!g_registry_mutex_ready) {
        sample_mutex_init(&g_registry_mutex);
        g_registry_mutex_ready = 1;
    }
}

static OvKVCacheStatus provider_open(void* self,
                                     const OvKVCacheNamespace* ns,
                                     const OvKVCacheStoreOpenOptions* opts,
                                     OvKVCacheStoreOpenResult* out) {
    const char* directory;
    SampleStore* store = NULL;
    DirectoryStore* it;
    uint64_t ns_len;
    const uint8_t* ns_bytes;
    (void)self; /* A real plugin would honor capacity_bytes/properties too; this sample ignores them. */
    if (out == NULL) {
        return OV_KVCACHE_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));

    directory = (opts != NULL && opts->directory != NULL) ? opts->directory : "";
    ns_len = (ns != NULL && ns->bytes != NULL) ? ns->len : 0;
    ns_bytes = ns_len > 0 ? ns->bytes : NULL;

    /* ABI 1.1: the host passes group sizes and a logger; a real plugin would pre-size its storage from opts->groups. */
    if (opts != NULL && opts->log != NULL) {
        opts->log(OV_KVCACHE_LOG_INFO, "sample-in-memory: opening store");
    }

    ensure_registry_mutex();
    sample_mutex_lock(&g_registry_mutex);
    for (it = g_directories; it != NULL; it = it->next) {
        if (strcmp(it->directory, directory) == 0 && it->ns_len == ns_len &&
            (ns_len == 0 || memcmp(it->ns_bytes, ns_bytes, (size_t)ns_len) == 0)) {
            store = it->store;
            break;
        }
    }
    if (store == NULL) {
        DirectoryStore* created = (DirectoryStore*)malloc(sizeof(DirectoryStore));
        SampleStore* fresh = created != NULL ? (SampleStore*)malloc(sizeof(SampleStore)) : NULL;
        char* directory_copy = fresh != NULL ? (char*)malloc(strlen(directory) + 1) : NULL;
        uint8_t* ns_copy = (directory_copy != NULL && ns_len > 0) ? (uint8_t*)malloc((size_t)ns_len) : NULL;
        if (created == NULL || fresh == NULL || directory_copy == NULL || (ns_len > 0 && ns_copy == NULL)) {
            free(created);
            free(fresh);
            free(directory_copy);
            free(ns_copy);
            sample_mutex_unlock(&g_registry_mutex);
            out->status = OV_KVCACHE_ERR_CAPACITY;
            return out->status;
        }
        fresh->head = NULL;
        sample_mutex_init(&fresh->mutex);
        strcpy(directory_copy, directory);
        if (ns_len > 0) {
            memcpy(ns_copy, ns_bytes, (size_t)ns_len);
        }
        created->directory = directory_copy;
        created->ns_bytes = ns_copy;
        created->ns_len = ns_len;
        created->store = fresh;
        created->next = g_directories;
        g_directories = created;
        store = fresh;
    }
    sample_mutex_unlock(&g_registry_mutex);

    out->store = store;
    out->vtable = &g_store_vtable;
    out->status = OV_KVCACHE_OK;
    return out->status;
}

static OvKVCacheStatus provider_purge(void* self, const OvKVCacheNamespace* ns) {
    (void)self;
    (void)ns;
    /* This sample keeps no state outside of individual store instances, so there is nothing to
     * purge independently of destroying the instance that was opened. */
    return OV_KVCACHE_ERR_UNSUPPORTED;
}

static OvKVCacheStatus provider_list(void* self,
                                     OvKVCacheStoredNamespaceInfo* out,
                                     size_t capacity,
                                     size_t* out_count) {
    (void)self;
    (void)out;
    (void)capacity;
    if (out_count != NULL) {
        *out_count = 0;
    }
    return OV_KVCACHE_OK;
}

static const OvKVCacheStoreProviderVTable g_provider_vtable = {
    sizeof(OvKVCacheStoreProviderVTable),
    0,
    &provider_destroy,
    &provider_name,
    &provider_open,
    &provider_purge,
    &provider_list,
    NULL, /* register_host_memory: OV_KVCACHE_CAP_REGISTER_HOST_MEMORY not advertised */
};

static const OvKVCacheStoreProviderDescriptor g_provider_descriptor = {
    &g_provider_vtable,
    NULL,
    OV_KVCACHE_CAP_KEY_IO | OV_KVCACHE_CAP_EXISTS,
    "sample-in-memory",
};

static const OvKVCacheStoreProviderDescriptor* get_provider(uint32_t index) {
    return index == 0 ? &g_provider_descriptor : NULL;
}

/* --------------------------------------------------------------------------
 * Module descriptor and the single exported entry point
 * ----------------------------------------------------------------------- */

/* A real plugin build must set this to match the SDK/ABI build it was compiled against; the host
 * rejects any mismatch rather than risk loading a plugin built for an incompatible layout. There
 * is no public, stable value to hardcode here - a vendor obtains the correct fingerprint from the
 * host SDK release they target. The literal below only matches this repository's own dev builds,
 * which is why this sample can successfully load against the in-tree test host. */
#define SAMPLE_SDK_BUILD_FINGERPRINT "ov-genai-kvcache-store-abi-dev"

static const OvKVCacheStoreModuleInfo g_module_info = {
    OV_KVCACHE_STORE_ABI_MAJOR,
    OV_KVCACHE_STORE_ABI_MINOR,
    SAMPLE_SDK_BUILD_FINGERPRINT,
    "sample-kv-cache-store-plugin",
    "1.0",
    1,
    0,
    &get_provider,
};

OV_KVCACHE_STORE_PLUGIN_EXPORT const OvKVCacheStoreModuleInfo* ov_pipeline_query_kvcache_store_providers(void) {
    return &g_module_info;
}
