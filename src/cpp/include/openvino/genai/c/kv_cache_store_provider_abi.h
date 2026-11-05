// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// =============================================================================
// KV cache store plugin ABI (stable, host-neutral, pure C)
//
// This header is the long-term stable binary contract for third-party
// SSD/storage backends ("plugins"). It is consumed by more than one host:
//   * Phase 1: the genai pipeline store tier (KV-cache offloading)
//   * Phase 3: the openvino runtime (device-direct I/O)
// It therefore MUST NOT include, forward-declare, or depend on any genai or
// OpenVINO C++ type. Only C types, fixed-width integers and function pointers.
//
// The "connector" (store registry, the adapter that turns this C ABI into an
// internal C++ seam, the I/O pool, tier/write policy, prefix matching) is host
// private and is NOT part of this header.
//
// -----------------------------------------------------------------------------
// WHAT IS *NOT* IN THIS ABI (deliberately, for stability):
//   * tier / host / write-policy / residency / eviction policy
//   * block / chunk / layer / tensor layout, bytes_per_block, sub-byte packing
//   * the content-hash algorithm and the plugin's internal index structure
//   * token ids, prompt text, or any reversible content ("preimage")
//   * IPrefixIndex / IHostCache / IStagingBuffer / PlacementCopies
//   * C++ types, STL containers, exceptions, ov::* types
// Changing the hash algorithm, index format, sharding or on-disk layout must
// NEVER require an ABI change: those live behind the opaque key and the
// plugin's `store_semantics_fingerprint`.
//
// -----------------------------------------------------------------------------
// ADDRESSING MODEL (single, key-addressed):
//   * The host owns content identity: tokenization, chat-template application,
//     segmentation into fixed, page-granularity blocks, and the stable 128-bit
//     content hash that forms the chain root (arch_002 D22 / B1).
//   * The host passes `OvKVCacheBlockKey` (fixed page-granularity key) plus an
//     opaque payload. Keys are content-addressed, so a plugin can build its own
//     internal index, deduplicate and shard using only the key.
//   * The plugin owns storage identity: how keys map to its own files/objects.
//   * A plugin hit is a *discovery*, not the truth. The host re-derives keys and
//     verifies the returned `witness` before admitting anything; the plugin is
//     never part of the host's correctness TCB.
//
// -----------------------------------------------------------------------------
// THREADING / LIFETIME / ERROR CONTRACT (binding on every implementation):
//   * The host never calls into the plugin while holding a host lock, and no
//     un-terminated async token may exist while the host holds a host lock.
//     `wait` must have reported a terminal status before the host reads any
//     result under a lock.
//   * A store instance may be called concurrently from the host I/O pool unless
//     a provider documents otherwise at open time; implementations must be
//     either internally synchronized or explicitly single-threaded and
//     serialized by the host.
//   * The plugin MUST NOT call back into the host from any entry point.
//   * No C++ exception may cross this boundary. All failures are returned as
//     `OvKVCacheStatus`; unknown/negative statuses are treated as failure.
//   * Synchronous I/O is mandatory. Asynchronous I/O is an OPTIONAL capability
//     (`OV_KVCACHE_CAP_ASYNC_IO`) whose submit/wait/cancel entry points are
//     appended at the tail of the store vtable; a sync-only plugin leaves them
//     unset and stays fully conformant. `struct_bytes` bounds every tail
//     dereference (see VERSIONING below).
//   * Host buffers (`base` / `source` / `witness_out.data`) are owned by the
//     host. For a synchronous call they are valid only for the duration of the
//     call; for an async batch they MUST stay valid and unmodified until the
//     matching `wait` reports a terminal status. Plugin-owned pointers (e.g.
//     `name`) are valid for the module/store lifetime, never per-call scratch.
//   * Batch calls report per-item status. `has_blocks` is side-effect free
//     (probe) and safe to call repeatedly, e.g. during multi-round prefix
//     fixed-point convergence; `read_blocks` is the fetch.
//   * Byte exactness: on read the plugin MUST fill exactly `target.bytes`; a
//     short fill of a host staging ring would leave stale bytes from a previous
//     request. The host hard-rejects `bytes_read != requested` for every item
//     whose status is not `OV_KVCACHE_ERR_CANCELLED`.
//   * Third-party stores MUST return a `witness` that is a strong secondary
//     digest, never raw token ids (see `OvKVCacheWitness`).
//
// -----------------------------------------------------------------------------
// SCOPE:
//   This header covers the *store tier only*. It is not the device buffer ABI
//   (`ICacheBufferProvider`); the two have opposite ownership models and must
//   not be merged.
//
// -----------------------------------------------------------------------------
// VERSIONING:
//   * Major mismatch -> reject. Minor newer than host -> reject.
//   * Capability bits, vtable entries and status codes are append-only; never
//     reuse or renumber. `reserved*` fields must be zeroed by the producer and
//     ignored by the consumer.
//   * Every vtable starts with `struct_bytes` (the size of the structure the
//     producer built). The host MUST NOT dereference a field beyond
//     `struct_bytes`. A capability bit implies the vtable is at least long
//     enough to hold the entries it gates, and the host checks `struct_bytes`
//     before using them.
//   * The plugin library is never dlclose'd by the host.
// =============================================================================

#include <stddef.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// Module export / discovery
// -----------------------------------------------------------------------------

/// Module ABI semantic version. Bumping MAJOR is a hard wire-level break (e.g.
/// a renamed entry symbol); the host rejects a major mismatch loudly instead of
/// loading a stale plugin and failing later on symbol lookup.
#define OV_KVCACHE_STORE_ABI_MAJOR 1
#define OV_KVCACHE_STORE_ABI_MINOR 1

/// The plugin module's single C discovery entry symbol.
#define OV_KVCACHE_STORE_ENTRY_SYMBOL "ov_pipeline_query_kvcache_store_providers"

/// Plugin-side export decoration. The host only ever resolves the single entry
/// symbol above; individual vtable functions are reached through descriptors.
#ifndef OV_KVCACHE_STORE_PLUGIN_EXPORT
#if defined(_WIN32)
#define OV_KVCACHE_STORE_PLUGIN_EXPORT __declspec(dllexport)
#else
#define OV_KVCACHE_STORE_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Status codes
// -----------------------------------------------------------------------------

/// Signed status. 0 == OK. Negative == terminal failure. Positive == non-failure
/// whose meaning is entry-point-specific (`PARTIAL` = partial batch success,
/// `PENDING` = async batch not yet complete). Any status the host does not
/// recognise, or any negative status, is treated as failure.
typedef int32_t OvKVCacheStatus;

enum {
    OV_KVCACHE_OK = 0,
    OV_KVCACHE_PARTIAL = 1,  ///< positive: some batch items succeeded, see per-item status
    OV_KVCACHE_PENDING = 2,  ///< positive: `wait` only, batch not terminal yet, token still valid

    OV_KVCACHE_ERR_INVALID_ARG = -1,
    OV_KVCACHE_ERR_NOT_OPEN = -2,
    OV_KVCACHE_ERR_NOT_FOUND = -3,
    OV_KVCACHE_ERR_CAPACITY = -4,
    OV_KVCACHE_ERR_IO = -5,
    OV_KVCACHE_ERR_TIMEOUT = -6,
    OV_KVCACHE_ERR_UNSUPPORTED = -7,
    OV_KVCACHE_ERR_ABI_MISMATCH = -8,
    OV_KVCACHE_ERR_BUFFER_TOO_SMALL = -9,
    OV_KVCACHE_ERR_INTERNAL = -10,
    OV_KVCACHE_ERR_CANCELLED = -11,  ///< terminal: batch was cancelled; token released
};

// -----------------------------------------------------------------------------
// Capabilities
// -----------------------------------------------------------------------------

/// Append-only capability bitmask (64-bit so later tiers/hosts have room).
typedef uint64_t OvKVCacheCapabilities;

#define OV_KVCACHE_CAP_KEY_IO (1ull << 0)                 ///< required: key-addressed read/write
#define OV_KVCACHE_CAP_EXISTS (1ull << 1)                 ///< `has_blocks` implemented
#define OV_KVCACHE_CAP_REGISTER_HOST_MEMORY (1ull << 2)   ///< provider `register_host_memory` implemented
#define OV_KVCACHE_CAP_SEMANTICS_FINGERPRINT (1ull << 3)  ///< open reports `store_semantics_fingerprint`
#define OV_KVCACHE_CAP_ASYNC_IO (1ull << 4)               ///< async read/write submit + wait + cancel

// -----------------------------------------------------------------------------
// Async completion (optional; OV_KVCACHE_CAP_ASYNC_IO)
// -----------------------------------------------------------------------------

/// Plugin-assigned completion token for one submitted async batch. 0 is never a
/// valid token. Producers MUST allocate tokens from a per-instance monotonic
/// counter and MUST NOT reuse a value within the lifetime of the store instance,
/// so a stale/replayed token can only fail (`OV_KVCACHE_ERR_INVALID_ARG`) and
/// never alias a live batch.
typedef uint64_t OvKVCacheAsyncToken;

/// `wait` timeout sentinel: block until the batch reaches a terminal status.
/// Any other value is a millisecond bound; 0 is a non-blocking poll.
#define OV_KVCACHE_WAIT_INFINITE 0xFFFFFFFFu

// -----------------------------------------------------------------------------
// Target buffer
// -----------------------------------------------------------------------------

/// Buffer kind a plugin may read into. DEVICE is reserved for Phase 3 and MUST
/// be rejected by a Phase 1 host; the `ctx` slot is likewise reserved and its
/// concrete meaning (cl_mem vs ze handle vs handle table) is intentionally NOT
/// frozen yet.
typedef uint32_t OvKVCacheTargetKind;

enum {
    OV_KVCACHE_TARGET_HOST = 1,    ///< host memory, contiguous (Phase 1: staging lease)
    OV_KVCACHE_TARGET_DEVICE = 2,  ///< reserved, Phase 3
};

typedef struct OvKVCacheTarget {
    OvKVCacheTargetKind kind;
    uint32_t reserved0;
    void* base;      ///< target memory; host-owned; valid for the call, or until `wait` is terminal for an async read
    uint64_t bytes;  ///< exact number of bytes the plugin must read/write
    void* ctx;       ///< reserved (device context / memory handle), Phase 3
} OvKVCacheTarget;

// -----------------------------------------------------------------------------
// Content identity (host-computed) and block key
// -----------------------------------------------------------------------------

/// Stable 128-bit content digest. Byte layout (lo/hi, endianness) is part of
/// the frozen contract.
typedef struct OvKVCacheContentHash {
    uint64_t lo;
    uint64_t hi;
} OvKVCacheContentHash;

/// Fixed page-granularity block key. `group` identifies the cache group (e.g.
/// "kv", "conv_state_table"); it is borrowed and valid only for the call.
/// Keys are always fixed page-granularity: a "segment" is expressed as an
/// array of keys, never as {seq_id, start, end}.
typedef struct OvKVCacheBlockKey {
    const char* group;
    uint32_t group_len;
    uint32_t reserved0;
    OvKVCacheContentHash hash;
} OvKVCacheBlockKey;

/// Opaque, host-resolved namespace (namespace key + model/tokenizer/spec
/// fingerprints). The plugin treats it as an opaque byte range.
typedef struct OvKVCacheNamespace {
    const uint8_t* bytes;
    uint64_t len;
} OvKVCacheNamespace;

/// Host-defined, host-verified witness persisted next to a block.
///
/// On write the host supplies this value; the plugin stores it opaquely and
/// returns it unchanged on read. The host compares it before admitting a block.
/// For third-party stores it MUST be a strong secondary digest derived with a
/// host-private algorithm/key -- NEVER raw token ids (which would be reversible
/// and would leak user content across the ABI). Its algorithm version belongs
/// in the namespace/spec fingerprint.
typedef struct OvKVCacheWitness {
    const void* data;
    uint64_t len;
} OvKVCacheWitness;

/// Host-provided storage for the persisted witness of one read. The plugin
/// writes the witness bytes into `data` (up to `capacity`) and reports the
/// written length via `out[i].witness.len`; `out[i].witness.data` is set to
/// `data`. The host owns this buffer and verifies its contents after the read
/// completes, so the plugin MUST NOT return a pointer into its own storage.
/// `capacity` too small for the stored witness is `OV_KVCACHE_ERR_BUFFER_TOO_SMALL`.
typedef struct OvKVCacheWitnessBuffer {
    void* data;
    uint64_t capacity;
} OvKVCacheWitnessBuffer;

// -----------------------------------------------------------------------------
// Read / write batches
// -----------------------------------------------------------------------------

typedef struct OvKVCacheStoreReadRequest {
    OvKVCacheBlockKey key;
    OvKVCacheTarget target;
    OvKVCacheWitnessBuffer witness_out;  ///< host-owned; plugin writes the persisted witness here
} OvKVCacheStoreReadRequest;

typedef struct OvKVCacheStoreReadResult {
    OvKVCacheStatus status;
    uint64_t bytes_read;
    /// Persisted witness, for host verification. `.data` mirrors the request's
    /// `witness_out.data` (host memory) and `.len` is the number of bytes the
    /// plugin wrote there; zeroed when the item failed.
    OvKVCacheWitness witness;
} OvKVCacheStoreReadResult;

typedef struct OvKVCacheStoreWriteRequest {
    OvKVCacheBlockKey key;
    const void* source;        ///< host-owned; valid for the call, or until `wait` is terminal for an async write
    uint64_t bytes;            ///< must equal the block's canonical byte size
    OvKVCacheWitness witness;  ///< host-supplied, stored verbatim
} OvKVCacheStoreWriteRequest;

typedef struct OvKVCacheStoreWriteResult {
    OvKVCacheStatus status;
    uint64_t bytes_written;
} OvKVCacheStoreWriteResult;

// -----------------------------------------------------------------------------
// Stats / open options / namespace listing
// -----------------------------------------------------------------------------

typedef struct OvKVCacheStoreStats {
    uint64_t bytes;
    uint64_t blocks;
    uint64_t chunks;
} OvKVCacheStoreStats;

/// Flat string property bag (deliberately not ov::AnyMap, which cannot cross a
/// C ABI). Keys and values are UTF-8, borrowed for the call.
typedef struct OvKVCacheProperty {
    const char* key;
    const char* value;
} OvKVCacheProperty;

/// Describes one cache group the host will use. Strings are borrowed for the `open` call.
typedef struct OvKVCacheGroupInfo {
    const char* group;       ///< group name; equals the namespace passed to `open`
    uint32_t group_len;
    uint32_t block_tokens;   ///< tokens one block covers (for a state group: the checkpoint interval); 0 if not token-addressed
    uint64_t block_bytes;    ///< exact byte size of every read and write of this group
} OvKVCacheGroupInfo;

/// Host log severity: ERROR 1, WARNING 2, INFO 3, DEBUG 4.
typedef uint32_t OvKVCacheLogLevel;
#define OV_KVCACHE_LOG_ERROR 1
#define OV_KVCACHE_LOG_WARNING 2
#define OV_KVCACHE_LOG_INFO 3
#define OV_KVCACHE_LOG_DEBUG 4

/// Host logger handed to the plugin. `message` is NUL-terminated and borrowed for the call;
/// the callback is thread-safe and valid for the process lifetime.
typedef void (*OvKVCacheLogFn)(OvKVCacheLogLevel level, const char* message);

/// Fields after `reserved0` were appended in ABI 1.1. A plugin may read them only if it was built
/// against 1.1 or newer; the host never loads a plugin whose minor exceeds its own.
typedef struct OvKVCacheStoreOpenOptions {
    const char* directory;
    uint64_t capacity_bytes;
    const OvKVCacheProperty* properties;
    uint64_t property_count;
    uint32_t reserved0;
    // ---- appended in 1.1 ----
    const OvKVCacheGroupInfo* groups;  ///< every cache group the host will use; may be NULL when `group_count` is 0
    uint64_t group_count;
    uint64_t witness_bytes_max;        ///< largest witness the host will pass to a write
    OvKVCacheLogFn log;                ///< host logger; may be NULL
} OvKVCacheStoreOpenOptions;

// Forward declaration: the store instance vtable is produced by `open` and is
// defined after the request/result types it uses.
struct OvKVCacheStoreVTable;

typedef struct OvKVCacheStoreOpenResult {
    OvKVCacheStatus status;
    const struct OvKVCacheStoreVTable* vtable;         ///< store instance operations; MUST be set on success
    void* store;                                       ///< opaque store instance (the vtable's `self`)
    OvKVCacheContentHash store_semantics_fingerprint;  ///< valid iff has_semantics_fingerprint != 0
    uint8_t has_semantics_fingerprint;
    uint8_t reserved0[7];
} OvKVCacheStoreOpenResult;

typedef struct OvKVCacheStoredNamespaceInfo {
    OvKVCacheNamespace ns;
    uint64_t bytes;
    uint64_t blocks;
} OvKVCacheStoredNamespaceInfo;

// -----------------------------------------------------------------------------
// Store instance vtable (per namespace)
// -----------------------------------------------------------------------------

/// One store instance per namespace, created by `open`. Operations are
/// batch-oriented with per-item statuses. The synchronous group is mandatory;
/// the async group at the tail exists only when `OV_KVCACHE_CAP_ASYNC_IO` is
/// advertised by the provider.
///
/// `has_blocks` is the pure probe (no side effects, safe for repeated
/// multi-round prefix fixed-point convergence); `read_blocks` is the fetch.
typedef struct OvKVCacheStoreVTable {
    /// Size of this structure in bytes, set by the producer. The host MUST NOT
    /// read a tail entry beyond `struct_bytes` (append-only minor-version safety).
    uint32_t struct_bytes;
    uint32_t reserved0;

    /// Destroy the store instance and release its resources. Null-safe. MUST
    /// block until every outstanding async token on this instance is terminal
    /// and the plugin will no longer touch any host buffer.
    void (*destroy)(void* self);

    /// Probe. `out_present[i]` is 0 or 1 for `keys[i]`. Side-effect free.
    OvKVCacheStatus (*has_blocks)(void* self, const OvKVCacheBlockKey* keys, uint8_t* out_present, size_t count);

    /// Fetch. On success each item fills exactly `reqs[i].target.bytes` and
    /// writes the persisted witness into `reqs[i].witness_out`. The host rejects
    /// a short/long fill.
    OvKVCacheStatus (*read_blocks)(void* self,
                                   const OvKVCacheStoreReadRequest* reqs,
                                   OvKVCacheStoreReadResult* out,
                                   size_t count);

    /// Put. Idempotent per key; overwriting a key advances the plugin's own
    /// internal epoch. The supplied witness is stored verbatim.
    OvKVCacheStatus (*write_blocks)(void* self,
                                    const OvKVCacheStoreWriteRequest* reqs,
                                    OvKVCacheStoreWriteResult* out,
                                    size_t count);

    /// Drop the given keys. Missing keys are not an error.
    OvKVCacheStatus (*drop_blocks)(void* self, const OvKVCacheBlockKey* keys, size_t count);

    /// Flush pending writes to the backing store.
    OvKVCacheStatus (*flush)(void* self);

    /// Snapshot of capacity accounting.
    OvKVCacheStatus (*stats)(void* self, OvKVCacheStoreStats* out);

    // ---- async I/O (present only with OV_KVCACHE_CAP_ASYNC_IO) ----------------

    /// Submit a read batch without blocking. On success `*out_token` is non-zero
    /// and `wait` MUST be driven to a terminal status. On synchronous failure
    /// `*out_token` is 0 and no `wait` is required. `PARTIAL` is not a valid
    /// submit result. Per-item `out[i]` is undefined until the batch is
    /// terminal; `reqs`, `out`, every `reqs[i].target.base` and every
    /// `reqs[i].witness_out.data` stay host-owned and MUST remain valid and
    /// unmodified until then.
    OvKVCacheStatus (*read_blocks_async)(void* self,
                                         const OvKVCacheStoreReadRequest* reqs,
                                         OvKVCacheStoreReadResult* out,
                                         size_t count,
                                         OvKVCacheAsyncToken* out_token);

    /// Submit a write batch without blocking. Same completion/lifetime rules as
    /// `read_blocks_async`; `reqs[i].source` and `reqs[i].witness.data` stay
    /// valid until the batch is terminal.
    OvKVCacheStatus (*write_blocks_async)(void* self,
                                          const OvKVCacheStoreWriteRequest* reqs,
                                          OvKVCacheStoreWriteResult* out,
                                          size_t count,
                                          OvKVCacheAsyncToken* out_token);

    /// Block the calling thread until `token` is terminal or `timeout_ms`
    /// elapses. Exactly one waiter per token. `OK`/`PARTIAL` = terminal success,
    /// per-item statuses final, token released. `ERR_CANCELLED` = terminal after
    /// a `cancel`, token released. `PENDING` = not terminal, token still valid,
    /// call again. Any other status is terminal failure and releases the token.
    /// `timeout_ms == 0` polls without blocking; `OV_KVCACHE_WAIT_INFINITE` blocks.
    OvKVCacheStatus (*wait)(void* self, OvKVCacheAsyncToken token, uint32_t timeout_ms);

    /// Best-effort, non-blocking cancel. Queued items are dropped; items already
    /// in flight are allowed to finish with their results discarded. `cancel`
    /// does NOT release the token: the host must still drive `wait` to a terminal
    /// status before reusing buffers. Callable from any thread, concurrently with
    /// `wait`; the returned status is advisory only.
    OvKVCacheStatus (*cancel)(void* self, OvKVCacheAsyncToken token);
} OvKVCacheStoreVTable;

// -----------------------------------------------------------------------------
// Provider vtable (factory)
// -----------------------------------------------------------------------------

/// One provider instance per loaded module, discovered via
/// `OvKVCacheStoreModuleInfo::get_provider`.
typedef struct OvKVCacheStoreProviderVTable {
    /// Size of this structure in bytes, set by the producer. The host MUST NOT
    /// read a tail entry beyond `struct_bytes` (append-only minor-version safety).
    uint32_t struct_bytes;
    uint32_t reserved0;

    /// Destroy the provider instance. Null-safe.
    void (*destroy)(void* self);

    /// Stable provider name (e.g. "file", "vendor-x"). Valid module lifetime.
    const char* (*name)(void* self);

    /// Open (or create) a store instance for a namespace. On success both
    /// `out->store` and `out->vtable` MUST be set, and `out->vtable->struct_bytes`
    /// must cover the entries implied by the advertised capabilities.
    OvKVCacheStatus (*open)(void* self,
                            const OvKVCacheNamespace* ns,
                            const OvKVCacheStoreOpenOptions* opts,
                            OvKVCacheStoreOpenResult* out);

    /// Atomically purge every block of a namespace.
    OvKVCacheStatus (*purge)(void* self, const OvKVCacheNamespace* ns);

    /// Enumerate stored namespaces. Fills up to `capacity` entries; `out_count`
    /// receives the total number available. Pass `out == NULL` to query the
    /// count only.
    OvKVCacheStatus (*list)(void* self, OvKVCacheStoredNamespaceInfo* out, size_t capacity, size_t* out_count);

    /// Optional (OV_KVCACHE_CAP_REGISTER_HOST_MEMORY): one-time registration of
    /// a host memory region for RDMA MR / GPUDirect Storage style backends.
    OvKVCacheStatus (*register_host_memory)(void* self, void* base, uint64_t bytes);
} OvKVCacheStoreProviderVTable;

// -----------------------------------------------------------------------------
// Module descriptor
// -----------------------------------------------------------------------------

typedef struct OvKVCacheStoreProviderDescriptor {
    const OvKVCacheStoreProviderVTable* vtable;
    void* self;
    OvKVCacheCapabilities capabilities;
    const char* name;
} OvKVCacheStoreProviderDescriptor;

/// Standard-layout self-description the plugin hands to the host. Read directly
/// by memory layout across the ABI; see the static_asserts below.
typedef struct OvKVCacheStoreModuleInfo {
    uint32_t abi_major;                 ///< must == host MAJOR, else rejected
    uint32_t abi_minor;                 ///< must be <= host MINOR, else rejected
    const char* sdk_build_fingerprint;  ///< must equal the host SDK fingerprint byte-wise
    const char* module_name;
    const char* module_version;
    uint32_t provider_count;
    uint32_t reserved0;
    /// Returns the provider descriptor at `index` (< provider_count), or NULL.
    /// Valid for the module lifetime.
    const OvKVCacheStoreProviderDescriptor* (*get_provider)(uint32_t index);
} OvKVCacheStoreModuleInfo;

// -----------------------------------------------------------------------------
// C entry point (implemented by the plugin)
// -----------------------------------------------------------------------------

/// Returns the module descriptor, valid for the module lifetime. Resolved by
/// the host via dlsym / GetProcAddress using OV_KVCACHE_STORE_ENTRY_SYMBOL.
OV_KVCACHE_STORE_PLUGIN_EXPORT const OvKVCacheStoreModuleInfo* ov_pipeline_query_kvcache_store_providers(void);

#ifdef __cplusplus
}  // extern "C"
#endif

// -----------------------------------------------------------------------------
// Compile-time layout guarantees (C++ translation units only; no implementation)
// -----------------------------------------------------------------------------

#ifdef __cplusplus
#include <cstddef>
#include <type_traits>

static_assert(std::is_standard_layout<OvKVCacheTarget>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheContentHash>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheBlockKey>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheNamespace>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheWitness>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheWitnessBuffer>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreReadRequest>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreReadResult>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreWriteRequest>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreWriteResult>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreStats>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheProperty>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheGroupInfo>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreOpenOptions>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreOpenResult>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoredNamespaceInfo>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreProviderDescriptor>::value, "ABI struct must be standard layout");
static_assert(std::is_standard_layout<OvKVCacheStoreModuleInfo>::value, "ABI struct must be standard layout");

// `struct_bytes` must lead every vtable: the host uses it to bound tail reads.
static_assert(offsetof(OvKVCacheStoreVTable, struct_bytes) == 0, "vtable must start with struct_bytes");
static_assert(offsetof(OvKVCacheStoreProviderVTable, struct_bytes) == 0, "vtable must start with struct_bytes");
#endif
