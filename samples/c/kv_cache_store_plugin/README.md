# KV Cache Store Plugin Sample (`sample_kv_cache_store_plugin`)

[E2E validation guide for your own plugin](E2E_VALIDATION.md)

A minimal, standalone, third-party-style implementation of the KV cache store plugin ABI
declared in
[`openvino/genai/c/kv_cache_store_provider_abi.h`](../../../src/cpp/include/openvino/genai/c/kv_cache_store_provider_abi.h).
Use it as a starting point for a real vendor storage plugin (e.g. an SSD-backed store), or to
sanity-check that your own host integration loads and drives a plugin correctly.

## What this is

By default, OpenVINO GenAI's KV cache offload (`SchedulerConfig::cache_offload`) writes
evicted blocks to a file under `kv_cache_dir` (the built-in default store). This ABI lets a third party replace that
default with its own storage backend - without linking against `openvino::genai` or any C++ code
at all - by shipping a shared library that exports one C entry symbol.

This sample implements the **full** ABI surface and exports two providers:

- `sample-in-memory`: a process-local, in-memory linked list protected by a mutex. Deliberately **not production
  quality** (no persistence, no real I/O, O(n) key lookup, unbounded memory growth) - its only job is to show
  exactly what a vendor plugin needs to implement, and to prove the pattern works end-to-end with a real model.
- `sample-persistent-file`: one file per block under `CacheOffloadConfig.kv_cache_dir` (set it to a directory), written to a
  temporary name, synced and renamed. Blocks survive the process, so blocks stored by one process (see
  `CacheOffloadWritePolicy.WRITE_THROUGH`) can be reused by another (see `verify_chunk_store.py`). Still a reference:
  no eviction, no locking between processes, and a store reopened with a smaller budget than the blocks it already
  holds is rejected by the host.
- `sample-self-evict-memory`: the in-memory store, bounded by its byte budget and evicting its own oldest blocks when
  full. It also shows the ABI 1.3 capabilities: one instance for all cache groups and the tokens of linear attention
  checkpoints.

Key properties demonstrated by this sample:
- **Zero dependency on genai.** The `.c` file only includes the one ABI header; the CMake target
  is not linked against `openvino::genai`.
- **Single entry point.** The host discovers everything through one exported C symbol,
  `ov_pipeline_query_kvcache_store_providers` (see the bottom of
  [`sample_kv_cache_store_plugin.c`](sample_kv_cache_store_plugin.c)).
- **Plugin hits are a discovery, not the truth.** The host always re-derives the content hash and
  verifies the `witness` this plugin returns before trusting any bytes read back from it.

## Build

Built automatically as part of the regular GenAI samples build (see
[`samples/CMakeLists.txt`](../../CMakeLists.txt)); no extra flags are required:

```sh
cmake -S . -B build -DENABLE_SAMPLES=ON ...
cmake --build build --target sample_kv_cache_store_plugin
```

This produces a shared library next to the other sample binaries
(`sample_kv_cache_store_plugin.dll` on Windows, `libsample_kv_cache_store_plugin.so`/`.dylib` on
Linux/macOS).

## Use it with a host pipeline

Point `CacheOffloadConfig.plugin` at the built library. Setting a plugin with a `library_path` is the explicit
opt-in to load third-party code; a plugin that cannot be loaded, fails the ABI handshake, has no such provider or cannot
open a store makes the pipeline fail to start, instead of falling back to the default store. The SSD tier needs an
existing `kv_cache_dir`, which this in-memory sample does not use:

```python
import openvino_genai as ov_genai

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.enable_prefix_caching = True  # offloaded blocks are rediscovered via prefix hash
scheduler_config.cache_offload = ov_genai.CacheOffloadConfig(
    ssd_cache_capacity_gb=1,
    kv_cache_dir="/path/to/an/existing/directory",
    plugin=ov_genai.CacheStorePluginConfig(
        library_path="/path/to/sample_kv_cache_store_plugin.dll",
        provider_name="sample-in-memory",  # optional if a module exports just one
    ),
)
```

Or via a `tools/llm_bench` `--cb_config` JSON file:

```json
{
  "enable_prefix_caching": true,
  "cache_offload": {
    "ssd_cache_capacity_gb": 1,
    "kv_cache_dir": "/path/to/an/existing/directory",
    "plugin": {
      "library_path": "/path/to/sample_kv_cache_store_plugin.dll",
      "provider_name": "sample-in-memory"
    }
  }
}
```

Once the plugin is open, a failing read or write is only a cache miss: offload stays best-effort at run time.

### Settings the host hands to the plugin

`CacheOffloadConfig` carries the settings of an SSD-backed store. The host passes them to `open` in typed fields of
`OvKVCacheStoreOpenOptions`; `plugin.properties` reaches the plugin unchanged in `properties` and the host adds
nothing to it:

| `CacheOffloadConfig` field | Reaches the plugin as |
|---|---|
| `kv_cache_dir` | `OvKVCacheStoreOpenOptions.directory` |
| `ssd_cache_capacity_gb` | `OvKVCacheStoreOpenOptions.capacity_bytes`, in bytes (GiB * 2^30), split over the KV and the linear attention group by the host so that both reach the same number of tokens |
| `dram_cache_capacity_gb` | `OvKVCacheStoreOpenOptions.dram_capacity_bytes`, only for a provider that advertises `OV_KVCACHE_CAP_DRAM_TIER` (0 otherwise) |
| `reuse_ssd_cache` | `provider->purge(ns)`, called right after `open` when it is `false`; anything but `OV_KVCACHE_OK` fails the pipeline construction |
| `plugin.debug_log_path` | `OvKVCacheStoreOpenOptions.debug_log_path` (NULL when empty) |
| the inference device | `OvKVCacheStoreOpenOptions.device` |

With `reuse_ssd_cache` set to `false` the host calls `purge` for the opened namespace before any block I/O, so the
plugin has to purge what it holds for it; `sample-persistent-file` deletes the block files of that namespace. The
in-tree file store ignores `dram_cache_capacity_gb`.

`CacheStorePluginConfig.timeout_ms` (default 10000) bounds how long the host waits for one asynchronous batch.
A batch that is not terminal by then is cancelled through the `cancel` entry, which belongs to the async capability:
a provider that advertises `OV_KVCACHE_CAP_ASYNC_IO` without `cancel` is used through its synchronous entries.
After the cancel the host waits one more `timeout_ms`, so a plugin has to drive a cancelled token to a terminal
status (`OV_KVCACHE_ERR_CANCELLED`) within that time. A cancelled batch that ends in time is counted as a cache miss.
If the token is still pending then, the host gives up the store: every later request to it is a miss, the buffers of
the batch stay allocated for the rest of the process because the plugin may still use them, and the instance is
neither flushed nor destroyed. The host hands the plugin copies of the blocks it writes asynchronously, so a batch
never refers to memory the host frees. Synchronous entry points cannot be timed out.

## Writing your own plugin

1. Copy [`sample_kv_cache_store_plugin.c`](sample_kv_cache_store_plugin.c) as a starting point.
2. Replace the in-memory list in `store_read_blocks` / `store_write_blocks` / `store_drop_blocks`
   with real I/O against your storage, and replace the naive linked-list index with your own
   key -> location mapping. The host only ever gives you an opaque
   `OvKVCacheBlockKey{group, hash}` plus raw bytes; how you map that to your own
   files/objects/offsets is entirely up to you.
3. Keep the binding contract in the ABI header: pure C, no exceptions/STL/`ov::*` types crossing
   the boundary, no callbacks into the host, every vtable's first field is `struct_bytes` (for
   forward-compat bounds checking), and a `witness` must be a strong secondary digest - never raw
   token ids.
4. Update `OV_KVCACHE_STORE_ABI_MAJOR`/`MINOR` handling and capability bits only as documented in
   the header's `VERSIONING` section; never reuse or renumber an existing status code, capability
   bit, or vtable entry.

See the header's top-of-file comment block for the full binding contract (threading/lifetime
rules, addressing model, what is deliberately *not* part of this ABI).
