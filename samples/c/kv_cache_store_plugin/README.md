# KV Cache Store Plugin Sample (`sample_kv_cache_store_plugin`)

[E2E validation guide for your own plugin](E2E_VALIDATION.md)

A minimal, standalone, third-party-style implementation of the KV cache store plugin ABI
declared in
[`openvino/genai/c/kv_cache_store_provider_abi.h`](../../../src/cpp/include/openvino/genai/c/kv_cache_store_provider_abi.h).
Use it as a starting point for a real vendor storage plugin (e.g. an SSD-backed store), or to
sanity-check that your own host integration loads and drives a plugin correctly.

## What this is

By default, OpenVINO GenAI's KV cache disk offload (`SchedulerConfig::use_cache_offload`) writes
evicted blocks to a plain file (`FileKVCacheStore`). This ABI lets a third party replace that
default with its own storage backend - without linking against `openvino::genai` or any C++ code
at all - by shipping a shared library that exports one C entry symbol.

This sample implements the **full** ABI surface using nothing but a process-local, in-memory
linked list protected by a mutex. It is deliberately **not production quality** (no persistence,
no real I/O, O(n) key lookup, unbounded memory growth) - its only job is to show exactly what a
vendor plugin needs to implement, and to prove the pattern works end-to-end with a real model.

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

Point `CacheOffloadConfig` at the built library. `allow_untrusted_store_plugin` must be set
explicitly - it defaults to `false` so no third-party code is ever loaded without an explicit
opt-in:

```python
import openvino_genai as ov_genai

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.use_cache_offload = True
scheduler_config.enable_prefix_caching = True  # offloaded blocks are rediscovered via prefix hash
scheduler_config.cache_offload_config = ov_genai.CacheOffloadConfig(
    capacity_bytes=1 * 1024 * 1024 * 1024,
    buffer_slots=128,
    store_library_path="/path/to/sample_kv_cache_store_plugin.dll",
    store_provider_name="sample-in-memory",  # this sample's only provider; optional if a module exports just one
    allow_untrusted_store_plugin=True,
)
```

Or via a `tools/llm_bench` `--cb_config` JSON file:

```json
{
  "use_cache_offload": true,
  "enable_prefix_caching": true,
  "cache_offload_config": {
    "capacity_bytes": 1073741824,
    "buffer_slots": 128,
    "store_library_path": "/path/to/sample_kv_cache_store_plugin.dll",
    "store_provider_name": "sample-in-memory",
    "allow_untrusted_store_plugin": true
  }
}
```

If the library fails to load, fails the ABI handshake, or the named provider is not found, the
host logs a warning and falls back to the default `FileKVCacheStore` - offload is always
best-effort, never a hard dependency on the plugin being present.

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
