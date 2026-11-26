---
sidebar_position: 6
---

# KV Cache Offload

## Overview

With [prefix caching](./prefix-caching.md), a block of the paged KV cache can be reused by a later request that starts with the same tokens, but only while the block is still in device memory. **KV cache offload** adds a second level behind it: a block that is about to be overwritten is first copied to a store on an SSD, and a later request that shares the prefix reads it back instead of recomputing it. It is available in `ContinuousBatchingPipeline`, and in `LLMPipeline` and `VLMPipeline` when they run on the PagedAttention backend (see [Continuous Batching](./continuous-batching.md)).

Offload is best effort. A store that fails, runs out of space, or does not answer in time only costs recomputation of the affected blocks; it never fails a request.

## Enabling Offload

Offload is configured through `SchedulerConfig`:

```python
import openvino_genai as ov_genai

scheduler_config = ov_genai.SchedulerConfig()
scheduler_config.enable_prefix_caching = True   # offloaded blocks are found again through the prefix hash
scheduler_config.use_cache_offload = True
scheduler_config.cache_offload_config = ov_genai.CacheOffloadConfig(
    kv_cache_dir="/mnt/ssd/kv_cache",           # existing directory; the system temporary directory when empty
    capacity_bytes=16 * 1024 * 1024 * 1024,
)

pipe = ov_genai.ContinuousBatchingPipeline(models_path, scheduler_config, "GPU")
```

```cpp
#include "openvino/genai/continuous_batching_pipeline.hpp"

ov::genai::SchedulerConfig scheduler_config;
scheduler_config.enable_prefix_caching = true;
scheduler_config.use_cache_offload = true;
scheduler_config.cache_offload_config.kv_cache_dir = "/mnt/ssd/kv_cache";
scheduler_config.cache_offload_config.capacity_bytes = 16ull * 1024 * 1024 * 1024;

ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, "GPU");
```

Offload needs `enable_prefix_caching`. It cannot be combined with `use_cache_eviction`, `use_sparse_attention`, or models whose sparse attention keeps an index cache per block; the pipeline rejects these configurations at creation.

The size of the store is set with `capacity_bytes`, or with `ssd_cache_capacity_gb` instead of it. All fields are described in [src/cpp/include/openvino/genai/cache_offload.hpp](../../../../src/cpp/include/openvino/genai/cache_offload.hpp).

## What Is Reused, and When

The default store is a file that belongs to one pipeline instance: it is created with the pipeline and removed with it. Out of the box, offload therefore only helps within one run, when a block has been evicted from device memory and the same prefix comes back. It does not carry over to the next process.

Reuse across processes, for example after a restart, needs two more settings:

* A **persistent store plugin** that keeps its blocks beyond the process, set with `store_library_path` and `allow_untrusted_store_plugin=True`. The plugin implements the C ABI declared in [kv_cache_store_provider_abi.h](../../../../src/cpp/include/openvino/genai/c/kv_cache_store_provider_abi.h); [samples/c/kv_cache_store_plugin](../../../../samples/c/kv_cache_store_plugin/sample_kv_cache_store_plugin.c) is a reference implementation. A plugin that cannot be loaded is reported in the log, and the pipeline falls back to the default store. `reuse_ssd_cache` tells the plugin whether to keep what an earlier run left in `kv_cache_dir`.
* `store_completed_blocks=True`, which stores every block right after the forward pass that completed it instead of only when it is about to be overwritten. Without it, a block that was never evicted is not in the store when the process ends.

Blocks are stored under a name derived from the model, the device and the cache layout, so a store written for another model or device is not reused.

```python
scheduler_config.cache_offload_config = ov_genai.CacheOffloadConfig(
    kv_cache_dir="/mnt/ssd/kv_cache",
    capacity_bytes=16 * 1024 * 1024 * 1024,
    store_library_path="/path/to/libmy_kv_cache_store.so",
    allow_untrusted_store_plugin=True,
    store_completed_blocks=True,
)
```

A plugin is arbitrary native code that runs in the process, which is why it has to be allowed explicitly. If a plugin does not finish a batch within `store_timeout_ms`, the batch is cancelled and its blocks count as a cache miss.

A prefix is read back from the store in one or more rounds, so a plugin cannot tell from its own read calls how much of a request was reused. A plugin that implements the optional `on_restore_finish` entry (`OV_KVCACHE_CAP_RESTORE_STATS`, ABI 1.2) is told once per request, after the restore is decided, how many prompt tokens were already in the device cache, how many were read from its store and how many the request finally reuses, and the host then reports no reuse metrics for that store. For any other store the host logs one summary line per restore at the `INFO` log level.

A plugin that advertises `OV_KVCACHE_CAP_MULTI_GROUP` and `OV_KVCACHE_CAP_SELF_EVICT` (ABI 1.3) is opened once for all cache groups of a pipeline (the KV cache and, for a hybrid model, the linear attention cache), with one budget for both, so it can keep their blocks aligned itself. A plugin that also advertises `OV_KVCACHE_CAP_PREFIX_TOKENS` is told, before the host probes or writes linear attention checkpoints, which prompt tokens each checkpoint covers. These token ids are user content and are sent only to a plugin that asks for them.

## Hybrid Models

For models with linear attention layers (see [Hybrid Attention Models Cache](./hybrid-attention-models-cache.md)), offload covers the linear-attention state as well, but only when `linear_attention_capacity_bytes` is greater than 0. It sizes a separate store for the checkpoints of that state. With the default 0, only the KV cache of such a model is offloaded, and a prefix can be reused only as far as the checkpoints that are available.

## Related Material

* [Prefix Caching](./prefix-caching.md)
* [Continuous Batching](./continuous-batching.md)
* [KV cache store plugin sample](../../../../samples/c/kv_cache_store_plugin/sample_kv_cache_store_plugin.c)
