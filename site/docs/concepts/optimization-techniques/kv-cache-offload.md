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
scheduler_config.cache_offload = ov_genai.CacheOffloadConfig(
    ssd_cache_capacity_gb=16,                   # budget of the SSD tier, in GiB
    kv_cache_dir="/mnt/ssd/kv_cache",           # existing, writable directory
)

pipe = ov_genai.ContinuousBatchingPipeline(models_path, scheduler_config, "GPU")
```

```cpp
#include "openvino/genai/continuous_batching_pipeline.hpp"

ov::genai::SchedulerConfig scheduler_config;
scheduler_config.enable_prefix_caching = true;
ov::genai::CacheOffloadConfig offload;
offload.ssd_cache_capacity_gb = 16;
offload.kv_cache_dir = "/mnt/ssd/kv_cache";
scheduler_config.cache_offload = offload;

ov::genai::ContinuousBatchingPipeline pipe(models_path, scheduler_config, "GPU");
```

Offload needs `enable_prefix_caching`. It cannot be combined with `use_cache_eviction`, `use_sparse_attention`, or models whose sparse attention keeps an index cache per block; the pipeline rejects these configurations at creation.

Setting `cache_offload` enables offload. The size of the store is set with `ssd_cache_capacity_gb` (GiB, fractions allowed), and `kv_cache_dir` has to name an existing, writable directory. `dram_cache_capacity_gb` is reserved for a DRAM tier: the built-in one is not implemented yet, so it is only used by a plugin that advertises a DRAM tier and is otherwise ignored with a warning next to an SSD tier. All fields are described in [src/cpp/include/openvino/genai/cache_offload.hpp](../../../../src/cpp/include/openvino/genai/cache_offload.hpp).

## What Is Reused, and When

The default store is a set of files under `kv_cache_dir`, one directory per model, device and cache layout. Blocks of one run are reused when a block has been evicted from device memory and the same prefix comes back. They are also found by the next process that opens the same model on the same device with the same `kv_cache_dir`, unless `reuse_ssd_cache=False` purges them at start. Two settings decide how much carries over:

* `write_policy=CacheOffloadWritePolicy.WRITE_THROUGH`, which stores every block right after the forward pass that completed it instead of only when it is about to be overwritten. Without it, a block that was never evicted is not in the store when the process ends.
* A **store plugin**, set with `CacheOffloadConfig.plugin`, when the blocks should live in storage of your own; giving a `library_path` is the explicit opt-in to load it. The plugin implements the C ABI declared in [kv_cache_store_provider_abi.h](../../../../src/cpp/include/openvino/genai/c/kv_cache_store_provider_abi.h); [samples/c/kv_cache_store_plugin](../../../../samples/c/kv_cache_store_plugin/sample_kv_cache_store_plugin.c) is a reference implementation. A plugin that cannot be loaded, fails the handshake or cannot open its store makes the pipeline fail to start; there is no fallback to the default store. `reuse_ssd_cache` tells the plugin whether to keep what an earlier run left in `kv_cache_dir`.

Blocks are stored under a name derived from the model, the device and the cache layout, so a store written for another model or device is not reused. The model is told apart by a digest of its graph and weights; for a model read from a directory the weights digest is remembered under `<kv_cache_dir>/ov_kvcache_v1/.identity`, so a later start checks a few pages of the weight file instead of hashing all of it. Besides the digest, the identity covers the runtime versions, the inference precision and the device generation. A model the digest cannot cover is not reused by a later run.

For a vision-language model the blocks of image and audio embeddings are keyed by every value of their embeddings, so two different images never share a block.

```python
scheduler_config.cache_offload = ov_genai.CacheOffloadConfig(
    ssd_cache_capacity_gb=16,
    kv_cache_dir="/mnt/ssd/kv_cache",
    write_policy=ov_genai.CacheOffloadWritePolicy.WRITE_THROUGH,
    plugin=ov_genai.CacheStorePluginConfig(library_path="/path/to/libmy_kv_cache_store.so"),
)
```

A plugin is arbitrary native code that runs in the process, which is why it has to be allowed explicitly. If a plugin does not finish a batch within `CacheStorePluginConfig.timeout_ms`, the batch is cancelled and its blocks count as a cache miss.

A prefix is read back from the store in one or more rounds, so a plugin cannot tell from its own read calls how much of a request was reused. A plugin that implements the optional `on_restore_finish` entry (`OV_KVCACHE_CAP_RESTORE_STATS`, ABI 1.2) is told once per request, after the restore is decided, how many prompt tokens were already in the device cache, how many were read from its store and how many the request finally reuses, and the host then reports no reuse metrics for that store. For any other store the host logs one summary line per restore at the `INFO` log level.

A plugin that advertises `OV_KVCACHE_CAP_MULTI_GROUP` (ABI 1.3) is opened once for all cache groups of a pipeline (the KV cache and, for a hybrid model, the linear attention cache), with one budget for both, so it can keep their blocks aligned itself. A plugin that also advertises `OV_KVCACHE_CAP_PREFIX_TOKENS` is told, before the host probes or writes linear attention checkpoints, which prompt tokens each checkpoint covers. These token ids are user content and are sent only to a plugin that asks for them.

## Behavior Changes

* A model with linear attention layers is offloaded as a hybrid model without any further setting: the linear-attention state is stored beside the KV blocks under one shared budget (see below).
* A store plugin owns eviction. `write_blocks` must make room by its own policy instead of failing for lack of space, the host never asks it to drop blocks to free capacity, and any block may be gone when it is asked for again, which only costs recomputation.
* Offload is configured with the flat fields of `CacheOffloadConfig` (`ssd_cache_capacity_gb`, `kv_cache_dir`, `write_policy`, `plugin`); there is no per-tier setting.

## Hybrid Models

For models with linear attention layers (see [Hybrid Attention Models Cache](./hybrid-attention-models-cache.md)), offload covers the linear-attention state as well. The budget is shared: the host splits `ssd_cache_capacity_gb` over the KV blocks and the checkpoints so that both cover the same number of tokens, and the pipeline fails to start when the budget cannot hold one block of each.

## Related Material

* [Prefix Caching](./prefix-caching.md)
* [Continuous Batching](./continuous-batching.md)
* [KV cache store plugin sample](../../../../samples/c/kv_cache_store_plugin/sample_kv_cache_store_plugin.c)
