# KV Cache Store Plugin: End-to-End Validation Guide

[Sample README](README.md)

This guide walks you through validating your own compiled KV cache store plugin (the shared
library implementing [`kv_cache_store_provider_abi.h`](../../../src/cpp/include/openvino/genai/c/kv_cache_store_provider_abi.h))
against the two scenarios this ABI is used for in OpenVINO GenAI:

1. **Opportunistic KV cache offload** (`SchedulerConfig::cache_offload`) - blocks evicted during
   normal generation are written through your plugin and rediscovered later via prefix-cache hash.
2. **Reuse after a restart** (`CacheOffloadWritePolicy::WRITE_THROUGH`) - every completed block is also
   written through your plugin right after the forward pass that completed it, so a new pipeline or process that opens
   the same persistent store finds the blocks of an earlier conversation as a prefix-cache hit and continues it
   without re-running prefill. No extra API is involved: the new pipeline just gets the history with `add_request()`.

Both scenarios below were run against the standalone sample plugin in this directory
([`sample_kv_cache_store_plugin.c`](sample_kv_cache_store_plugin.c)) with **Qwen3.5-9B** (a hybrid
KV + linear-attention model): Scenario 1 on CPU **and** GPU, Scenario 2 on GPU (and with Qwen3-8B on
CPU). Substitute your own plugin's path and provider name. GPU's paged-attention blocks hold fewer tokens than CPU's, so the
same `num_kv_blocks` lands differently - the GPU value used here is called out in Scenario 1.

## Prerequisites

- An OpenVINO GenAI build or package with Python bindings (`openvino_genai`).
- Your compiled plugin library (`.dll`/`.so`), built against `kv_cache_store_provider_abi.h`.
- A text-generation model in OpenVINO IR format. This guide uses Qwen3.5-9B, a hybrid model, so
  both the KV and the linear-attention payloads go through your plugin. For a model without linear
  attention, the budget is simply used by the KV cache alone.

## Scenario 1: Opportunistic KV cache offload

Point `CacheOffloadConfig.plugin` at your plugin and make the device cache too small to keep an earlier
prompt around, so a repeated prompt can only be served by reading its blocks back from your plugin.
[`perf_prompts.jsonl`](perf_prompts.jsonl) holds three requests: a long prompt `P0` (3110 tokens
with Qwen3.5's tokenizer), a differently-worded filler `P1` (3231 tokens) that pushes `P0`'s blocks
out of the device cache, then `P0` again as `P2`. It is run twice and the two runs are compared:

- [`perf_baseline_cb_config.json`](perf_baseline_cb_config.json): no offload, so `P2` is always a
  full recompute.
- [`perf_offload_cb_config.json`](perf_offload_cb_config.json): offload through your plugin, so `P2`
  should be restored from it. Replace `plugin.library_path` with your plugin's path first and `kv_cache_dir` with an existing directory:

```json
{
  "num_kv_blocks": 120,
  "max_num_batched_tokens": 4096,
  "max_num_seqs": 1,
  "dynamic_split_fuse": true,
  "enable_prefix_caching": true,
  "cache_offload": {
    "ssd_cache_capacity_gb": 5,
    "kv_cache_dir": "/path/to/an/existing/directory",
    "plugin": {
      "library_path": "/path/to/your_plugin.dll",
      "provider_name": "your-provider-name"
    }
  }
}
```

- `num_kv_blocks: 120` holds one of these prompts but not two, which is what forces `P0` out.
  **On GPU use `220`** in both configs: GPU blocks hold fewer tokens, and 120 cannot even hold one
  prompt (the request fails with `Request 0 was dropped by the scheduler because it did not fit in
  the available cache budget`).
- `ssd_cache_capacity_gb` is one budget for all cache groups (GiB, fractions allowed). For a hybrid model the host
  splits it over the KV and the linear-attention group so that both cover the same number of tokens; a model with
  linear attention layers always offloads them too, since without their states a repeated prompt is recomputed (still
  correct, just no speedup). For Qwen3.5-9B a linear-attention block is about 51 MB, so 5 GiB hold the blocks of a
  few thousand tokens of both groups. A budget that cannot hold one block of every group makes the pipeline fail to
  start.

Run both with `tools/llm_bench/benchmark.py`. Qwen3.5 is natively multimodal, so use
`--task visual_text_gen`; `--task text_gen` feeds pre-tokenized input and fails on this model with
`Tensor was not initialized`:

```sh
cd tools/llm_bench
# Baseline: no offload, P2 always recomputed from scratch.
python benchmark.py -m <model_dir> -d CPU --task visual_text_gen \
    -pf ../../samples/c/kv_cache_store_plugin/perf_prompts.jsonl -pi 0 1 2 -n 0 -ic 1 \
    --disable_prompt_permutation \
    --cb_config ../../samples/c/kv_cache_store_plugin/perf_baseline_cb_config.json

# Offload: P2 should be restored from your plugin instead of recomputed.
python benchmark.py -m <model_dir> -d CPU --task visual_text_gen \
    -pf ../../samples/c/kv_cache_store_plugin/perf_prompts.jsonl -pi 0 1 2 -n 0 -ic 1 \
    --disable_prompt_permutation \
    --cb_config ../../samples/c/kv_cache_store_plugin/perf_offload_cb_config.json  # edit plugin.library_path and kv_cache_dir first
```

**What a pass looks like:**

- `P2`'s `Result MD5` equals `P0`'s in **both** runs. `P2` is the same prompt as `P0`, so this shows
  the blocks read back from your plugin reproduce the original result.
- In the offload run `P2`'s first token latency is far below `P0`'s and below the baseline's `P2`.
- No `[ERROR]` lines about your plugin. A plugin that fails to load, handshake or open is logged
  as a warning and the run silently falls back to the in-tree default store, so if the offload run
  shows no speedup, check the log for `Falling back to the default file store` first.

### Results

**Real numbers from this exact pair of commands against the sample plugin** (Qwen3.5-9B,
`-ic 1`, so the time shown is first token latency):

| Device | `num_kv_blocks` | Baseline `P2` TTFT | Offload (via plugin) `P2` TTFT | Speedup | `P0` / `P2` MD5 |
|---|---|---|---|---|---|
| CPU | 120 | 154682.83 ms | 10853.54 ms | **~14.3x** | identical (`e067b6f4...`) |
| GPU | 220 | 2285.72 ms | 838.94 ms | **~2.7x** | identical (`e067b6f4...`) |

The CPU figures are a single run; the GPU figures are the mean of two baseline/offload pairs run
back to back (baseline `P2` 2424.85 / 2146.59 ms, offload `P2` 768.47 / 909.40 ms).

The absolute gain is far larger on CPU because CPU prefill is slower, so there is more recompute to
avoid. Writing the evicted blocks is not free, but on this setup it is small: GPU `P0` / `P1` took
2454 / 2721 ms and 2722 / 2925 ms with offload against 2584 / 2638 ms and 3431 / 2309 ms without, and
on CPU `P1` took 162.9 s against 160.5 s - differences that are within the run-to-run spread (the baseline
GPU `P0` alone ranged from 2584 to 3431 ms between two runs). Longer prompts widen the gap and very short
ones can erase it, so benchmark with prompt lengths representative of your workload rather than
relying on these numbers.

**A hybrid model only reuses a prefix up to its last linear-attention checkpoint.** The recurrent
state is checkpointed every `cache_interval` tokens (about 2880 on CPU and 2592 on GPU with the
default sizing, printed as `cache_interval=` in the `[KV_TRACE]` log at `OPENVINO_LOG_LEVEL=3`),
and a repeated prompt cannot resume from a checkpoint that already includes its final token. A
prompt shorter than one interval therefore gets no reuse at all - it is recomputed, and the result
is still correct. That is why this guide's prompts are over 3000 tokens; with short prompts offload
through your plugin will show no speedup on a hybrid model.

### Storing blocks right after each forward pass (chunk store)

By default a block reaches your plugin only when the device cache is about to overwrite it. With
`write_policy: WRITE_THROUGH` (the default, `ON_EVICT`, only writes evicted blocks) every block is also stored as soon as the
forward pass that completed it has finished: the pipeline thread copies the block out and the offload writer
thread writes it, so generation does not wait for your plugin. The block stays in the device cache as before.
This makes blocks reusable from the store without having been evicted first, for example by another process
that opens the same persistent store. Blocks restored from the store or the device cache are not written again,
and a block stored this way is not stored a second time when it is later overwritten.

[`verify_chunk_store.py`](verify_chunk_store.py) checks it with three separate processes against one store
directory (`baseline` without offload, `write`, then `read`); the device cache is large enough that nothing is
evicted, so everything the `read` process finds was written by the chunk store:

```sh
cd samples/c/kv_cache_store_plugin
python verify_chunk_store.py -m <model_dir> -d CPU --plugin /path/to/sample_kv_cache_store_plugin.dll \
    --store-dir /tmp/kv_chunk --phase baseline
python verify_chunk_store.py -m <model_dir> -d CPU --plugin /path/to/sample_kv_cache_store_plugin.dll \
    --store-dir /tmp/kv_chunk --phase write
python verify_chunk_store.py -m <model_dir> -d CPU --plugin /path/to/sample_kv_cache_store_plugin.dll \
    --store-dir /tmp/kv_chunk --phase read
```

Real numbers with the sample's `sample-persistent-file` provider (Qwen3-8B: a single run; Qwen3.5-9B: the median of
five runs; the prompt is the sample sentence repeated, 300 times for Qwen3-8B and 310 times for Qwen3.5-9B,
`--num-kv-blocks` 600 and 400):

| Model | Device | TTFT baseline | TTFT `write` (chunk store on) | TTFT `read` (new process) | Blocks read back | Generated text |
|---|---|---|---|---|---|---|
| Qwen3-8B | CPU | 84724 ms | 81584 ms | 9463 ms (**~9.0x**) | 94 KV | identical |
| Qwen3.5-9B | GPU | 2842 ms | 3445 ms | 1448 ms (**~2.0x**) | 194 KV + 1 linear-attention | identical |

The timings vary a lot from run to run on the GPU: over the five runs the baseline ranged from 2392 to 3435 ms,
`read` from 938 to 1968 ms, and the speedup from 1.5x to 3.7x. On CPU the `write` run was not measurably slower than
the baseline. On the GPU it was 19% to 31% slower in every pair of runs (about 0.5 to 1 s for this prompt), so
turning `WRITE_THROUGH` on has a visible cost on the first request of a long prefill there. The copy of each
finished block runs on the pipeline thread, so a long prefill pays for copying its blocks right after the pass
(0.23 s of copying for the 397 KV blocks of the `llm_bench` run below on GPU); the writing itself happens on the
writer thread. Blocks that do not fit into the writer's queue are skipped and only cost recomputation
later.

`WRITE_THROUGH` also works together with opportunistic offload (Qwen3.5-9B, GPU, the in-memory provider):

- `llm_bench`, `num_kv_blocks` 220, `"write_policy": "WRITE_THROUGH"` in `cache_offload`:
  P0/P1/P2 give the same output hashes as without the flag, and P2 (the P0 prompt again after P1 evicted it) takes
  773 ms against 3078 ms for P0 (194 KV blocks and 1 linear-attention block warmed from the store; 397 KV blocks
  and 4 linear-attention blocks stored over the run).

### What the host expects from your plugin's stores

The host opens **one store instance per group** and passes the group as the `OvKVCacheNamespace`
(`kv` for KV blocks and `la` for linear-attention blocks) unless the provider advertises
`OV_KVCACHE_CAP_MULTI_GROUP` (ABI 1.3, see below). The group string is an opaque byte string: when a plugin is
configured the host appends a digest of the model and cache layout (for example `kv:<32 hex digits>`),
so that stores shared by different models never serve each other's blocks. Treat the namespace as a
key, do not parse it. `OvKVCacheGroupInfo` (ABI 1.1) tells you each group's `block_bytes` and
`block_tokens` (for `la`: the checkpoint interval). It sizes each store against its own
capacity and checks that the store's `stats()` stays within it. Your plugin must therefore keep
the namespaces apart: if it puts KV and linear-attention blocks in one store, `stats()` reports
more blocks than the smaller store's capacity and the run fails with `Check 'stats.blocks <=
m_capacity_blocks' failed ... ABI store reported more used blocks than its capacity`. Also make
sure the budget (`ssd_cache_capacity_gb`) is large enough for your plugin's real per-block storage overhead, for the
same reason.

#### Plugin settings and plugin-side eviction

- `CacheStorePluginConfig.properties` (a string-to-string map; `"properties": {...}` under `plugin` in the `llm_bench`
  cache offload JSON) reaches the plugin unchanged in `OvKVCacheStoreOpenOptions.properties`; the host adds nothing
  and does not interpret them. The settings of the host are typed fields instead: `device`, `debug_log_path` (when
  `CacheStorePluginConfig.debug_log_path` is set) and `dram_capacity_bytes` (only for a provider with
  `OV_KVCACHE_CAP_DRAM_TIER`). The sample plugin logs every property it receives when it opens a store.
- `CacheOffloadConfig.kv_cache_dir` reaches the plugin as `OvKVCacheStoreOpenOptions.directory`, and
  `ssd_cache_capacity_gb` converted to bytes as `OvKVCacheStoreOpenOptions.capacity_bytes`. `dram_cache_capacity_gb` reaches a provider with `OV_KVCACHE_CAP_DRAM_TIER` as `dram_capacity_bytes`.
- `reuse_ssd_cache` is `true` by default. With `false` the host calls `provider->purge(ns)` for the opened namespace
  (one per model and cache kind) right after the open, and a failing purge fails the construction; `sample-persistent-file` deletes that namespace's block
  files, so a restart finds nothing.
- Every provider owns eviction. The host never drops blocks to free capacity and does not count blocks against the
  budget; `write_blocks` must not fail with `OV_KVCACHE_ERR_CAPACITY` for lack of space but make room by the
  plugin's own policy. The host asks the plugin (`has_blocks`, reads) whenever it needs to know whether a block is
  still there and treats a block that disappeared as a miss, which only costs recomputation.
  The sample's `sample-self-evict-memory` provider shows the contract: it keeps `capacity_bytes` bytes
  of blocks and evicts the oldest ones when a write finds it full.
- The host writes the blocks queued by the writer thread in one `write_blocks` call (a bounded number per
  call), after probing each block's chain from block 0.
- A prefix is restored in one or more `read_blocks` rounds: the host holds at most 256 MiB of read-ahead blocks on
  the host at a time, copies them to the device cache and reads the next round. A long prefix is therefore read in
  several calls, and a plugin cannot tell from its own calls how much of a request was reused.
- A provider that advertises `OV_KVCACHE_CAP_RESTORE_STATS` (ABI 1.2) gets one `on_restore_finish` call per request
  once its prefix restore is decided, also when nothing was found, with `OvKVCacheRestoreStats`: the prompt length,
  the tokens that were already in the device cache, the tokens read from this store (all rounds together) and the
  prefix the request finally reuses. The host calls it with no host lock held for each store of the request (one per
  group, so a hybrid model's `kv` and `la` stores each get a call; for `la` the counts are checkpoint positions),
  and logs no reuse metrics of its own for a store that takes it. A store without it gets one summary line in the
  host log instead. A plugin that prints its own reuse metrics should do it here and not in `read_blocks`. The
  sample plugin prints one line per call to stderr. From ABI 1.3 the report names its group (`group`), which is how
  the reports of the `kv` and the `la` store are told apart when one instance serves both.
- A provider that advertises `OV_KVCACHE_CAP_MULTI_GROUP` (ABI 1.3) is opened
  **once** for the whole pipeline: `OvKVCacheStoreOpenOptions.groups` lists `kv` and, for a hybrid model, `la` with
  their block sizes, `OvKVCacheBlockKey::group` tells them apart, the namespace identifies the model and cache layout
  only, and `capacity_bytes` is the budget of both groups together (`ssd_cache_capacity_gb`). This lets a plugin keep the linear-attention checkpoints and the KV blocks of a
  request aligned itself. Without the bit the host opens one instance per group, as before.
- A provider that advertises `OV_KVCACHE_CAP_PREFIX_TOKENS` (ABI 1.3) gets `announce_prefix_tokens` before the host
  probes or writes linear-attention checkpoints: for each checkpoint whose tokens the host still knows, the whole
  prompt prefix `[0, token_count)` as token ids, so the plugin can place a checkpoint at `token_count` among the KV
  blocks of `block_tokens` tokens. Token ids are user content, so they are sent only to a plugin that asks for them,
  only for the `la` group, and the witness stays a digest. The sample `sample-self-evict-memory` provider advertises
  both capabilities and prints what it is told.

Checked on Qwen3.5-9B / GPU with `sample-self-evict-memory` (and `"properties": {"tier": "ssd-test",
"policy": "lru"}` under `plugin`, both of which the plugin logged on open) and `"write_policy": "WRITE_THROUGH"`: with room for
every block the `llm_bench` P2 run takes 646 ms against 2446 ms for P0 with identical output hashes; with a store
of 250 blocks (a budget of 41472000 bytes for the KV group, 165888 bytes per KV block; with the pooled budget of ABI 1.3 set `ssd_cache_capacity_gb` so that both groups together stay that small) the plugin evicts the oldest blocks, which
are the start of P0's prefix, so nothing is warmed for P2 (2369 ms), the output is still identical and the host
reports no failed store. With `ON_EVICT` only the blocks the device cache overwrites reach the
store, about 195 of them, so the same 250-block store still holds all of P0's prefix and P2 is warmed (648 ms).
Which block a plugin evicts first decides how much of a prefix survives, which is why the choice belongs to the
plugin.

## Scenario 2: Reuse after a restart

There is no save or restore call. With `write_policy` `WRITE_THROUGH`, every completed KV block (and,
for a hybrid model, every linear-attention checkpoint) reaches your plugin right after the forward pass that
completed it. A pipeline created later against the same persistent stores finds those blocks like any other
prefix-cache hit: `add_request()` with the history so far (the prompt plus the tokens generated earlier) reads them
back through your plugin and recomputes only what is missing. Consequences for your plugin and configuration:

- The pipeline must be created with `enable_prefix_caching` and `SchedulerConfig.cache_offload` with the `WRITE_THROUGH`
  write policy and your plugin. The new pipeline has to open the same stores: the same
  plugin and properties, and the same model, device and cache layout, because the namespace carries a digest of
  them and another model never sees these blocks.
- The plugin must keep blocks across the process. A plugin that does not, such as the sample's `sample-in-memory`
  provider, still passes a restart inside one process but loses everything when the process exits. The new pipeline
  must keep `reuse_ssd_cache` at its default `true`; with `false` the plugin purges the stores on open.
- Only completed blocks are stored. The unfinished last block (fewer tokens than one block) is recomputed, and for
  a hybrid model so is everything after the last linear-attention checkpoint, which is one checkpoint interval at
  most (see Scenario 1 for its size).
- Nothing guarantees a block is still there: the store may be full, a block may not have fit the writer's
  queue, or the plugin may evict it. The host then recomputes that part and the output stays the same;
  there is no error. Blocks still queued when a process is killed are lost, while a pipeline that ends normally
  writes every queued block before it goes away.
- Stored blocks count against the budget (`ssd_cache_capacity_gb`) like any other offloaded block.
  For Qwen3.5-9B a linear-attention checkpoint is about 51 MB, so size the budget for the number of checkpoints you
  want to keep.

Validating Scenario 1 does not guarantee Scenario 2 works (it also needs durable writes that survive the process), so
run [`verify_chunk_store.py`](verify_chunk_store.py) from the chunk store section above against your plugin: its
`write` and `read` phases are separate processes against one store directory, and `read` must print the same text as
`baseline` with a clearly lower TTFT. The numbers in that section (about 8.1x on CPU for Qwen3-8B, about 3.2x on GPU
for Qwen3.5-9B) were measured this way.

## Known limitations of the sample plugin (do not assume your plugin needs to share these)

The sample plugin in this directory is intentionally a minimal reference. Its default provider,
`sample-in-memory`, has **no real persistence**: its data lives in process memory and is gone once the process
exits, so a restart through it can only be checked with a second `ContinuousBatchingPipeline` instance in the *same*
process. That validates the ABI call sequence and data correctness, but not that your plugin survives a real process
restart - that depends on your own storage backend.

The second provider, `sample-persistent-file`, keeps every block as a file under `CacheOffloadConfig.kv_cache_dir` and is
what the real multi-process check uses, see the three `verify_chunk_store.py` phases above. The store then holds one
`ns_*` directory per group, `kv` and `la`. With your own plugin, run the same commands against your storage.

The host uses the optional capabilities your provider advertises:

- `OV_KVCACHE_CAP_ASYNC_IO` (`read_blocks_async`/`write_blocks_async`/`wait`): when it is set and the vtable
  carries those entries, every read and write goes through them. The host submits one batch, then calls `wait`
  with `CacheStorePluginConfig.timeout_ms` (default 10000 ms). If the status is still `PENDING` after that, the host
  calls `cancel` and waits once more for the same time, and the batch counts as a cache miss. A plugin that does not
  reach a terminal status within that second wait is given up: later requests to its store are misses and the store is
  neither flushed nor destroyed. A provider without a `cancel` entry is used through the synchronous entries. It does
  not overlap submissions: each batch is waited on before the next one starts. Without the bit the host uses the
  synchronous `read_blocks`/`write_blocks`, which cannot be timed out.
- Reads of a prompt's reusable prefix are one `read_blocks` batch (async or not), not one call per block.
- `OV_KVCACHE_CAP_SEMANTICS_FINGERPRINT`: the fingerprint `open` returns is logged at `OPENVINO_LOG_LEVEL=3` as
  `semantics_fingerprint=` and kept on the store; the host does not compare or persist it yet.
- A provider must advertise `OV_KVCACHE_CAP_KEY_IO` and `OV_KVCACHE_CAP_EXISTS`, or the host logs a warning and
  falls back to the default file store.

Both sample providers advertise async I/O and a fingerprint. Their async entries run the batch inside the submit
call and `wait` only hands back its status, which is enough to show the call sequence; a real plugin would hand
the batch to its I/O threads and block in `wait`. `register_host_memory`
(`OV_KVCACHE_CAP_REGISTER_HOST_MEMORY`) is still not called by this host.
