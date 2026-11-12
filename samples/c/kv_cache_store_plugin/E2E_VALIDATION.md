# KV Cache Store Plugin: End-to-End Validation Guide

[Sample README](README.md)

This guide walks you through validating your own compiled KV cache store plugin (the shared
library implementing [`kv_cache_store_provider_abi.h`](../../../src/cpp/include/openvino/genai/c/kv_cache_store_provider_abi.h))
against the two scenarios this ABI is used for in OpenVINO GenAI:

1. **Opportunistic KV cache offload** (`SchedulerConfig::use_cache_offload`) - blocks evicted during
   normal generation are written through your plugin and rediscovered later via prefix-cache hash.
2. **Reuse after a restart** (`CacheOffloadConfig::store_completed_blocks`) - every completed block is also
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
  attention, `linear_attention_capacity_bytes` is simply ignored.

## Scenario 1: Opportunistic KV cache offload

Point `CacheOffloadConfig` at your plugin and make the device cache too small to keep an earlier
prompt around, so a repeated prompt can only be served by reading its blocks back from your plugin.
[`perf_prompts.jsonl`](perf_prompts.jsonl) holds three requests: a long prompt `P0` (3110 tokens
with Qwen3.5's tokenizer), a differently-worded filler `P1` (3231 tokens) that pushes `P0`'s blocks
out of the device cache, then `P0` again as `P2`. It is run twice and the two runs are compared:

- [`perf_baseline_cb_config.json`](perf_baseline_cb_config.json): no offload, so `P2` is always a
  full recompute.
- [`perf_offload_cb_config.json`](perf_offload_cb_config.json): offload through your plugin, so `P2`
  should be restored from it. Replace `store_library_path` with your plugin's path first:

```json
{
  "num_kv_blocks": 120,
  "max_num_batched_tokens": 4096,
  "max_num_seqs": 1,
  "dynamic_split_fuse": true,
  "enable_prefix_caching": true,
  "use_cache_offload": true,
  "cache_offload_config": {
    "capacity_bytes": 1073741824,
    "linear_attention_capacity_bytes": 4294967296,
    "buffer_slots": 256,
    "store_library_path": "/path/to/your_plugin.dll",
    "store_provider_name": "your-provider-name",
    "allow_untrusted_store_plugin": true
  }
}
```

- `num_kv_blocks: 120` holds one of these prompts but not two, which is what forces `P0` out.
  **On GPU use `220`** in both configs: GPU blocks hold fewer tokens, and 120 cannot even hold one
  prompt (the request fails with `Request 0 was dropped by the scheduler because it did not fit in
  the available cache budget`).
- `capacity_bytes` sizes the KV store and `linear_attention_capacity_bytes` sizes the
  linear-attention store. **Both must be set for a hybrid model**: without the second one only KV
  blocks are offloaded, a linear-attention state is never found again, and the repeated prompt is
  recomputed (still correct, just no speedup). For Qwen3.5-9B a linear-attention block is about
  51 MB, so `4294967296` gives 80 of them.

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
    --cb_config ../../samples/c/kv_cache_store_plugin/perf_offload_cb_config.json  # edit store_library_path first
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
| CPU | 120 | 76874.31 ms | 5562.59 ms | **~13.8x** | identical (`e067b6f4...`) |
| GPU | 220 | 2636.81 ms | 777.23 ms | **~3.4x** | identical (`e067b6f4...`) |

The CPU figures are a single run; the GPU figures are the mean of two baseline/offload pairs run
back to back (baseline `P2` 2703.94 / 2569.68 ms, offload `P2` 772.60 / 781.85 ms).

The absolute gain is far larger on CPU because CPU prefill is slower, so there is more recompute to
avoid. Writing the evicted blocks is not free, but on this setup it is small: GPU `P0` / `P1` took
2711 / 2804 ms and 2916 / 2792 ms with offload against 2927 / 2774 ms and 2725 / 2655 ms without, and
on CPU `P1` took 80.7 s against 84.4 s - differences of a few percent at most, comparable to the
run-to-run spread. Longer prompts widen the gap and very short
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
`cache_offload_config.store_completed_blocks: true` (off by default) every block is also stored as soon as the
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

Real numbers with the sample's `sample-persistent-file` provider (single runs; the prompt is the sample sentence
repeated, 300 times for Qwen3-8B and 310 times for Qwen3.5-9B, `--num-kv-blocks` 600 and 400):

| Model | Device | TTFT baseline | TTFT `write` (chunk store on) | TTFT `read` (new process) | Blocks read back | Generated text |
|---|---|---|---|---|---|---|
| Qwen3-8B | CPU | 55807 ms | 53925 ms | 6848 ms (**~8.1x**) | 94 KV | identical |
| Qwen3.5-9B | GPU | 3644 ms | 3797 ms | 1130 ms (**~3.2x**) | 194 KV + 1 linear-attention | identical |

Turning it on did not slow the `write` run measurably (the difference is within run-to-run spread). The copy of each
finished block runs on the pipeline thread, so a long prefill pays for copying its blocks right after the pass
(about 0.2 s of copying for the 94 blocks above on CPU and 0.08 s for the 194 blocks on GPU); the writing itself
(1.1 s and 1.7 s) happens on the writer thread. Blocks that do not fit into the writer's `buffer_slots` queue are
skipped and only cost recomputation later.

`store_completed_blocks` also works together with opportunistic offload (Qwen3.5-9B, GPU, the in-memory provider):

- `llm_bench`, `num_kv_blocks` 220, `"store_completed_blocks": true` in `cache_offload_config`:
  P0/P1/P2 give the same output hashes as without the flag, and P2 (the P0 prompt again after P1 evicted it) takes
  780 ms against 3465 ms for P0 (194 blocks warmed from the store, 397 blocks stored as `published` plus 4 as
  `overwritten`).

### What the host expects from your plugin's stores

The host opens **one store instance per group** and passes the group as the `OvKVCacheNamespace`
(`kv` for KV blocks and `la` for linear-attention blocks). The group string is an opaque byte string: when a plugin is
configured the host appends a digest of the model and cache layout (for example `kv:<32 hex digits>`),
so that stores shared by different models never serve each other's blocks. Treat the namespace as a
key, do not parse it. `OvKVCacheGroupInfo` (ABI 1.1) tells you each group's `block_bytes` and
`block_tokens` (for `la`: the checkpoint interval). It sizes each store against its own
capacity and checks that the store's `stats()` stays within it. Your plugin must therefore keep
the namespaces apart: if it puts KV and linear-attention blocks in one store, `stats()` reports
more blocks than the smaller store's capacity and the run fails with `Check 'stats.blocks <=
m_capacity_blocks' failed ... ABI store reported more used blocks than its capacity`. Also make
sure `capacity_bytes` is large enough for your plugin's real per-block storage overhead, for the
same reason.

#### Plugin settings and plugin-side eviction

- `CacheOffloadConfig.store_properties` (a string-to-string map; `"store_properties": {...}` in the `llm_bench`
  cache offload JSON) reaches the plugin in `OvKVCacheStoreOpenOptions.properties`, after the entries the host
  always adds: `block_size_bytes`, `device`, `dram_cache_capacity_gb` and `reuse_ssd_cache`, plus `debug_log_path`
  when `CacheOffloadConfig.debug_log_path` is set. The host ignores attempts to override those. The host
  does not interpret the rest. The sample plugin logs every extra property it receives when it opens a store.
- `CacheOffloadConfig.kv_cache_dir` reaches the plugin as `OvKVCacheStoreOpenOptions.directory`, and
  `capacity_bytes` (or `ssd_cache_capacity_gb` converted to bytes; set only one of them) as
  `OvKVCacheStoreOpenOptions.capacity_bytes`. `dram_cache_capacity_gb` is a decimal string, `0` meaning no DRAM budget.
- `reuse_ssd_cache` is `true` by default. With `false` the plugin has to purge what it holds for the opened namespace
  (one per model and cache kind) before serving requests; `sample-persistent-file` deletes that namespace's block
  files, so a restart finds nothing.
- A provider that advertises `OV_KVCACHE_CAP_SELF_EVICT` owns eviction. The host then does not drop blocks to free
  capacity and does not count blocks against `capacity_bytes`; it asks the plugin (`has_blocks`, reads) whenever
  it needs to know whether a block is still there and treats a block that disappeared as a miss, which only
  costs recomputation.
  The sample's `sample-self-evict-memory` provider shows the contract: it keeps `capacity_bytes / block_bytes`
  blocks and evicts the oldest one when a write finds it full.
- The host writes the blocks queued by the writer thread in one `write_blocks` call (at most `buffer_slots` per
  call), after probing each block's chain from block 0.

Checked on Qwen3.5-9B / GPU with `sample-self-evict-memory` (and `"store_properties": {"tier": "ssd-test",
"policy": "lru"}`, both of which the plugin logged on open): with room for every block the `llm_bench` P2 run
takes 925 ms against 3375 ms for P0 with identical output hashes; with a store of 250 blocks the plugin evicts
the oldest blocks, which are the start of P0's prefix, so nothing is warmed for P2 (3069 ms), the output is
still identical and the host reports no failed store. Which block a plugin evicts first decides how much of a prefix
survives, which is why the choice belongs to the plugin.

## Scenario 2: Reuse after a restart

There is no save or restore call. With `cache_offload_config.store_completed_blocks` on, every completed KV block (and,
for a hybrid model, every linear-attention checkpoint) reaches your plugin right after the forward pass that
completed it. A pipeline created later against the same persistent stores finds those blocks like any other
prefix-cache hit: `add_request()` with the history so far (the prompt plus the tokens generated earlier) reads them
back through your plugin and recomputes only what is missing. Consequences for your plugin and configuration:

- The pipeline must be created with `enable_prefix_caching`, `use_cache_offload` and `store_completed_blocks`, with
  your plugin in `SchedulerConfig.cache_offload_config`. The new pipeline has to open the same stores: the same
  plugin and properties, and the same model, device and cache layout, because the namespace carries a digest of
  them and another model never sees these blocks.
- The plugin must keep blocks across the process. A plugin that does not, such as the sample's `sample-in-memory`
  provider, still passes a restart inside one process but loses everything when the process exits. The new pipeline
  must keep `reuse_ssd_cache` at its default `true`; with `false` the plugin purges the stores on open.
- Only completed blocks are stored. The unfinished last block (fewer tokens than one block) is recomputed, and for
  a hybrid model so is everything after the last linear-attention checkpoint, which is one checkpoint interval at
  most (see Scenario 1 for its size).
- Nothing guarantees a block is still there: the store may be full, a block may not have fit the writer's
  `buffer_slots` queue, or the plugin may evict it. The host then recomputes that part and the output stays the same;
  there is no error. Blocks still queued when a process is killed are lost, while a pipeline that ends normally
  writes every queued block before it goes away.
- Stored blocks count against `capacity_bytes` / `linear_attention_capacity_bytes` like any other offloaded block.
  For Qwen3.5-9B a linear-attention checkpoint is about 51 MB, so size the linear-attention store for the number of
  checkpoints you want to keep.

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
  with `OV_KVCACHE_WAIT_INFINITE` until the status is not `PENDING`, so no token is left open and `cancel` is
  never needed. It does not overlap submissions: each batch is waited on before the next one starts. Without the
  bit the host uses the synchronous `read_blocks`/`write_blocks`.
- Reads of a prompt's reusable prefix are one `read_blocks` batch (async or not), not one call per block.
- `OV_KVCACHE_CAP_SEMANTICS_FINGERPRINT`: the fingerprint `open` returns is logged at `OPENVINO_LOG_LEVEL=3` as
  `semantics_fingerprint=` and kept on the store; the host does not compare or persist it yet.
- A provider must advertise `OV_KVCACHE_CAP_KEY_IO` and `OV_KVCACHE_CAP_EXISTS`, or the host logs a warning and
  falls back to the default file store.

Both sample providers advertise async I/O and a fingerprint. Their async entries run the batch inside the submit
call and `wait` only hands back its status, which is enough to show the call sequence; a real plugin would hand
the batch to its I/O threads and block in `wait`. `register_host_memory`
(`OV_KVCACHE_CAP_REGISTER_HOST_MEMORY`) is still not called by this host.
