# KV Cache Store Plugin: End-to-End Validation Guide

[Sample README](README.md)

This guide walks you through validating your own compiled KV cache store plugin (the shared
library implementing [`kv_cache_store_provider_abi.h`](../../../src/cpp/include/openvino/genai/c/kv_cache_store_provider_abi.h))
against the two scenarios this ABI is used for in OpenVINO GenAI:

1. **Opportunistic KV cache offload** (`SchedulerConfig::use_cache_offload`) - blocks evicted during
   normal generation are written through your plugin and rediscovered later via prefix-cache hash.
2. **Session snapshot save/restore** (`ContinuousBatchingPipeline::save_state()`/`restore_state()`) -
   a running request's full KV (and, for hybrid models, linear-attention) cache state is written
   through your plugin instead of being embedded inline in the snapshot file, so it can be resumed
   in a fresh process without re-running prefill.

Both scenarios below were run against the standalone sample plugin in this directory
([`sample_kv_cache_store_plugin.c`](sample_kv_cache_store_plugin.c)) with **Qwen3.5-9B** (a hybrid
KV + linear-attention model): Scenario 1 on CPU **and** GPU, Scenario 2 on GPU. Substitute your own
plugin's path and provider name. GPU's paged-attention blocks hold fewer tokens than CPU's, so the
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
| CPU | 120 | 104668.61 ms | 7852.92 ms | **~13x** | identical (`e067b6f4...`) |
| GPU | 220 | 3091.46 ms | 1529.72 ms | **~2.0x** | identical (`e067b6f4...`) |

The absolute gain is far larger on CPU because CPU prefill is slower, so there is more recompute to
avoid. Writing the evicted blocks is not free: on GPU `P1` took 4266 ms with offload against 3275 ms
without (and `P0` 3637 ms against 3239 ms), while on CPU the difference was within noise (`P1`
113.9 s against 118.0 s). Each figure is a single run. Longer prompts widen the gap and very short
ones can erase it, so benchmark with prompt lengths representative of your workload rather than
relying on these numbers.

**A hybrid model only reuses a prefix up to its last linear-attention checkpoint.** The recurrent
state is checkpointed every `cache_interval` tokens (about 2880 on CPU and 2592 on GPU with the
default sizing, printed as `cache_interval=` in the `[KV_TRACE]` log at `OPENVINO_LOG_LEVEL=3`),
and a repeated prompt cannot resume from a checkpoint that already includes its final token. A
prompt shorter than one interval therefore gets no reuse at all - it is recomputed, and the result
is still correct. That is why this guide's prompts are over 3000 tokens; with short prompts offload
through your plugin will show no speedup on a hybrid model.

### What the host expects from your plugin's stores

The host opens **one store instance per group** and passes the group as the `OvKVCacheNamespace`
(`kv`, `la`, and `session_snapshot_*` for Scenario 2). It sizes each store against its own
capacity and checks that the store's `stats()` stays within it. Your plugin must therefore keep
the namespaces apart: if it puts KV and linear-attention blocks in one store, `stats()` reports
more blocks than the smaller store's capacity and the run fails with `Check 'stats.blocks <=
m_capacity_blocks' failed ... ABI store reported more used blocks than its capacity`. Also make
sure `capacity_bytes` is large enough for your plugin's real per-block storage overhead, for the
same reason.

## Scenario 2: Session snapshot save/restore

Session snapshot addresses blocks by sequence order, not content hash, so this is a *separate*
code path from Scenario 1 even when pointed at the same plugin - validating Scenario 1 does not
imply Scenario 2 also works. Use [`verify_session_snapshot.py`](verify_session_snapshot.py) in
this directory, which:

1. Runs a request to completion, uninterrupted, as the baseline.
2. Runs the same request again, calls `save_state()` partway through with your plugin configured
   (so the KV/linear-attention block payload is written through your plugin), destroys that
   pipeline instance, then constructs a **fresh** pipeline instance (simulating a process restart)
   and calls `restore_state()` with the same plugin configuration to finish generation.
3. Asserts the stitched token stream from step 2 matches step 1's baseline exactly.

```sh
cd samples/c/kv_cache_store_plugin
python verify_session_snapshot.py -m <model_dir> -d GPU \
    --plugin /path/to/your_plugin.dll --provider your-provider-name
```

**What a pass looks like** (actual output from this exact command against the sample plugin, Qwen3.5-9B on GPU):

```
PASS: the restored continuation matches the uninterrupted baseline token-for-token.
```

The script exits 0 on success and non-zero (printing both token sequences) on any mismatch, so it
is safe to use directly in CI. `[ERROR] BlockManager leaked sequence block tables` /
`BlockAllocator leaked blocks` lines on stderr are expected and harmless here - `save_state()` is
only valid while a request is still RUNNING, so the save pipeline's request is deliberately never
allowed to finish before that pipeline instance is destroyed; this is a destructor-time sanity log,
not a failure (the script's own PASS/FAIL assertion, not these log lines, is the actual verdict).

### Performance: TTFT with vs. without session restore

Unless skipped with `--skip-perf`, the script also measures time-to-first-token (TTFT) for a much
longer conversation (the sample sentence repeated 400 times) restored through your plugin, against
a full re-prefill baseline - the correctness check above deliberately uses a short prompt to run
fast, which is not representative of the real performance win (same prefix-length caveat as
Scenario 1).

**Real numbers from this exact script** (Qwen3.5-9B, GPU, sample plugin):

| Device | TTFT without restore (full re-prefill) | TTFT with restore (via your plugin) | Speedup |
|---|---|---|---|
| GPU | 3794.20 ms | 598.76 ms | **~6.3x** |

```
PASS: the restored continuation matches the uninterrupted baseline token-for-token.
PERF: TTFT without session restore (full re-prefill): 3794.20 ms
PERF: TTFT with session restore (via your plugin):    598.76 ms
PERF: Speedup: 6.34x
```

Unlike Scenario 1, this does not depend on prefix-cache hits, so the linear-attention checkpoint
interval caveat does not apply: the correctness check uses a short prompt far below that interval
and still resumes token-for-token.

## Known limitations of the sample plugin (do not assume your plugin needs to share these)

The sample plugin in this directory is intentionally a minimal, in-memory reference - it has
**no real persistence**: its data lives in process memory and is gone once the process actually
exits. `verify_session_snapshot.py`'s "process restart" is simulated with a second
`ContinuousBatchingPipeline` instance in the *same* process, which is sufficient to validate the
ABI call sequence and data correctness, but does **not** prove your plugin survives a real process
restart - that property depends entirely on your own plugin's storage backend actually persisting
to disk/SSD, which the sample deliberately does not do. If your plugin is disk-backed, you can
make this test stronger yourself by running the save half and the restore half as two genuinely
separate OS processes.

The async I/O (`OV_KVCACHE_CAP_ASYNC_IO`: `read_blocks_async`/`write_blocks_async`/`wait`/`cancel`)
and `register_host_memory` (`OV_KVCACHE_CAP_REGISTER_HOST_MEMORY`) capabilities are part of the
frozen ABI shape but are **not yet called by this host version** - the sample plugin does not
implement them (its vtable entries are `NULL`), and neither scenario above exercises them. Treat
them as reserved for a future host version, not something you need to implement or that this guide
validates.
