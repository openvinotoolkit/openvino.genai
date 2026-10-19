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
([`sample_kv_cache_store_plugin.c`](sample_kv_cache_store_plugin.c)) on Qwen3-8B/CPU to confirm
this guide's exact commands work as written; substitute your own plugin's path and provider name.

## Prerequisites

- An OpenVINO GenAI build or package with Python bindings (`openvino_genai`).
- Your compiled plugin library (`.dll`/`.so`), built against `kv_cache_store_provider_abi.h`.
- Any text-generation model in OpenVINO IR format (a plain causal LM is enough for both scenarios;
  a hybrid KV+linear-attention model additionally exercises the linear-attention payload).

## Scenario 1: Opportunistic KV cache offload

Point `CacheOffloadConfig` at your plugin and force real eviction by giving the pipeline far less
KV cache capacity than the conversation needs. [`example_offload_cb_config.json`](example_offload_cb_config.json)
in this directory is a ready-to-edit starting point - replace `store_library_path` with your
plugin's path:

```json
{
  "num_kv_blocks": 4,
  "max_num_batched_tokens": 2048,
  "max_num_seqs": 1,
  "dynamic_split_fuse": true,
  "enable_prefix_caching": true,
  "use_cache_offload": true,
  "cache_offload_config": {
    "capacity_bytes": 16777216,
    "buffer_slots": 64,
    "store_library_path": "/path/to/your_plugin.dll",
    "store_provider_name": "your-provider-name",
    "allow_untrusted_store_plugin": true
  }
}
```

`num_kv_blocks: 4` is deliberately tiny so a single ~130-token generation already exceeds it,
forcing real writes/reads through your plugin instead of only validating the no-eviction-needed
path. Run it with `tools/llm_bench/benchmark.py`:

```sh
cd tools/llm_bench
python benchmark.py -m <model_dir> -d CPU \
    -p "Tell me a long story about a robot learning to paint." \
    -n 1 -ic 150 --disable_prompt_permutation \
    --cb_config ../../samples/c/kv_cache_store_plugin/example_offload_cb_config.json
```

**What a pass looks like** (actual output from this exact command against the sample plugin):

```
[ INFO ] Running average of the KV cache usage 66.31%, max KV cache usage: 100.00%
[ INFO ] [warm-up][P0] Result MD5:['05a053881ab169c397058f5071d5552b']
...
[ INFO ] [1][P0] Result MD5:['05a053881ab169c397058f5071d5552b']
```

- `max KV cache usage: 100.00%` with only 4 blocks configured confirms blocks were actually
  evicted and had to be restored - not just an idle plugin that was never exercised.
- The two MD5 lines (warm-up iteration vs. the measured iteration) must match each other - that
  confirms deterministic, correct round-tripping through your plugin (greedy decoding + no
  eviction-order bugs would otherwise make the second run diverge).
- No `[ERROR]` lines about your plugin failing to load/handshake/open (those are logged as
  warnings and silently fall back to the in-tree default store - check the log for
  `KV cache store plugin ... falling back to the default file store` if the run otherwise looks
  suspiciously fast or `max KV cache usage` never climbs - that means your plugin was never
  actually loaded).

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
python verify_session_snapshot.py -m <model_dir> -d CPU \
    --plugin /path/to/your_plugin.dll --provider your-provider-name
```

**What a pass looks like** (actual output from this exact command against the sample plugin):

```
PASS: the restored continuation matches the uninterrupted baseline token-for-token.
```

The script exits 0 on success and non-zero (printing both token sequences) on any mismatch, so it
is safe to use directly in CI. `[ERROR] BlockManager leaked sequence block tables` /
`BlockAllocator leaked blocks` lines on stderr are expected and harmless here - `save_state()` is
only valid while a request is still RUNNING, so the save pipeline's request is deliberately never
allowed to finish before that pipeline instance is destroyed; this is a destructor-time sanity log,
not a failure (the script's own PASS/FAIL assertion, not these log lines, is the actual verdict).

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
