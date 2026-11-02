# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compares time-to-first-token (TTFT) of resuming a long conversation from scratch (full
re-prefill) against restoring it from a session snapshot via ContinuousBatchingPipeline's
save_state()/restore_state(). Mirrors tests/cpp/session_snapshot_e2e.cpp's
RestoreAvoidsFullPrefillLatency, but exercised through the Python bindings instead of C++.

The snapshot keeps its cache blocks in the prefix-caching offload store, and the restoring pipeline is a new
instance, so a store plugin that outlives the first pipeline is required (for example
samples/c/kv_cache_store_plugin).

Usage:
    python session_snapshot_bench.py -m <model_dir> -d CPU --plugin <path to the store plugin> --provider <name>
"""

import argparse
import os
import tempfile
import time

import openvino_genai as ov_genai


def make_scheduler_config(plugin_path: str = None, provider: str = "") -> ov_genai.SchedulerConfig:
    config = ov_genai.SchedulerConfig()
    config.enable_prefix_caching = True
    config.max_num_batched_tokens = 8192
    config.max_num_seqs = 4
    config.dynamic_split_fuse = False
    if plugin_path is not None:
        config.use_cache_offload = True
        offload = ov_genai.CacheOffloadConfig()
        offload.store_library_path = plugin_path
        offload.store_provider_name = provider
        offload.allow_untrusted_store_plugin = True
        # The snapshot's blocks are pinned in the store, so it must hold the whole conversation.
        offload.capacity_bytes = 2 * 1024 ** 3
        offload.linear_attention_capacity_bytes = 4 * 1024 ** 3
        config.cache_offload_config = offload
    return config


def make_greedy_config(max_new_tokens: int) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    config.num_beams = 1
    config.ignore_eos = True
    return config


def measure_baseline_ttft(model_dir: str, device: str, prompt: str) -> float:
    pipeline = ov_genai.ContinuousBatchingPipeline(model_dir, make_scheduler_config(), device)
    # Timed from before the request is submitted, the same way as the restore path below.
    start = time.perf_counter()
    # The handle must outlive the loop: destroying it stops the request before any token is produced.
    handle = pipeline.add_request(61, prompt, make_greedy_config(1))
    while pipeline.has_non_finished_requests():
        pipeline.step()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if len(handle.read_all()[0].generated_ids) != 1:
        raise RuntimeError("the baseline request did not produce exactly one token")
    return elapsed_ms


def measure_restore_ttft(model_dir: str, device: str, prompt: str, snapshot_path: str,
                         plugin_path: str, provider: str) -> float:
    # Save mid-generation (still RUNNING): a finished request's cache is freed immediately, so
    # there is no window to save it afterwards.
    pipeline = ov_genai.ContinuousBatchingPipeline(model_dir, make_scheduler_config(plugin_path, provider), device)
    request_id = 62
    handle = pipeline.add_request(request_id, prompt, make_greedy_config(50))
    for _ in range(5):
        if handle.get_status() != ov_genai.GenerationStatus.RUNNING:
            break
        pipeline.step()
    if handle.get_status() != ov_genai.GenerationStatus.RUNNING:
        raise RuntimeError("the request finished naturally before it could be interrupted")
    pipeline.save_state(request_id, snapshot_path)
    del pipeline, handle

    # Simulate a process restart: brand-new pipeline instance restores from the store plugin.
    pipeline = ov_genai.ContinuousBatchingPipeline(model_dir, make_scheduler_config(plugin_path, provider), device)
    # The handle's own TTFT clock starts only after restore_state() has loaded the snapshot, so time
    # the whole restore instead.
    start = time.perf_counter()
    handle = pipeline.restore_state(63, snapshot_path, make_greedy_config(1))
    while pipeline.has_non_finished_requests():
        pipeline.step()
    elapsed_ms = (time.perf_counter() - start) * 1000
    if len(handle.read_all()[0].generated_ids) != 1:
        raise RuntimeError("the restored request did not produce exactly one token")
    return elapsed_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", required=True, help="Path to an OpenVINO IR model directory")
    parser.add_argument("-d", "--device", default="CPU", help="Inference device (default: CPU)")
    parser.add_argument("--repeats", type=int, default=400, help="Number of times to repeat the filler sentence")
    parser.add_argument("--plugin", required=True, help="Path to a KV cache store plugin (.dll/.so)")
    parser.add_argument("--provider", default="", help="Provider name to select (empty = module's first provider)")
    args = parser.parse_args()

    prompt = "The quick brown fox jumps over the lazy dog. " * args.repeats

    baseline_ttft_ms = measure_baseline_ttft(args.model, args.device, prompt)

    snapshot_path = os.path.join(tempfile.gettempdir(), "openvino_genai_session_snapshot_bench.bin")
    try:
        restore_ttft_ms = measure_restore_ttft(args.model, args.device, prompt, snapshot_path, args.plugin, args.provider)
    finally:
        if os.path.exists(snapshot_path):
            os.remove(snapshot_path)

    print(f"TTFT without session restore (full re-prefill): {baseline_ttft_ms:.2f} ms")
    print(f"TTFT with session restore (no re-prefill):       {restore_ttft_ms:.2f} ms")
    print(f"Speedup: {baseline_ttft_ms / restore_ttft_ms:.2f}x")


if __name__ == "__main__":
    main()
