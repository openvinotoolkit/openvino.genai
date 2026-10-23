#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Verifies a third-party KV cache store plugin against the session snapshot
(save_state()/restore_state()) scenario.

Runs the same request three ways:
  1. Start to finish, uninterrupted - the baseline.
  2. Partway through, call save_state() with your plugin configured, so the KV (and, for hybrid
     models, linear-attention) block payload is written through your plugin instead of being
     embedded inline in the snapshot file.
  3. In a fresh ContinuousBatchingPipeline instance (simulating a process restart), call
     restore_state() with the same plugin configuration and finish generation.

The stitched token stream from steps 2+3 must match step 1's baseline exactly. Prints PASS/FAIL
and exits non-zero on failure or any exception.

Usage:
    python verify_session_snapshot.py -m <model_dir> --plugin <path to your compiled plugin>
                                       [-d CPU] [--provider <provider name>]
"""

import argparse
import os
import sys
import tempfile

import openvino_genai as ov_genai

TOTAL_NEW_TOKENS = 20
# Large enough that manually driving a handful of steps below cannot reach it naturally.
SPLIT_RUN_BUDGET = 1000
NUM_MANUAL_STEPS = 15
PROMPT = "The quick brown fox jumps over the lazy dog. " * 10 + " Continue the story:"


def make_scheduler_config() -> ov_genai.SchedulerConfig:
    config = ov_genai.SchedulerConfig()
    config.enable_prefix_caching = True
    config.max_num_batched_tokens = 2048
    config.max_num_seqs = 4
    config.dynamic_split_fuse = False
    return config


def make_greedy_config(max_new_tokens: int) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    config.num_beams = 1
    config.ignore_eos = True
    return config


def make_store_config(plugin_path: str, provider_name: str) -> ov_genai.CacheOffloadConfig:
    config = ov_genai.CacheOffloadConfig()
    config.store_library_path = plugin_path
    config.store_provider_name = provider_name
    config.allow_untrusted_store_plugin = True
    return config


def run_baseline(model: str, device: str) -> list:
    pipeline = ov_genai.ContinuousBatchingPipeline(model, make_scheduler_config(), device)
    handle = pipeline.add_request(1, PROMPT, make_greedy_config(TOTAL_NEW_TOKENS))
    while pipeline.has_non_finished_requests():
        pipeline.step()
    outputs = handle.read_all()
    assert outputs, "baseline run produced no output"
    baseline_ids = outputs[0].generated_ids
    assert len(baseline_ids) == TOTAL_NEW_TOKENS, (
        f"expected {TOTAL_NEW_TOKENS} baseline tokens, got {len(baseline_ids)}")
    return baseline_ids


def run_save_then_restore(model: str, device: str, store_config: ov_genai.CacheOffloadConfig,
                           snapshot_path: str) -> list:
    already_generated_ids = []
    pipeline = ov_genai.ContinuousBatchingPipeline(model, make_scheduler_config(), device)
    handle = pipeline.add_request(2, PROMPT, make_greedy_config(SPLIT_RUN_BUDGET))
    for _ in range(NUM_MANUAL_STEPS):
        if handle.get_status() != ov_genai.GenerationStatus.RUNNING:
            break
        pipeline.step()
        while handle.can_read():
            for _, output in handle.read().items():
                already_generated_ids.extend(output.generated_ids)
    assert handle.get_status() == ov_genai.GenerationStatus.RUNNING, (
        "the request finished naturally before it could be interrupted - test setup is invalid")
    assert len(already_generated_ids) < TOTAL_NEW_TOKENS, (
        "manual stepping already produced as many tokens as the baseline, nothing left to resume")

    pipeline.save_state(2, snapshot_path, store_config)
    del pipeline, handle

    # Fresh pipeline instance simulates a process restart; restore_state() must find the KV/LA
    # block payload through the SAME plugin configuration save_state() was called with.
    pipeline = ov_genai.ContinuousBatchingPipeline(model, make_scheduler_config(), device)
    remaining_tokens = TOTAL_NEW_TOKENS - len(already_generated_ids)
    handle = pipeline.restore_state(3, snapshot_path, make_greedy_config(remaining_tokens), store_config)
    while pipeline.has_non_finished_requests():
        pipeline.step()
    outputs = handle.read_all()
    assert outputs, "restored run produced no output"
    resumed_ids = outputs[0].generated_ids

    return already_generated_ids + resumed_ids


def measure_speedup(model: str, device: str, store_config: ov_genai.CacheOffloadConfig,
                     snapshot_path: str) -> None:
    """Compares TTFT of a full re-prefill against restoring from a snapshot whose KV/linear-
    attention payload was written through the plugin. Uses a much longer prompt than the
    correctness check above - a short prompt's prefill is cheap enough that the gap barely shows;
    this is the same lesson that applies to opportunistic offload (see E2E_VALIDATION.md)."""
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 400
    scheduler_config = make_scheduler_config()
    scheduler_config.max_num_batched_tokens = 8192

    pipeline = ov_genai.ContinuousBatchingPipeline(model, scheduler_config, device)
    handle = pipeline.add_request(11, long_prompt, make_greedy_config(1))
    while pipeline.has_non_finished_requests():
        pipeline.step()
    baseline_ttft_ms = handle.get_perf_metrics().get_ttft().mean
    del pipeline, handle

    pipeline = ov_genai.ContinuousBatchingPipeline(model, scheduler_config, device)
    handle = pipeline.add_request(12, long_prompt, make_greedy_config(50))
    for _ in range(5):
        if handle.get_status() != ov_genai.GenerationStatus.RUNNING:
            break
        pipeline.step()
    assert handle.get_status() == ov_genai.GenerationStatus.RUNNING, (
        "the request finished naturally before it could be interrupted - test setup is invalid")
    pipeline.save_state(12, snapshot_path, store_config)
    del pipeline, handle

    pipeline = ov_genai.ContinuousBatchingPipeline(model, scheduler_config, device)
    handle = pipeline.restore_state(13, snapshot_path, make_greedy_config(1), store_config)
    while pipeline.has_non_finished_requests():
        pipeline.step()
    restore_ttft_ms = handle.get_perf_metrics().get_ttft().mean
    del pipeline, handle
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)

    print(f"PERF: TTFT without session restore (full re-prefill): {baseline_ttft_ms:.2f} ms")
    print(f"PERF: TTFT with session restore (via your plugin):    {restore_ttft_ms:.2f} ms")
    print(f"PERF: Speedup: {baseline_ttft_ms / restore_ttft_ms:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="Path to an OpenVINO IR model directory")
    parser.add_argument("-d", "--device", default="CPU", help="Inference device (default: CPU)")
    parser.add_argument("--plugin", required=True, help="Path to your compiled store plugin (.dll/.so)")
    parser.add_argument("--provider", default="", help="Provider name to select (empty = module's first provider)")
    parser.add_argument("--skip-perf", action="store_true",
                        help="Skip the TTFT speedup measurement (which uses a much longer prompt and takes longer)")
    args = parser.parse_args()

    baseline_ids = run_baseline(args.model, args.device)

    store_config = make_store_config(args.plugin, args.provider)
    snapshot_path = os.path.join(tempfile.gettempdir(), "openvino_genai_plugin_session_snapshot_verify.bin")
    try:
        stitched_ids = run_save_then_restore(args.model, args.device, store_config, snapshot_path)
    finally:
        if os.path.exists(snapshot_path):
            os.remove(snapshot_path)

    if stitched_ids != baseline_ids:
        print("FAIL: the restored continuation diverged from the uninterrupted baseline.")
        print(f"  baseline: {baseline_ids}")
        print(f"  stitched: {stitched_ids}")
        return 1
    print("PASS: the restored continuation matches the uninterrupted baseline token-for-token.")

    if not args.skip_perf:
        measure_speedup(args.model, args.device, store_config, snapshot_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
