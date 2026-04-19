#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark script for Qwen3-ASR model using openvino.genai built-in PerfMetrics.

Measures: throughput (tokens/sec), TTFT, TPOT, RTF, feature extraction time,
          encoder time, decoder time using openvino.genai's PerfMetrics API.

Usage:
    python benchmark_qwen3_asr.py <model_dir> <wav_file> [device] [--runs N]

Example:
    python benchmark_qwen3_asr.py ./qwen3-asr-1.7b-ov speech.wav CPU --runs 5

Note: openvino.genai WhisperPipeline auto-detects Qwen3-ASR from config.json.
      RTF (Real-Time Factor) = processing_time / audio_duration.
      RTF < 1.0 means faster than real-time.
"""

import argparse
import sys
import time
import numpy as np

try:
    import openvino_genai as ov_genai
except ImportError:
    print("ERROR: openvino_genai not found. Build and install openvino.genai first.")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("ERROR: librosa not found. Install with: pip install librosa")
    sys.exit(1)


def read_wav(filepath):
    """Load audio file, resample to 16kHz mono float32."""
    raw_speech, _ = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()


def format_table_row(label, mean, std, unit="ms"):
    """Format a single table row."""
    if unit == "tok/s":
        return f"  │ {label:<30} │ {mean:>10.1f}  │ {std:>7.1f}   │"
    elif unit == "x":
        return f"  │ {label:<30} │ {mean:>10.3f}  │ {std:>7.3f}   │"
    elif unit == "count":
        return f"  │ {label:<30} │ {mean:>10.0f}  │ {std:>7.0f}   │"
    else:
        return f"  │ {label:<30} │ {mean:>10.1f}  │ {std:>7.1f}   │"


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-ASR with openvino.genai")
    parser.add_argument("model_dir", help="Path to exported Qwen3-ASR model directory")
    parser.add_argument("wav_file", help="Path to WAV/FLAC audio file (will be resampled to 16kHz)")
    parser.add_argument("device", nargs="?", default="CPU", help="Device (default: CPU)")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens (default: 512)")
    args = parser.parse_args()

    # Load audio
    print(f"Loading audio: {args.wav_file}")
    raw_speech = read_wav(args.wav_file)
    audio_duration_sec = len(raw_speech) / 16000.0
    print(f"  Duration: {audio_duration_sec:.2f}s ({len(raw_speech)} samples at 16kHz)")

    # Load pipeline
    print(f"\nLoading model: {args.model_dir} on {args.device}")
    pipe = ov_genai.WhisperPipeline(args.model_dir, args.device)
    load_time = pipe.get_generation_config()  # triggers lazy load if needed

    config = pipe.get_generation_config()
    config.max_new_tokens = args.max_tokens

    # Warmup
    print(f"\nWarmup ({args.warmup} run{'s' if args.warmup > 1 else ''})...")
    for i in range(args.warmup):
        result = pipe.generate(raw_speech, config)
        print(f"  Warmup {i+1}: \"{result.texts[0][:70]}...\"")

    # Benchmark
    print(f"\nBenchmark ({args.runs} runs)...")
    all_metrics = []
    all_rtfs = []

    for i in range(args.runs):
        result = pipe.generate(raw_speech, config)
        m = result.perf_metrics

        # RTF from generate_duration (genai tracks this internally)
        gen_dur_ms = m.get_generate_duration().mean
        rtf = (gen_dur_ms / 1000.0) / audio_duration_sec

        all_metrics.append(m)
        all_rtfs.append(rtf)

        tps = m.get_throughput().mean
        ttft = m.get_ttft().mean
        tpot = m.get_tpot().mean
        n_tok = m.get_num_generated_tokens()
        print(f"  Run {i+1}: {n_tok} tok | {gen_dur_ms:.0f}ms | "
              f"RTF={rtf:.3f} | {tps:.1f} tok/s | TTFT={ttft:.0f}ms | TPOT={tpot:.1f}ms")

    # Aggregate
    def agg(fn):
        vals = [fn(m) for m in all_metrics]
        means = [v.mean if hasattr(v, 'mean') else v for v in vals]
        return np.mean(means), np.std(means)

    print("\n" + "=" * 70)
    print(f"  Performance Summary — Qwen3-ASR-1.7B on {args.device}")
    print("=" * 70)
    print(f"  Audio duration:      {audio_duration_sec:.2f}s")
    print(f"  Runs:                {args.runs}")
    load_ms = all_metrics[0].get_load_time()
    print(f"  Model load time:     {load_ms:.0f}ms")
    print()

    hdr = "  ┌──────────────────────────────┬──────────────┬───────────┐"
    sep = "  ├──────────────────────────────┼──────────────┼───────────┤"
    ftr = "  └──────────────────────────────┴──────────────┴───────────┘"

    print(hdr)
    print(f"  │ {'Metric':<30} │ {'Mean':>12} │ {'Std':>9} │")
    print(sep)

    m, s = agg(lambda x: x.get_generate_duration())
    print(format_table_row("Generate Duration (ms)", m, s))

    m, s = agg(lambda x: x.get_inference_duration())
    print(format_table_row("Inference Duration (ms)", m, s))

    m, s = agg(lambda x: x.get_features_extraction_duration())
    print(format_table_row("Feature Extraction (ms)", m, s))

    m, s = agg(lambda x: x.get_ttft())
    print(format_table_row("TTFT (ms)", m, s))

    m, s = agg(lambda x: x.get_tpot())
    print(format_table_row("TPOT (ms/token)", m, s))

    m, s = agg(lambda x: x.get_ipot())
    print(format_table_row("IPOT (ms/token)", m, s))

    m, s = agg(lambda x: x.get_throughput())
    print(format_table_row("Throughput (tokens/sec)", m, s, "tok/s"))

    m, s = np.mean(all_rtfs), np.std(all_rtfs)
    print(format_table_row("RTF (Real-Time Factor)", m, s, "x"))

    m, s = agg(lambda x: x.get_num_generated_tokens())
    print(format_table_row("Generated Tokens", m, s, "count"))

    m, s = agg(lambda x: x.get_detokenization_duration())
    print(format_table_row("Detokenization (ms)", m, s))

    print(ftr)
    print()

    rtf_mean = np.mean(all_rtfs)
    if rtf_mean < 1.0:
        print(f"  ✅ RTF = {rtf_mean:.3f} → {1/rtf_mean:.1f}x faster than real-time")
    else:
        print(f"  ⚠️  RTF = {rtf_mean:.3f} → {rtf_mean:.1f}x slower than real-time")

    print()
    print("  Metrics explanation:")
    print("    TTFT  = Time To First Token (encoder + decoder prefill)")
    print("    TPOT  = Time Per Output Token (autoregressive decode)")
    print("    IPOT  = Inference-only time Per Output Token")
    print("    RTF   = Real-Time Factor (total_time / audio_duration)")
    print("            RTF < 1.0 means faster than real-time")
    print("=" * 70)


if __name__ == "__main__":
    main()
