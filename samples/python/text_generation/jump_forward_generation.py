#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import statistics
import time

from openvino_genai import (
    ContinuousBatchingPipeline,
    GenerationConfig,
    SchedulerConfig,
    StructuredOutputConfig as SOC,
)


NUM_RUNS = 5


def choice(*values):
    return SOC.Union([SOC.ConstString(json.dumps(value)) for value in values])


grammar = SOC.Concat(
    SOC.ConstString('{"request_id":"jf-demo-001","priority":'),
    choice("low", "medium", "high"),
    SOC.ConstString(',"route":'),
    choice("cpu", "gpu", "npu"),
    SOC.ConstString(',"approved":'),
    choice(True, False),
    SOC.ConstString(',"metadata":{"source":"openvino-genai","version":1}}'),
)


def make_config(enable_jump_forward):
    config = GenerationConfig(max_new_tokens=256, do_sample=False)
    config.structured_output_config = SOC(
        structural_tags_config=grammar,
        enable_jump_forward=enable_jump_forward,
    )
    return config


def generate(pipe, prompt_ids, config):
    start = time.perf_counter()
    result = pipe.generate([prompt_ids], [config])[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    token_ids = result.m_generation_ids[0]
    return {
        "text": pipe.get_tokenizer().decode(token_ids),
        "tokens": len(token_ids),
        "iterations": len(result.perf_metrics.raw_metrics.m_durations),
        "elapsed_ms": elapsed_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare structured output with and without jump-forward decoding.")
    parser.add_argument(
        "model_dir",
        help="Path to a converted OpenVINO model directory.",
    )
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()

    pipe = ContinuousBatchingPipeline(
        args.model_dir,
        SchedulerConfig(),
        args.device,
    )
    prompt_ids = (
        pipe.get_tokenizer()
        .encode("Classify this request. Choose a priority, execution route, and whether to approve it.")
        .input_ids
    )
    configs = {
        "Regular": make_config(False),
        "Jump-forward": make_config(True),
    }

    print("Warming up both decoding modes ...")
    for config in configs.values():
        generate(pipe, prompt_ids, config)

    runs = {name: [] for name in configs}
    for run_index in range(NUM_RUNS):
        order = configs if run_index % 2 == 0 else reversed(configs)
        for name in order:
            runs[name].append(generate(pipe, prompt_ids, configs[name]))

    texts = {result["text"] for mode_runs in runs.values() for result in mode_runs}
    if len(texts) != 1:
        raise RuntimeError("Regular and jump-forward outputs differ")
    output = json.loads(texts.pop())

    results = {}
    for name, mode_runs in runs.items():
        results[name] = mode_runs[0] | {"elapsed_ms": statistics.median(run["elapsed_ms"] for run in mode_runs)}

    regular = results["Regular"]
    jump_forward = results["Jump-forward"]
    iteration_reduction = regular["iterations"] - jump_forward["iterations"]
    time_saved = regular["elapsed_ms"] - jump_forward["elapsed_ms"]

    print(f"\nStructured-output A/B comparison ({NUM_RUNS}-run median)")
    for name, result in results.items():
        print(
            f"{name:<13} tokens={result['tokens']}, "
            f"model iterations={result['iterations']}, "
            f"elapsed={result['elapsed_ms']:.2f} ms"
        )
    print(f"Iteration reduction: {iteration_reduction} ({100 * iteration_reduction / regular['iterations']:.1f}%)")
    print(
        f"Wall-time reduction: {time_saved:.2f} ms "
        f"({100 * time_saved / regular['elapsed_ms']:.1f}%), "
        f"speedup={regular['elapsed_ms'] / jump_forward['elapsed_ms']:.2f}x"
    )
    print("Output verification: identical and valid")
    print("\nGenerated JSON:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
