#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import openvino_genai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("predictor_path", help="LES predictor file in safetensors format")
    parser.add_argument("prompt")
    parser.add_argument("--offload_ratio", type=int, default=50, help="% of the routed experts kept in host memory")
    args = parser.parse_args()

    device = "GPU"  # LES is implemented in the GPU plugin

    # Lookahead expert scheduling (LES) for Mixture-of-Experts models with offloaded experts: offload_ratio % of the
    # routed experts stay in host memory and are uploaded to the GPU on demand. The LES predictor tells the GPU plugin
    # which experts the next MoE layer is likely to select, so it can start uploading them in advance.
    # The predictor does not change the generated text, only the decode speed.
    # The GPU plugin reads the offloaded experts from the weights file of the model (WEIGHTS_PATH).
    predictor = openvino_genai.LESPredictor(args.predictor_path)
    pipe = openvino_genai.LLMPipeline(
        args.model_dir,
        device,
        les_predictor=predictor,
        OFFLOAD_RATIO=args.offload_ratio,
        WEIGHTS_PATH=str(Path(args.model_dir).resolve() / "openvino_model.bin"),
    )

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    result = pipe.generate([args.prompt], config)  # a list of prompts gives DecodedResults with perf_metrics
    print(result.texts[0])

    metrics = result.perf_metrics
    print(
        f"\nOFFLOAD_RATIO {args.offload_ratio}: TTFT {metrics.get_ttft().mean:.1f} ms, "
        f"decode {metrics.get_throughput().mean:.1f} tokens/s"
    )


if "__main__" == __name__:
    main()
