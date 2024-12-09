#!/usr/bin/env python3
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai as ov_genai
from PIL import Image
from openvino import Tensor
import numpy as np


def read_image(path: str) -> Tensor:
    '''

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    '''
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return Tensor(image_data)


def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default="The Sky is blue because", help="Prompt")
    parser.add_argument("-i", "--image", type=str, default="image.jpg", help="Image")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=20, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")

    args = parser.parse_args()

    # Perf metrics is stored in VLMDecodedResults.
    # In order to get VLMDecodedResults instead of a string input should be a list.
    prompt = args.prompt
    models_path = args.model
    image = read_image(args.image)
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    pipe = ov_genai.VLMPipeline(models_path, device)

    for _ in range(num_warmup):
        pipe.generate(prompt, images=image, generation_config=config)

    res = pipe.generate(prompt, images=image, generation_config=config)
    perf_metrics = res.perf_metrics
    for _ in range(num_iter - 1):
        res = pipe.generate(prompt, images=image, generation_config=config)
        perf_metrics += res.perf_metrics

    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(
        f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(
        f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(
        f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(
        f"Embeddings preparation time: {perf_metrics.get_prepare_embeddings_duration().mean:.2f} ± {perf_metrics.get_prepare_embeddings_duration().std:.2f} ms")
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")


if __name__ == "__main__":
    main()
