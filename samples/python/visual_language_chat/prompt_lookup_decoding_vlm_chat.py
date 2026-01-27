#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import openvino_genai
from openvino import Tensor


def streamer(subword):
    print(subword, end="", flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING


def read_image(path: str) -> Tensor:
    """

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    """
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("image_dir", help="Image file or dir with images")
    parser.add_argument(
        "--disable_lookup",
        action="store_false",
        dest="enable_lookup",
        default=True,
        help="Disable lookup decoding (i.e., set enable_lookup to False).",
    )

    args = parser.parse_args()

    device = "CPU"

    # Prompt lookup decoding in VLM pipeline enforces ContinuousBatching backend.
    pipe = openvino_genai.VLMPipeline(args.model_dir, device, prompt_lookup=args.enable_lookup)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    if args.enable_lookup:
        # add parameter to enable prompt lookup decoding to generate `num_assistant_tokens` candidates per iteration
        config.num_assistant_tokens = 5
        # Define max_ngram_size
        config.max_ngram_size = 3

    rgbs = read_images(args.image_dir)

    chat_history = openvino_genai.ChatHistory()
    while True:
        try:
            prompt = input("\n----------\nquestion:\n")
        except EOFError:
            break
        # New images and videos can be passed at each turn
        chat_history.append({"role": "user", "content": prompt})
        decoded_results: openvino_genai.DecodedResults = pipe.generate(
            chat_history, images=rgbs, generation_config=config, streamer=streamer
        )
        chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})
    print()


if __name__ == "__main__":
    main()
