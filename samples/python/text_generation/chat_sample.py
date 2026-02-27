#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def streamer(subword):
    print(subword, end="", flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    device = args.device
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    chat_history = openvino_genai.ChatHistory()
    while True:
        try:
            prompt = input("question:\n")
        except EOFError:
            break
        chat_history.append({"role": "user", "content": prompt})
        decoded_results: openvino_genai.DecodedResults = pipe.generate(chat_history, config, streamer)
        chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})
        print("\n----------")


if "__main__" == __name__:
    main()
