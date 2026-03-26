#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
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
    parser.add_argument('--system_prompt', type=str, default=None,
                        help='Optional system prompt to set the LLM persona/instructions')
    args = parser.parse_args()

    device = args.device
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    chat_history = openvino_genai.ChatHistory()
    if args.system_prompt:
        chat_history.append({"role": "system", "content": args.system_prompt})
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
