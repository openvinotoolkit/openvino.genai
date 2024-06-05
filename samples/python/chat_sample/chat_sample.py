#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def streamer(subword):
    print(subword, end='')
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = pipe.get_generation_config()
    config.max_new_tokens = 20
    config.do_sample = False

    pipe.start_chat()
    while True:
        prompt = input('question:\n')
        if 'Stop!' == prompt:
            break
        pipe.generate(prompt, config, streamer)

        print('\n----------')
    pipe.finish_chat()


if '__main__' == __name__:
    main()
