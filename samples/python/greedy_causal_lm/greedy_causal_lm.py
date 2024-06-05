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
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = pipe.get_generation_config()
    config.max_new_tokens = 100
    config.do_sample = False

    # Since the streamer is set, the results will
    # be printed each time a new token is generated
    pipe.generate(args.prompt, config, streamer)


if '__main__' == __name__:
    main()
