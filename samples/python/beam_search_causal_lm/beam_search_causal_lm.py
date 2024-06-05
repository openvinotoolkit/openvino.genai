#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompts', nargs='+')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = pipe.get_generation_config()
    config.max_new_tokens = 20
    config.num_beam_groups = 3
    config.num_beams = 15
    config.num_return_sequences = config.num_beams * len(args.prompts)

    beams = pipe.generate(args.prompts, config)
    print(beams)


if '__main__' == __name__:
    main()
