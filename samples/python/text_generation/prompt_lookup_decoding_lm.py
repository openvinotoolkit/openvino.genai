#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai

def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped. 
    return openvino_genai.StreamingStatus.RUNNING

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'

    pipe = openvino_genai.LLMPipeline(args.model_dir, device, prompt_lookup=True)
    
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    # add parameter to enable prompt lookup decoding to generate `num_assistant_tokens` candidates per iteration
    config.num_assistant_tokens = 5
    # Define max_ngram_size
    config.max_ngram_size = 3

    # Since the streamer is set, the results will be printed 
    # every time a new token is generated and put into the streamer queue.
    pipe.generate(args.prompt, config, streamer)
    print()

if '__main__' == __name__:
    main()
