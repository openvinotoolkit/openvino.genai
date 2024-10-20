#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import queue
import threading

def streamer(subword): 
        print(subword, end='', flush=True) 
        # Return flag corresponds whether generation should be stopped. 
        # False means continue generation. 
        return False 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('draft_model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    # User can run main and draft model on different devices.
    # Please, set device for main model in `LLMPipeline` constructor and in in `ov::genai::draft_model` for draft.
    main_device = 'CPU'  # GPU can be used as well
    draft_device = main_device

    scheduler_config = openvino_genai.SchedulerConfig()
    # cache params
    scheduler_config.cache_size = 2
    scheduler_config.block_size = 32

    draft_model = openvino_genai.draft_model(args.draft_model_dir, draft_device,)

    ov_config = { "scheduler_config": scheduler_config, "draft_model": draft_model }

    pipe = openvino_genai.LLMPipeline(args.model_dir, main_device, ov_config)
    
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    config.num_assistant_tokens = 5

    # Since the streamer is set, the results will be printed 
    # every time a new token is generated and put into the streamer queue.
    pipe.generate(args.prompt, config, streamer)

if '__main__' == __name__:
    main()
