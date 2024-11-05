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
    # Please, set device for main model in `openvino_genai.LLMPipeline` constructor and in openvino_genai.draft_model` for draft.
    main_device = 'CPU'  # GPU can be used as well
    draft_device = 'CPU'

    scheduler_config = openvino_genai.SchedulerConfig()
    # cache params
    scheduler_config.cache_size = 2

    draft_scheduler_config = openvino_genai.SchedulerConfig()
    # cache params
    draft_scheduler_config.cache_size = 2

    draft_model = openvino_genai.draft_model(args.draft_model_dir, draft_device, scheduler_config=draft_scheduler_config)

    pipe = openvino_genai.LLMPipeline(args.model_dir, main_device, scheduler_config=scheduler_config, draft_model=draft_model)
    
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    # Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded
    # add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration
    config.num_assistant_tokens = 5
    # add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is higher than `assistant_confidence_threshold`
    # config.assistant_confidence_threshold = 0.4

    # Since the streamer is set, the results will be printed 
    # every time a new token is generated and put into the streamer queue.
    pipe.generate(args.prompt, config, streamer)

if '__main__' == __name__:
    main()
