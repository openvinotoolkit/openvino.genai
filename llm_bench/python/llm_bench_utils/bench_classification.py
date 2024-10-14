# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import time
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import torch
import numpy as np
import llm_bench_utils.gen_output_data as gen_output_data


FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

def run_image_classification(model_path, framework, device, args, num_iters=10):
    if args['genai']:
        log.warning("GenAI pipeline is not supported for this task. Switched on default benchmarking")
    model, input_size = FW_UTILS[framework].create_image_classification_model(model_path, device, **args)

    data = torch.rand(input_size)

    test_time = []
    iter_data_list = []
    for num in range(num_iters or 10):
        start = time.perf_counter()
        model(data)
        end = time.perf_counter()
        generation_time = end - start
        test_time.append(generation_time)

        iter_data = gen_output_data.gen_iterate_data(iter_idx=num, in_size=input_size, infer_count=num_iters, gen_time=generation_time)
        iter_data_list.append(iter_data)
    log.info(f'Processed {num_iters} images in {np.sum(test_time)}s')
    log.info(f'Average processing time {np.mean(test_time)} s')
    return iter_data_list