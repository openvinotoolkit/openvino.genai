# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import datetime
from PIL import Image
import hashlib
import logging as log
from transformers import set_seed
import llm_bench_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import BenchPrompter

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_SUPER_RESOLUTION_STEPS = 50
DEFAULT_SUPER_RESOLUTION_WIDTH = 128
DEFAULT_SUPER_RESOLUTION_HEIGHT = 128


def run_ldm_super_resolution(img, num, pipe, args, framework, iter_data_list, image_id, tm_list, proc_id, mem_consumption):
    set_seed(args['seed'])
    nsteps = img.get('steps', DEFAULT_SUPER_RESOLUTION_STEPS)
    resize_image_width = img.get('width', DEFAULT_SUPER_RESOLUTION_WIDTH)
    resize_image_height = img.get('height', DEFAULT_SUPER_RESOLUTION_HEIGHT)
    log.info(
        f"[{'warm-up' if num == 0 else num}][P{image_id}] Input params: steps={nsteps}, "
        f'resize_width={resize_image_width}, resize_height={resize_image_height}'
    )
    low_res_img = Image.open(img['prompt']).convert('RGB')
    low_res_img = low_res_img.resize((resize_image_width, resize_image_height))
    mem_consumption.start(num)
    start = time.perf_counter()
    res = pipe(low_res_img, num_inference_steps=nsteps, tm_list=tm_list)
    end = time.perf_counter()
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    result_md5_list = []
    if framework == 'ov':
        rslt_img_fn = llm_bench_utils.output_file.output_gen_image(res[0], args, image_id, num, None, proc_id, '.png')
        result_md5_list.append(hashlib.md5(Image.open(rslt_img_fn).tobytes(), usedforsecurity=False).hexdigest())

    generation_time = end - start
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        infer_count=nsteps,
        gen_time=generation_time,
        res_md5=result_md5_list,
        prompt_idx=image_id,
        **memory_metrics,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        prompt_idx=image_id
    )
    metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn, prompt_idx=image_id)
    metrics_print.print_ldm_unet_vqvae_infer_latency(num, iter_data, tm_list, warm_up=(num == 0), prompt_idx=image_id)


def run_ldm_super_resolution_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    if args["genai"]:
        log.warning("GenAI pipeline is not supported for this task. Switched on default benchmarking")
    mem_consumption.update_marker("model")
    pipe, pretrain_time = FW_UTILS[framework].create_ldm_super_resolution_model(model_path, device, mem_consumption, **args)
    iter_data_list = []
    tm_list = []

    # Build the prompt schedule via BenchPrompter, which:
    #   - reads and parses the image prompt file (JSON or plain path)
    #   - resolves image paths relative to the prompt file
    #   - honours args['prompt_index'] for selective benchmarking
    #   - handles both subsequent=False (iter-major) and subsequent=True
    #     (prompt-major) scheduling in a single unified iter_schedule() loop,
    #     eliminating the previous duplicated if/else filter blocks.
    prompter = BenchPrompter(args)
    prompt_idx_list = prompter.active_indices
    image_list = prompter.active_items

    log.info(
        f'Benchmarking iter nums(exclude warm-up): {num_iters}, '
        f'image nums: {len(prompter)}, prompt idx: {prompt_idx_list}'
    )

    # if num_iters == 0, just output warm-up data
    proc_id = os.getpid()
    mem_consumption.activate_cooldown("after model compilation")
    iter_timestamp = model_utils.init_timestamp(num_iters, image_list, prompt_idx_list)
    for num, p_idx, prompt in prompter.iter_schedule(num_iters):
        mem_consumption.update_marker(f"step-{num}-{p_idx}")
        prefix = prompter.get_prefix(num, p_idx)
        # introduce_in_stdout prints prompt['prompt'] (the image path) on
        # num==0 and always logs repr(prompt) which includes probed dimensions.
        prompt.introduce_in_stdout(num, prefix)
        if num == 0 and args["output_dir"] is not None:
            llm_bench_utils.output_file.output_image_input_text(str(prompt['prompt']), args, p_idx, None, proc_id)
        iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
        run_ldm_super_resolution(prompt, num, pipe, args, framework, iter_data_list, p_idx, tm_list, proc_id, mem_consumption)
        if iter_data_list:
            iter_data_list[-1]["prompt_repr"] = repr(prompt)
        iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
        tm_list.clear()
        log.info(
            f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, "
            f"end: {iter_timestamp[num][p_idx]['end']}"
        )
    metrics_print.print_average(iter_data_list, prompt_idx_list, 1, False)
    return iter_data_list, pretrain_time, iter_timestamp
