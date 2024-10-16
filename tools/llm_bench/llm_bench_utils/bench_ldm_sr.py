# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
from pathlib import Path
from PIL import Image
import hashlib
import logging as log
from transformers import set_seed
import llm_bench_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data

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
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    start = time.perf_counter()
    res = pipe(low_res_img, num_inference_steps=nsteps, tm_list=tm_list)
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
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
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=image_id,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=image_id
    )
    metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn, prompt_idx=image_id)
    metrics_print.print_ldm_unet_vqvae_infer_latency(num, iter_data, tm_list, warm_up=(num == 0), prompt_idx=image_id)


def run_ldm_super_resolution_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    if args["genai"]:
        log.warning("GenAI pipeline is not supported for this task. Switched on default benchmarking")
    pipe, pretrain_time = FW_UTILS[framework].create_ldm_super_resolution_model(model_path, device, **args)
    iter_data_list = []
    json_data_list = []
    tm_list = []
    input_image_list = model_utils.get_image_param_from_prompt_file(args)
    if len(input_image_list) > 0:
        images = []
        for image in input_image_list:
            if args['prompt'] is None and args['prompt_file'] is None:
                raise RuntimeError('==Failure image is empty ==')
            elif args['prompt_file'] is not None and len(args['prompt_file']) > 0:
                image['prompt'] = os.path.join(os.path.dirname(args['prompt_file'][0]), image['prompt'].replace('./', ''))
            image['prompt'] = Path(image['prompt'])
            images.append(image)
    else:
        if args['images'] is not None:
            images = Path(args['images'])
            if images.is_dir():
                images = list(images.glob('*'))
            else:
                images = [images]
        else:
            raise RuntimeError('==Failure image is empty ==')

    prompt_idx_list = [image_id for image_id, image_param in enumerate(images)]
    if args['prompt_index'] is None:
        prompt_idx_list = [image_id for image_id, input_text in enumerate(images)]
        image_list = images
    else:
        prompt_idx_list = []
        image_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(images):
                image_list.append(images[i])
                prompt_idx_list.append(i)
    if len(image_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, prompt nums: {len(image_list)}, prompt idx: {prompt_idx_list}')

    # if num_iters == 0, just output warm-up data
    proc_id = os.getpid()
    for num in range(num_iters + 1):
        for image_id, img in enumerate(image_list):
            if num == 0:
                if args["output_dir"] is not None:
                    llm_bench_utils.output_file.output_image_input_text(str(img['prompt']), args, prompt_idx_list[image_id], None, proc_id)
            log.info(f"[{'warm-up' if num == 0 else num}][P{prompt_idx_list[image_id]}] Input image={img['prompt']}")
            run_ldm_super_resolution(img, num, pipe, args, framework, iter_data_list, prompt_idx_list[image_id], tm_list, proc_id, mem_consumption)
            tm_list.clear()
    metrics_print.print_average(iter_data_list, prompt_idx_list, 1, False)

    return iter_data_list, pretrain_time, json_data_list
