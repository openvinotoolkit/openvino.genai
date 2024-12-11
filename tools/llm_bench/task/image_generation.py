# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import datetime
from PIL import Image
import hashlib
import logging as log
from transformers import set_seed
from llm_bench_utils.hook_forward import StableDiffusionHook
import llm_bench_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
import llm_bench_utils.parse_json_data as parse_json_data

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_INFERENCE_STEPS = 20
LCM_DEFAULT_INFERENCE_STEPS = 4
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512

stable_diffusion_hook = StableDiffusionHook()


def collects_input_args(image_param, model_type, model_name):
    input_args = {}
    input_args["width"] = image_param.get('width', DEFAULT_IMAGE_WIDTH)
    input_args["height"] = image_param.get('height', DEFAULT_IMAGE_HEIGHT)
    input_args["num_inference_steps"] = image_param.get('steps', DEFAULT_INFERENCE_STEPS if 'lcm' not in model_name else LCM_DEFAULT_INFERENCE_STEPS)
    guidance_scale = image_param.get('guidance_scale', None)

    if guidance_scale is not None:
        input_args["guidance_scale"] = guidance_scale
    else:
        if 'turbo' in model_name:
            input_args["guidance_scale"] = 0.0

    return input_args


def run_image_generation(image_param, num, image_id, pipe, args, iter_data_list, proc_id, mem_consumption, callback=None):
    set_seed(args['seed'])
    input_text = image_param['prompt']
    input_args = collects_input_args(image_param, args['model_type'], args['model_name'])
    out_str = f"Input params: Batch_size={args['batch_size']}, " \
              f"steps={input_args['num_inference_steps']}, width={input_args['width']}, height={input_args['height']}"
    if 'guidance_scale' in input_args:
        out_str += f", guidance_scale={input_args['guidance_scale']}"
    log.info(f"[{'warm-up' if num == 0 else num}][P{image_id}]{out_str}")

    result_md5_list = []
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()

    input_text_list = [input_text] * args['batch_size']
    input_data = pipe.tokenizer(input_text, return_tensors='pt')
    input_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
    input_token_size = input_tokens[0].numel()
    if num == 0 and args["output_dir"] is not None:
        for bs_idx, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_image_input_text(in_text, args, image_id, bs_idx, proc_id)
    start = time.perf_counter()
    res = pipe(input_text_list, **input_args).images
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    for bs_idx in range(args['batch_size']):
        rslt_img_fn = llm_bench_utils.output_file.output_gen_image(res[bs_idx], args, image_id, num, bs_idx, proc_id, '.png')
        result_md5_list.append(hashlib.md5(Image.open(rslt_img_fn).tobytes(), usedforsecurity=False).hexdigest())
    generation_time = end - start
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=input_args["num_inference_steps"],
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
        stable_diffusion=stable_diffusion_hook,
        prompt_idx=image_id
    )
    metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn, prompt_idx=image_id)
    stable_diffusion_hook.clear_statistics()


def run_image_generation_genai(image_param, num, image_id, pipe, args, iter_data_list, proc_id, mem_consumption, callback=None):
    set_seed(args['seed'])
    input_text = image_param['prompt']
    input_args = collects_input_args(image_param, args['model_type'], args['model_name'])
    out_str = f"Input params: Batch_size={args['batch_size']}, " \
              f"steps={input_args['num_inference_steps']}, width={input_args['width']}, height={input_args['height']}"
    if 'guidance_scale' in input_args:
        out_str += f", guidance_scale={input_args['guidance_scale']}"
    log.info(f"[{'warm-up' if num == 0 else num}][P{image_id}] {out_str}")

    result_md5_list = []
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()

    input_text_list = [input_text] * args['batch_size']
    if num == 0 and args["output_dir"] is not None:
        for bs_idx, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_image_input_text(in_text, args, image_id, bs_idx, proc_id)
    callback.reset()
    start = time.perf_counter()
    res = pipe.generate(input_text, **input_args, callback=callback).data
    end = time.perf_counter()
    callback.duration = end - start
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    for bs_idx in range(args['batch_size']):
        image = Image.fromarray(res[bs_idx])
        rslt_img_fn = llm_bench_utils.output_file.output_gen_image(image, args, image_id, num, bs_idx, proc_id, '.png')
        result_md5_list.append(hashlib.md5(Image.open(rslt_img_fn).tobytes(), usedforsecurity=False).hexdigest())
    generation_time = end - start
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        infer_count=input_args["num_inference_steps"],
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
        stable_diffusion=callback,
        prompt_idx=image_id
    )
    metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn, prompt_idx=image_id)
    stable_diffusion_hook.clear_statistics()


def run_image_generation_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    pipe, pretrain_time, use_genai, callback = FW_UTILS[framework].create_image_gen_model(model_path, device, **args)
    iter_data_list = []
    input_image_list = get_image_prompt(args)
    if framework == "ov" and not use_genai:
        stable_diffusion_hook.new_text_encoder(pipe)
        stable_diffusion_hook.new_unet(pipe)
        stable_diffusion_hook.new_vae_decoder(pipe)

    if args['prompt_index'] is None:
        prompt_idx_list = [image_id for image_id, input_text in enumerate(input_image_list)]
        image_list = input_image_list
    else:
        prompt_idx_list = []
        image_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(input_image_list):
                image_list.append(input_image_list[i])
                prompt_idx_list.append(i)
    if len(image_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, prompt nums: {len(image_list)}, prompt idx: {prompt_idx_list}')

    if use_genai:
        image_gen_fn = run_image_generation_genai
    else:
        image_gen_fn = run_image_generation

    # if num_iters == 0, just output warm-up data
    proc_id = os.getpid()
    iter_timestamp = model_utils.init_timestamp(num_iters, image_list, prompt_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for image_id, image_param in enumerate(image_list):
                p_idx = prompt_idx_list[image_id]
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                image_gen_fn(image_param, num, prompt_idx_list[image_id], pipe, args, iter_data_list, proc_id, mem_consumption, callback)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for image_id, image_param in enumerate(image_list):
            p_idx = prompt_idx_list[image_id]
            for num in range(num_iters + 1):
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                image_gen_fn(image_param, num, p_idx, pipe, args, iter_data_list, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    if not use_genai:
        metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], False)
    return iter_data_list, pretrain_time, iter_timestamp


def get_image_prompt(args):
    input_image_list = []
    output_data_list, is_json_data = model_utils.get_param_from_file(args, 'prompt')
    if is_json_data is True:
        image_param_list = parse_json_data.parse_image_json_data(output_data_list)
        if len(image_param_list) > 0:
            for image_data in image_param_list:
                input_image_list.append(image_data)
    else:
        input_image_list.append({'prompt': output_data_list[0]})
    return input_image_list
