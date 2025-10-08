# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
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
from transformers.image_utils import load_image
import openvino as ov
import numpy as np

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_INFERENCE_STEPS = 20
LCM_DEFAULT_INFERENCE_STEPS = 4
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512

stable_diffusion_hook = StableDiffusionHook()


def read_image(image_path: str, ov_tensor=True):
    pil_image = load_image(image_path).convert("RGB")
    if ov_tensor:
        image_data = np.array(pil_image)[None]
        return ov.Tensor(image_data)
    return pil_image


def collects_input_args(image_param, model_name, infer_count=None, height=None, width=None, callback=None, image_as_ov_tensor=True):
    input_args = {}
    input_args["width"] = image_param.get('width', width or DEFAULT_IMAGE_WIDTH)
    input_args["height"] = image_param.get('height', height or DEFAULT_IMAGE_HEIGHT)
    if infer_count is None:
        input_args["num_inference_steps"] = image_param.get('steps', DEFAULT_INFERENCE_STEPS if 'lcm' not in model_name else LCM_DEFAULT_INFERENCE_STEPS)
    else:
        input_args["num_inference_steps"] = infer_count
    guidance_scale = image_param.get('guidance_scale', None)

    if guidance_scale is not None:
        input_args["guidance_scale"] = guidance_scale
    else:
        if 'turbo' in model_name:
            input_args["guidance_scale"] = 0.0
    if callback is not None:
        from openvino import get_version
        from packaging.version import parse

        version = model_utils.get_version_in_format_to_pars(get_version())
        is_callback_supported = parse(version) >= parse("2025.0.0")
        if is_callback_supported:
            input_args["callback"] = callback

    if image_param.get('media'):
        images = image_param['media'] if isinstance(image_param['media'], (list, tuple)) else [image_param['media']]
        initial_images_list = []
        for img in images:
            initial_images_list.append(read_image(img, image_as_ov_tensor))
        input_args["image"] = initial_images_list[0]

    if image_param.get('mask_image', None):
        input_args["mask_image"] = read_image(image_param.get('mask_image'), image_as_ov_tensor)

    if image_param.get('strength'):
        input_args["strength"] = image_param['strength']

    return input_args


def run_image_generation(image_param, num, image_id, pipe, args, iter_data_list, proc_id, mem_consumption, callback=None):
    set_seed(args['seed'])
    input_text = image_param['prompt']
    input_args = collects_input_args(image_param, args['model_name'], args["num_steps"],
                                     args.get("height"), args.get("width"), image_as_ov_tensor=False)
    out_str = f"Input params: Batch_size={args['batch_size']}, " \
              f"steps={input_args['num_inference_steps']}, width={input_args['width']}, height={input_args['height']}"
    if 'guidance_scale' in input_args:
        out_str += f", guidance_scale={input_args['guidance_scale']}"
    log.info(f"[{'warm-up' if num == 0 else num}][P{image_id}]{out_str}")

    result_md5_list = []
    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()

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
    res = pipe(input_text_list, **input_args, num_images_per_prompt=args['batch_size']).images
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()
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
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=image_id,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        stable_diffusion=stable_diffusion_hook,
        prompt_idx=image_id
    )
    metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn, prompt_idx=image_id)
    stable_diffusion_hook.clear_statistics()


def run_image_generation_genai(image_param, num, image_id, pipe, args, iter_data_list, proc_id, mem_consumption, callback=None):
    set_seed(args['seed'])
    input_text = image_param['prompt']
    input_token_size = callback.orig_tokenizer(input_text, return_tensors="pt").input_ids.numel()
    input_args = collects_input_args(image_param, args['model_name'], args["num_steps"], args.get("height"), args.get("width"), callback)
    out_str = f"Input params: Batch_size={args['batch_size']}, " \
              f"steps={input_args['num_inference_steps']}, width={input_args['width']}, height={input_args['height']}"
    if 'guidance_scale' in input_args:
        out_str += f", guidance_scale={input_args['guidance_scale']}"
    log.info(f"[{'warm-up' if num == 0 else num}][P{image_id}] {out_str}")

    if args.get("static_reshape", False) and 'guidance_scale' in input_args:
        reshaped_gs = pipe.get_generation_config().guidance_scale
        new_gs = input_args['guidance_scale']
        if new_gs != reshaped_gs:
            log.warning(f"image generation pipeline was reshaped with guidance_scale={reshaped_gs}, but is being passed into generate() as {new_gs}")

    result_md5_list = []
    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()

    input_text_list = [input_text] * args['batch_size']
    if num == 0 and args["output_dir"] is not None:
        for bs_idx, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_image_input_text(in_text, args, image_id, bs_idx, proc_id)
    callback.reset()

    if (args['empty_lora'] and (pipe.get_generation_config().adapters is not None)):
        import openvino_genai
        input_args['adapters'] = openvino_genai.AdapterConfig()

    start = time.perf_counter()
    res = pipe.generate(input_text, **input_args, num_images_per_prompt=args['batch_size']).data
    end = time.perf_counter()
    callback.duration = end - start

    performance_metrics = None
    if hasattr(pipe, 'get_performance_metrics'):
        performance_metrics = pipe.get_performance_metrics()
    elif "callback" in input_args:
        performance_metrics = callback

    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()
    for bs_idx in range(args['batch_size']):
        image = Image.fromarray(res[bs_idx])
        rslt_img_fn = llm_bench_utils.output_file.output_gen_image(image, args, image_id, num, bs_idx, proc_id, '.png')
        result_md5_list.append(hashlib.md5(Image.open(rslt_img_fn).tobytes(), usedforsecurity=False).hexdigest())
    generation_time = end - start
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=input_args["num_inference_steps"],
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=image_id,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        stable_diffusion=performance_metrics,
        prompt_idx=image_id
    )
    metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn, prompt_idx=image_id)
    stable_diffusion_hook.clear_statistics()


def run_image_generation_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    input_image_list = get_image_prompt(args)
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

    # If --static_reshape is specified, we need to get width, height, and guidance scale to drop into args
    # as genai's create_image_gen_model implementation will need those to reshape the pipeline before compile().
    if args.get("static_reshape", False):
        static_input_args = collects_input_args(image_list[0], args['model_name'], args["num_steps"],
                                                args.get("height"), args.get("width"), image_as_ov_tensor=False)
        args["height"] = static_input_args["height"]
        args["width"] = static_input_args["width"]
        if "guidance_scale" in static_input_args:
            args["guidance_scale"] = static_input_args["guidance_scale"]

    pipe, pretrain_time, use_genai, callback = FW_UTILS[framework].create_image_gen_model(model_path, device, mem_consumption, **args)
    iter_data_list = []

    if framework == "ov" and not use_genai:
        stable_diffusion_hook.new_text_encoder(pipe)
        stable_diffusion_hook.new_unet(pipe)
        stable_diffusion_hook.new_vae_decoder(pipe)

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

    metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], False)
    return iter_data_list, pretrain_time, iter_timestamp


def get_image_prompt(args):
    input_image_list = []

    input_key = ['prompt']
    if args.get("task") == args["use_case"].TASK["inpainting"]["name"] or ((args.get("media") or args.get("images")) and args.get("mask_image")):
        input_key = ['media', "mask_image", "prompt"]
    elif args.get("task") == args["use_case"].TASK["img2img"]["name"] or args.get("media") or args.get("images"):
        input_key = ['media', "prompt"]

    output_data_list, is_json_data = model_utils.get_param_from_file(args, input_key)
    if is_json_data is True:
        image_param_list = parse_json_data.parse_image_json_data(output_data_list)
        if len(image_param_list) > 0:
            for image_data in image_param_list:
                if args['prompt_file'] is not None and len(args['prompt_file']) > 0:
                    image_data['media'] = model_utils.resolve_media_file_path(image_data.get("media"), args['prompt_file'][0])
                    image_data['mask_image'] = model_utils.resolve_media_file_path(image_data.get("mask_image"), args['prompt_file'][0])
                input_image_list.append(image_data)
    else:
        input_image_list.append(output_data_list[0])
    return input_image_list
