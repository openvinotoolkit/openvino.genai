# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import time
from pathlib import Path
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils
import torch
import numpy as np
from openvino.runtime import get_version
import PIL
import hashlib
import llm_bench_utils.metrics_print
import llm_bench_utils.output_csv
import traceback
from transformers import set_seed
from PIL import Image
from llm_bench_utils.memory_profile import MemConsumption
from llm_bench_utils.hook_forward import StableDiffusionHook
import llm_bench_utils.output_json
import llm_bench_utils.output_file

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_INFERENCE_STEPS = 20
LCM_DEFAULT_INFERENCE_STEPS = 4
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_SUPER_RESOLUTION_STEPS = 50
DEFAULT_SUPER_RESOLUTION_WIDTH = 128
DEFAULT_SUPER_RESOLUTION_HEIGHT = 128
DEFAULT_OUTPUT_TOKEN_SIZE = 512

mem_consumption = MemConsumption()
stable_diffusion_hook = StableDiffusionHook()


def gen_iterate_data(
    iter_idx='',
    in_size='',
    infer_count='',
    out_size='',
    gen_time='',
    latency='',
    res_md5='',
    max_rss_mem='',
    max_shared_mem='',
    max_uss_mem='',
    prompt_idx='',
    tokenization_time=[],
):
    iter_data = {}
    iter_data['iteration'] = iter_idx
    iter_data['input_size'] = in_size
    iter_data['infer_count'] = infer_count
    iter_data['output_size'] = out_size
    iter_data['generation_time'] = gen_time
    iter_data['latency'] = latency
    iter_data['result_md5'] = res_md5
    iter_data['first_token_latency'] = ''
    iter_data['other_tokens_avg_latency'] = ''
    iter_data['first_token_infer_latency'] = ''
    iter_data['other_tokens_infer_avg_latency'] = ''
    iter_data['max_rss_mem_consumption'] = max_rss_mem
    iter_data['max_shared_mem_consumption'] = max_shared_mem
    iter_data['max_uss_mem_consumption'] = max_uss_mem
    iter_data['prompt_idx'] = prompt_idx
    iter_data['tokenization_time'] = tokenization_time[0] if len(tokenization_time) > 0 else ''
    iter_data['detokenization_time'] = tokenization_time[1] if len(tokenization_time) > 1 else ''
    return iter_data


def run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_index, bench_hook, model_precision, proc_id):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    tok_encode_start = time.perf_counter()
    input_data = tokenizer(input_text_list, return_tensors='pt')
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    input_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
    input_token_size = input_tokens[0].numel()
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)

    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    start = time.perf_counter()
    if args['infer_count'] is not None and args['end_token_stopping'] is False:
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
        result = model.generate(
            **input_data,
            max_new_tokens=int(max_gen_tokens),
            num_beams=args['num_beams'],
            use_cache=True,
            eos_token_id=None,
            do_sample=False
        )
    else:
        result = model.generate(
            **input_data,
            max_new_tokens=int(max_gen_tokens),
            num_beams=args['num_beams'],
            use_cache=True,
            do_sample=False
        )
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    generation_time = end - start
    tok_decode_start = time.perf_counter()
    generated_text = tokenizer.batch_decode(result)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        if 'sum' not in args['model_name'] and result[bs_idx][:input_token_size].equal(input_tokens[bs_idx]):
            generated_token_size = len(result[bs_idx]) - input_tokens[bs_idx].numel()
        else:
            generated_token_size = len(result[bs_idx])
        # Encoder-decoder models expect the `decoder_input_ids` to start with a special token
        # When counting the output length, subtract 1. The last token does not participate in inference.
        if model.config.is_encoder_decoder and result[bs_idx][0] == model.config.decoder_start_token_id:
            generated_token_size = generated_token_size - 1
        num_tokens += generated_token_size
        if generated_token_size > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    tm_list = []
    tm_infer_list = []
    if bench_hook is not None:
        tm_list = bench_hook.get_time_list()
        log.debug('latency of all tokens:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
        tm_infer_list = bench_hook.get_time_infer_list()
        log.debug('latency of all infers:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
        if args['num_beams'] == 1 and generated_token_size != len(tm_infer_list):
            log.warning(f'Output token size({generated_token_size}) is not equal to infer count({len(tm_infer_list)})')
    iter_data = gen_iterate_data(
        num,
        input_token_size * args['batch_size'],
        len(tm_infer_list),
        num_tokens,
        generation_time,
        per_token_time,
        result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
    )
    iter_data_list.append(iter_data)
    llm_bench_utils.metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size']
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            llm_bench_utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0])
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        llm_bench_utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0])
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()


def run_text_generation_genai(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_index, streamer, model_precision, proc_id):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    pt_inputs = tokenizer(input_text_list, return_tensors="pt")
    input_token_size = pt_inputs.input_ids.shape[1]
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)

    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    start = time.perf_counter()
    generation_result = model.generate(input_text_list, max_new_tokens=max_gen_tokens, num_beams=args["num_beams"])
    end = time.perf_counter()
    generated_text = generation_result.texts
    perf_metrics = generation_result.perf_metrics

    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    generation_time = end - start
    generated_tokens = [tokenizer(text).input_ids for text in generated_text]
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_text_len = len(generated_tokens[bs_idx])
        num_tokens += generated_text_len
        if generated_text_len > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    tm_list = np.array(perf_metrics.raw_metrics.m_durations) / 1000 / 1000
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    tokenization_time = (
        np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000,
        np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000
    )
    iter_data = gen_iterate_data(
        num,
        input_token_size * args['batch_size'],
        len(tm_list),
        num_tokens,
        generation_time,
        per_token_time,
        result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time
    )
    iter_data_list.append(iter_data)
    llm_bench_utils.metrics_print.print_metrics(
        num,
        iter_data,
        tm_list.tolist(),
        [],
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        tokenization_time=tokenization_time,
        batch_size=args['batch_size']
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            llm_bench_utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0])
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        llm_bench_utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0])


def run_text_generation_genai_with_stream(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_index, streamer, model_precision, proc_id):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    pt_inputs = tokenizer(input_text_list, return_tensors="pt")
    input_token_size = pt_inputs.input_ids.shape[1]
    pipe_tokenizer = model.get_tokenizer()
    tok_encode_start = time.perf_counter()
    input_data = pipe_tokenizer.encode(input_text_list)
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    streamer.reset()
    start = time.perf_counter()
    generated_tokens = model.generate(input_data, max_new_tokens=max_gen_tokens, num_beams=args["num_beams"], streamer=streamer).tokens
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    generation_time = end - start
    tok_decode_start = time.perf_counter()
    generated_text = pipe_tokenizer.decode(generated_tokens)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_text_len = len(generated_tokens[bs_idx])
        num_tokens += generated_text_len
        if generated_text_len > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    tm_list = streamer.get_time_list()
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    iter_data = gen_iterate_data(
        num,
        input_token_size * args['batch_size'],
        len(tm_list),
        num_tokens,
        generation_time,
        per_token_time,
        result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
    )
    iter_data_list.append(iter_data)
    llm_bench_utils.metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        [],
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size']
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            llm_bench_utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0])
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        llm_bench_utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0])
    streamer.reset()


def run_text_generation_benchmark(model_path, framework, device, args, num_iters):
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_gen_model(model_path, device, **args)
    model_precision = llm_bench_utils.model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_text_list = llm_bench_utils.model_utils.get_prompts(args)
    if args['prompt_index'] is None:
        prompt_idx_list = [prompt_idx for prompt_idx, input_text in enumerate(input_text_list)]
        text_list = input_text_list
    else:
        prompt_idx_list = []
        text_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(input_text_list):
                text_list.append(input_text_list[i])
                prompt_idx_list.append(i)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, prompt nums: {len(text_list)}, '
             f"prompt idx: {prompt_idx_list}, num_beams: {args['num_beams']}")

    # if num_iters == 0, just output warm-up data
    if not use_genai:
        text_gen_fn = run_text_generation
    elif bench_hook is not None:
        text_gen_fn = run_text_generation_genai_with_stream
    else:
        text_gen_fn = run_text_generation_genai
    proc_id = os.getpid()
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                if num == 0:
                    log.info(f'[warm-up] Input text: {input_text}')
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_idx_list[idx], bench_hook, model_precision, proc_id)
    else:
        for idx, input_text in enumerate(text_list):
            for num in range(num_iters + 1):
                if num == 0:
                    log.info(f'[warm-up] Input text: {input_text}')
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_idx_list[idx], bench_hook, model_precision, proc_id)

    llm_bench_utils.metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], True)
    return iter_data_list, pretrain_time


def run_image_generation(image_param, num, image_id, pipe, args, iter_data_list, proc_id):
    set_seed(args['seed'])
    input_text = image_param['prompt']
    image_width = image_param.get('width', DEFAULT_IMAGE_WIDTH)
    image_height = image_param.get('height', DEFAULT_IMAGE_HEIGHT)
    nsteps = image_param.get('steps', DEFAULT_INFERENCE_STEPS if 'lcm' not in args["model_name"] else LCM_DEFAULT_INFERENCE_STEPS)
    guidance_scale = image_param.get('guidance_scale', None)
    log.info(
        f"[{'warm-up' if num == 0 else num}] Input params: Batch_size={args['batch_size']}, "
        f'steps={nsteps}, width={image_width}, height={image_height}, guidance_scale={guidance_scale}'
    )
    result_md5_list = []
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    additional_args = {}
    if guidance_scale is not None:
        additional_args["guidance_scale"] = guidance_scale
    else:
        if 'lcm-sdxl' in args['model_type']:
            additional_args["guidance_scale"] = 1.0
        if 'turbo' in args['model_name']:
            additional_args["guidance_scale"] = 0.0
    input_text_list = [input_text] * args['batch_size']
    if num == 0 and args["output_dir"] is not None:
        for bs_idx, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_image_input_text(in_text, args, image_id, bs_idx, proc_id)
    start = time.perf_counter()
    res = pipe(input_text_list, num_inference_steps=nsteps, height=image_height, width=image_width, **additional_args).images
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    for bs_idx in range(args['batch_size']):
        rslt_img_fn = llm_bench_utils.output_file.output_gen_image(res[bs_idx], args, image_id, num, bs_idx, proc_id, '.png')
        result_md5_list.append(hashlib.md5(Image.open(rslt_img_fn).tobytes(), usedforsecurity=False).hexdigest())
    generation_time = end - start
    iter_data = gen_iterate_data(
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
    llm_bench_utils.metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        stable_diffusion=stable_diffusion_hook
    )
    llm_bench_utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn)
    stable_diffusion_hook.clear_statistics()


def run_image_generation_benchmark(model_path, framework, device, args, num_iters):
    if args['genai']:
        log.warning("GenAI pipeline is not supported for this task. Switched on default benchmarking")
    pipe, pretrain_time = FW_UTILS[framework].create_image_gen_model(model_path, device, **args)
    iter_data_list = []
    input_image_list = llm_bench_utils.model_utils.get_image_param_from_prompt_file(args)
    if framework == "ov":
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

    # if num_iters == 0, just output warm-up data
    proc_id = os.getpid()
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for image_id, image_param in enumerate(image_list):
                run_image_generation(image_param, num, prompt_idx_list[image_id], pipe, args, iter_data_list, proc_id)
    else:
        for image_id, image_param in enumerate(image_list):
            for num in range(num_iters + 1):
                run_image_generation(image_param, num, prompt_idx_list[image_id], pipe, args, iter_data_list, proc_id)

    llm_bench_utils.metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], False)
    return iter_data_list, pretrain_time


def run_ldm_super_resolution(img, num, pipe, args, framework, iter_data_list, image_id, tm_list, proc_id):
    set_seed(args['seed'])
    nsteps = img.get('steps', DEFAULT_SUPER_RESOLUTION_STEPS)
    resize_image_width = img.get('width', DEFAULT_SUPER_RESOLUTION_WIDTH)
    resize_image_height = img.get('height', DEFAULT_SUPER_RESOLUTION_HEIGHT)
    log.info(
        f"[{'warm-up' if num == 0 else num}] Input params: steps={nsteps}, "
        f'resize_width={resize_image_width}, resize_height={resize_image_height}'
    )
    low_res_img = PIL.Image.open(img['prompt']).convert('RGB')
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
    iter_data = gen_iterate_data(
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
    llm_bench_utils.metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption
    )
    llm_bench_utils.metrics_print.print_generated(num, warm_up=(num == 0), generated=rslt_img_fn)
    llm_bench_utils.metrics_print.print_ldm_unet_vqvae_infer_latency(num, iter_data, tm_list, warm_up=(num == 0))


def run_ldm_super_resolution_benchmark(model_path, framework, device, args, num_iters):
    if args["genai"]:
        log.warning("GenAI pipeline is not supported for this task. Switched on default benchmarking")
    pipe, pretrain_time = FW_UTILS[framework].create_ldm_super_resolution_model(model_path, device, **args)
    iter_data_list = []
    tm_list = []
    input_image_list = llm_bench_utils.model_utils.get_image_param_from_prompt_file(args)
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
            log.info(f"[{'warm-up' if num == 0 else num}] Input image={img['prompt']}")
            run_ldm_super_resolution(img, num, pipe, args, framework, iter_data_list, prompt_idx_list[image_id], tm_list, proc_id)
            tm_list.clear()
    llm_bench_utils.metrics_print.print_average(iter_data_list, prompt_idx_list, 1, False)

    return iter_data_list, pretrain_time


def num_iters_type(x):
    x = int(x)
    if x < 0:
        raise argparse.ArgumentTypeError('Minimum input value is 0')
    return x


def num_infer_count_type(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError('Minimum input value is 1')
    return x


def get_argprser():
    parser = argparse.ArgumentParser('LLM benchmarking tool', add_help=True, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='model folder including IR files or Pytorch files', required=TabError)
    parser.add_argument('-d', '--device', default='cpu', help='inference device')
    parser.add_argument('-r', '--report', help='report csv')
    parser.add_argument('-rj', '--report_json', help='report json')
    parser.add_argument('-f', '--framework', default='ov', help='framework')
    parser.add_argument('-p', '--prompt', default=None, help='one prompt')
    parser.add_argument('-pf', '--prompt_file', nargs='+', default=None,
                        help='Prompt file(s) in jsonl format. Multiple prompt files should be separated with space(s).')
    parser.add_argument('-pi', '--prompt_index', nargs='+', type=num_iters_type, default=None,
                        help='Run the specified prompt index. You can specify multiple prompt indexes, separated by spaces.')
    parser.add_argument(
        '-ic',
        '--infer_count',
        default=None,
        type=num_infer_count_type,
        help='set the output token size, the value must be greater than 0.'
    )
    parser.add_argument(
        '-n',
        '--num_iters',
        default=0,
        type=num_iters_type,
        help='number of benchmarking iterations, '
        'if the value is greater than 0, the average numbers exclude the first(0th) iteration,\n'
        'if the value equals 0 (default), execute the warm-up iteration(0th iteration).',
    )
    parser.add_argument('-i', '--images', default=None, help='test images for vision tasks. Can be directory or path to single image')
    parser.add_argument('-s', '--seed', type=int, default=42, required=False, help='specific random seed to generate fix result. Default 42.')
    parser.add_argument(
        '-lc',
        '--load_config',
        default=None,
        required=False,
        help='path to JSON file to load customized configurations.\n'
        'Example for OpenVINO: {\"INFERENCE_NUM_THREADS\":32,\"PERFORMANCE_HINT\":\"LATENCY\"}.\n'
        'Example for Pytorch: {\"PREC_BF16\":true}. Pytorch currently only supports bf16 settings.\n',
    )
    parser.add_argument(
        '-mc',
        '--memory_consumption',
        default=0,
        required=False,
        type=int,
        help='if the value is 1, output the maximum memory consumption in warm-up iterations. If the value is 2,'
        ' output the maximum memory consumption in all iterations.',
    )
    parser.add_argument('-bs', '--batch_size', type=int, default=1, required=False, help='Batch size value')
    parser.add_argument(
        '--fuse_decoding_strategy',
        action='store_true',
        help='Add decoding postprocessing for next token selection to the model as an extra ops. Original hf_model.generate function will be patched.',
    )
    parser.add_argument(
        '--save_prepared_model',
        default=None,
        help='Path to .xml file to save IR used for inference with all pre-/post processing included',
    )
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams in the decoding strategy, activates beam_search if greater than 1')
    parser.add_argument(
        '--torch_compile_backend',
        default='openvino',
        required=False,
        help='Enables running the torch.compile() with specified backend: pytorch or openvino (default)',
    )
    parser.add_argument(
        '--torch_compile_dynamic',
        action='store_true',
        help='Enables dynamic shape tracking for torch.compile()',
    )
    parser.add_argument(
        '--torch_compile_options',
        default=None,
        required=False,
        help='Options for torch.compile() in JSON format',
    )
    parser.add_argument(
        '--torch_compile_input_module',
        default=None,
        required=False,
        help='Specifies the module to decorate with torch.compile(). By default, parent module will be decorated.',
    )
    parser.add_argument(
        '--convert_tokenizer', action='store_true', help='Convert tokenizer to OpenVINO format'
    )
    parser.add_argument(
        '--subsequent',
        action='store_true',
        help='if the value is True, input prompts are processed in subsequent manner'
        'if the value is False (default), input prompts are processed in interleave manner'
    )
    parser.add_argument('-od', '--output_dir', help='Save the input text and generated text, images to files')
    llm_bench_utils.model_utils.add_stateful_model_arguments(parser)
    parser.add_argument("--genai", action="store_true", help="Use OpenVINO GenAI optimized pipelines for benchmarking")
    parser.add_argument("--use_cb", action="store_true", help="Use Continuous Batching inference mode")
    parser.add_argument("--cb_config", required=False, default=None, help="Path to file with Continuous Batching Scheduler settings or dict")
    parser.add_argument(
        '--end_token_stopping',
        action='store_true',
        help='Stop the generation even output token size does not achieve infer_count or max token size ({DEFAULT_OUTPUT_TOKEN_SIZE}}).'
    )

    return parser.parse_args()


CASE_TO_BENCH = {
    'text_gen': run_text_generation_benchmark,
    'image_gen': run_image_generation_benchmark,
    'code_gen': run_text_generation_benchmark,
    'ldm_super_resolution': run_ldm_super_resolution_benchmark,
}


def main():
    logging_kwargs = {"encoding": "utf-8"} if sys.version_info[1] > 8 else {}
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=os.environ.get("LOGLEVEL", log.INFO), stream=sys.stdout, **logging_kwargs)
    args = get_argprser()
    model_path, framework, model_args, model_name = llm_bench_utils.model_utils.analyze_args(args)

    # Set the device for running OpenVINO backend for torch.compile()
    if model_args['torch_compile_backend']:
        ov_torch_backend_device = str(args.device)
        os.putenv('OPENVINO_TORCH_BACKEND_DEVICE', ov_torch_backend_device.upper())
        os.system('echo [ INFO ] OPENVINO_TORCH_BACKEND_DEVICE=$OPENVINO_TORCH_BACKEND_DEVICE')

    out_str = 'Model path={}'.format(model_path)
    if framework == 'ov':
        out_str += ', openvino runtime version: {}'.format(get_version())
        if model_args['config'].get('PREC_BF16') and model_args['config']['PREC_BF16'] is True:
            log.warning('[Warning] Param bf16/prec_bf16 only work for framework pt. It will be disabled.')
        if 'cpu' in args.device.lower():
            env_omp = os.getenv('OMP_WAIT_POLICY')
            if env_omp is None or env_omp != 'PASSIVE':
                log.warning("It is recommended to set the environment variable OMP_WAIT_POLICY to PASSIVE, "
                            "so that OpenVINO inference can use all CPU resources without waiting.")
            original_torch_thread_nums = torch.get_num_threads()
            if model_args['num_beams'] > 1:
                torch.set_num_threads(int(original_torch_thread_nums / 2))
            else:
                torch.set_num_threads(1)
            log.info(f"The num_beams is {model_args['num_beams']}, update Torch thread num from "
                     f'{original_torch_thread_nums} to {torch.get_num_threads()}, avoid to use the CPU cores for OpenVINO inference.')
    log.info(out_str)
    if args.memory_consumption:
        mem_consumption.start_collect_mem_consumption_thread()
    try:
        iter_data_list, pretrain_time = CASE_TO_BENCH[model_args['use_case']](model_path, framework, args.device, model_args, args.num_iters)
        if args.report is not None or args.report_json is not None:
            model_precision = ''
            if framework == 'ov':
                ir_conversion_frontend = llm_bench_utils.model_utils.get_ir_conversion_frontend(model_name, model_path.parts)
                if ir_conversion_frontend != '':
                    framework = framework + '(' + ir_conversion_frontend + ')'
                model_precision = llm_bench_utils.model_utils.get_model_precision(model_path.parts)
            if args.report is not None:
                llm_bench_utils.output_csv.write_result(
                    args.report,
                    model_name,
                    framework,
                    args.device,
                    model_args,
                    iter_data_list,
                    pretrain_time,
                    model_precision,
                )
            if args.report_json is not None:
                llm_bench_utils.output_json.write_result(
                    args.report_json,
                    model_name,
                    framework,
                    args.device,
                    model_args,
                    iter_data_list,
                    pretrain_time,
                    model_precision,
                )
    except Exception:
        log.error('An exception occurred')
        log.info(traceback.format_exc())
        exit(1)
    finally:
        if args.memory_consumption:
            mem_consumption.end_collect_mem_consumption_thread()


if __name__ == '__main__':
    main()
