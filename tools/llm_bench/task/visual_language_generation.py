# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import datetime
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
import numpy as np
import hashlib
from transformers import set_seed
import llm_bench_utils.output_file
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import extract_prompt_data
from llm_bench_utils.prompt_utils import get_vlm_prompt


DEFAULT_OUTPUT_TOKEN_SIZE = 512
FW_UTILS = {
    'pt': llm_bench_utils.pt_utils,
    'ov': llm_bench_utils.ov_utils
}


def run_visual_language_generation_optimum(
        inputs, num, model, processor, args, iter_data_list, md5_list, prompt_index,
        bench_hook, model_precision, proc_id, mem_consumption):
    from optimum.intel.utils.import_utils import is_transformers_version
    set_seed(args['seed'])
    if args['batch_size'] != 1:
        log.warning("Only batch size 1 available for benchmarking")
        args["batch_size"] = 1

    decim_frames = args["video_frames"]
    prompts, images, videos = extract_prompt_data(inputs, decim_frames, False)
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(prompts):
            llm_bench_utils.output_file.output_input_text(
                in_text, args, model_precision,
                prompt_index, bs_index, proc_id)
    tok_encode_start = time.perf_counter()

    prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
    log.info(f'{prefix}[P{prompt_index}] Input image nums: {len(images)}')
    log.info(f'{prefix}[P{prompt_index}] Input video nums: {len(videos)}')
    input_data = model.preprocess_inputs(image=images[0] if images else None,
                                         video=videos[0] if videos else None,
                                         text=prompts[0], **processor)

    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
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
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    additional_args = {}
    if is_transformers_version(">=", "4.51"):
        additional_args["use_model_defaults"] = False
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
            do_sample=False,
            **additional_args
        )
    else:
        result = model.generate(
            **input_data,
            max_new_tokens=int(max_gen_tokens),
            num_beams=args['num_beams'],
            use_cache=True,
            do_sample=False,
            **additional_args
        )
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()

    generation_time = end - start
    tok_decode_start = time.perf_counter()
    generated_text = processor["tokenizer"].batch_decode(result[:, input_data["input_ids"].shape[1]:], skip_special_tokens=True)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_token_size = len(result[bs_idx]) - input_data["input_ids"][bs_idx].numel()
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
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
    tm_list = []
    tm_infer_list = []
    tm_mm_embeddings = ""
    if bench_hook is not None:
        tm_list = bench_hook.get_time_list()
        tm_mm_embeddings = np.mean(bench_hook.get_mm_embeddings_time_list()) * 1000 * 1000
        log.debug('latency of all tokens:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
        tm_infer_list = bench_hook.get_time_infer_list()
        log.debug('latency of all infers:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
        if args['num_beams'] == 1 and generated_token_size != len(tm_infer_list):
            log.warning(f'Output token size({generated_token_size}) is not equal to infer count({len(tm_infer_list)})')
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_infer_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time),
        mm_embeddings_preparation_time=tm_mm_embeddings
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()
        bench_hook.clear_mm_embeddins_time_list()


def run_visual_language_generation_genai(
        inputs, num, model, processor, args, iter_data_list, md5_list,
        prompt_index, streamer, model_precision, proc_id, mem_consumption):
    if args['batch_size'] != 1:
        log.warning("Only batch size 1 available for benchmarking")
        args["batch_size"] = 1

    decim_frames = args["video_frames"]
    prompts, images, videos = extract_prompt_data(inputs, decim_frames, True)
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(prompts):
            llm_bench_utils.output_file.output_input_text(
                in_text, args, model_precision,
                prompt_index, bs_index, proc_id)

    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    gen_config = model.get_generation_config()
    gen_config.max_new_tokens = max_gen_tokens
    gen_config.num_beams = args["num_beams"]
    gen_config.do_sample = False
    gen_config.ignore_eos = True
    if args["pruning_ratio"] is not None:
        gen_config.pruning_ratio = args["pruning_ratio"]
    if args["relevance_weight"] is not None:
        gen_config.relevance_weight = args["relevance_weight"]
    kwargs = {}
    prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
    log.info(f'{prefix}[P{prompt_index}] Input image nums: {len(images)}')
    log.info(f'{prefix}[P{prompt_index}] Input video nums: {len(videos)}')

    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos

    start = time.perf_counter()
    generation_result = model.generate(prompts[0], generation_config=gen_config, **kwargs)
    end = time.perf_counter()
    generated_text = generation_result.texts
    perf_metrics = generation_result.perf_metrics
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()

    generation_time = end - start
    result_md5_list = []
    generated_text_len = perf_metrics.get_num_generated_tokens()
    if generated_text_len > max_gen_tokens:
        log.error('Output token size is over max output token size!')
    result_text = generated_text[0]
    if args["output_dir"] is not None:
        llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, 0, proc_id)
    result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = ""
    if generated_text_len > 0:
        per_token_time = generation_time * 1000 / (generated_text_len / args['batch_size'])
    else:
        log.warning("No generated tokens")
    first_token_time = (perf_metrics.get_ttft().mean - perf_metrics.raw_metrics.tokenization_durations[-1] / 1000)
    second_tokens_durations = (
        np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
        - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1])
    ).tolist()

    tm_list = np.array([first_token_time] + second_tokens_durations) / 1000
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    tokenization_time = (
        np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000,
        np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000
    )
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=args['batch_size'] * perf_metrics.get_num_input_tokens(),
        infer_count=len(tm_list),
        out_size=generated_text_len,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time,
        mm_embeddings_preparation_time=perf_metrics.get_prepare_embeddings_duration().mean
    )
    iter_data_list.append(iter_data)
    inference_durations = np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list.tolist(),
        inference_durations.tolist(),
        warm_up=(num == 0),
        tokenization_time=tokenization_time,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)


def run_visual_language_generation_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    outs = FW_UTILS[framework].create_image_text_gen_model(model_path, device, mem_consumption, **args)
    model, processor, pretrain_time, bench_hook, use_genai = outs
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_image_text_list = get_vlm_prompt(args)
    if args['prompt_index'] is None:
        prompt_idx_list = list(range(0, len(input_image_text_list)))
        image_text_list = input_image_text_list
    else:
        prompt_idx_list = []
        image_text_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(input_image_text_list):
                image_text_list.append(input_image_text_list[i])
                prompt_idx_list.append(i)
    if len(input_image_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
             f'prompt nums: {len(image_text_list)}, prompt idx: {prompt_idx_list}')

    if use_genai:
        gen_fn = run_visual_language_generation_genai
    else:
        gen_fn = run_visual_language_generation_optimum

    proc_id = os.getpid()
    iter_timestamp = model_utils.init_timestamp(num_iters, image_text_list, prompt_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(image_text_list):
                p_idx = prompt_idx_list[idx]
                if num == 0:
                    prefix = f'[warm-up][P{p_idx}] Input text: {input_text}'
                    metrics_print.print_unicode(prefix, max_output=metrics_print.MAX_INPUT_TXT_IN_LOG)
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                gen_fn(
                    input_text, num, model, processor, args, iter_data_list, md5_list,
                    p_idx, bench_hook, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = f"[warm-up][P{p_idx}]" if num == 0 else f"[{num}][P{p_idx}]"
                log.info(f"{prefix} start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(image_text_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                if num == 0:
                    prefix = f'[warm-up][P{p_idx}] Input text: {input_text}'
                    metrics_print.print_unicode(prefix, max_output=metrics_print.MAX_INPUT_TXT_IN_LOG)
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                gen_fn(
                    input_text, num, model, processor, args, iter_data_list, md5_list,
                    prompt_idx_list[idx], bench_hook, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = f"[warm-up][P{p_idx}]" if num == 0 else f"[{num}][P{p_idx}]"
                log.info(f"{prefix} start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], True)
    return iter_data_list, pretrain_time, iter_timestamp
