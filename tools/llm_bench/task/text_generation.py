# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
import numpy as np
import hashlib
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.output_csv
from transformers import set_seed
import llm_bench_utils.output_json
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_OUTPUT_TOKEN_SIZE = 512


def run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                        prompt_index, bench_hook, model_precision, proc_id, mem_consumption):
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
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
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
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        loop_idx='',
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_infer_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
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
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()


def run_text_generation_genai(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_index,
                              streamer, model_precision, proc_id, mem_consumption):
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
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
    tm_list = np.array(perf_metrics.raw_metrics.m_durations) / 1000 / 1000
    if args['batch_size'] > 1:
        tm_list = np.multiply(tm_list, args['batch_size'])
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    tokenization_time = (
        np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000,
        np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000
    )
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        loop_idx='',
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list.tolist(),
        [],
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
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
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)


def run_text_generation_genai_with_stream(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                                          prompt_index, streamer, model_precision, proc_id, mem_consumption):
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
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
    tm_list = streamer.get_time_list()
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        loop_idx='',
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        tms=tm_list,
        tms_infer=[],
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
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
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    streamer.reset()


def run_text_generation_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_gen_model(model_path, device, **args)
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_text_list = model_utils.get_prompts(args)
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
    log.info(f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
             f'prompt nums: {len(text_list)}, prompt idx: {prompt_idx_list}')

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
                    log.info(f'[warm-up][P{prompt_idx_list[idx]}] Input text: {input_text}')
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                            prompt_idx_list[idx], bench_hook, model_precision, proc_id, mem_consumption)
    else:
        for idx, input_text in enumerate(text_list):
            for num in range(num_iters + 1):
                if num == 0:
                    log.info(f'[warm-up][P{prompt_idx_list[idx]}] Input text: {input_text}')
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                            prompt_idx_list[idx], bench_hook, model_precision, proc_id, mem_consumption)

    metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], True)
    return iter_data_list, pretrain_time
