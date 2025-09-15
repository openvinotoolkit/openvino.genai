# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import hashlib
import datetime
import logging as log
import soundfile as sf
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
from llm_bench_utils.hook_forward import TTSHook
import openvino as ov
import llm_bench_utils.metrics_print as metrics_print
from transformers import set_seed
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import get_text_prompt

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}


def run_text_to_speech_generation_optimum(
    input_text, num, model, processor, vocoder, args, iter_data_list, md5_list, prompt_index, tts_hook, model_precision, proc_id, mem_consumption
):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    tok_encode_start = time.perf_counter()
    input_data = processor(text=input_text_list, return_tensors='pt', padding=True, truncation=True)
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    input_data.pop('token_type_ids', None)
    input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
    input_token_size = input_tokens[0].numel()
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        log.info(out_str)

    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()
    start = time.perf_counter()
    if vocoder:
        result = model.generate(input_tokens, speaker_embeddings=args.get('speaker_embeddings'), vocoder=vocoder)
    else:
        result = model.generate(input_tokens, speaker_embeddings=args.get('speaker_embeddings'))

    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()

    generation_time = end - start
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        speech = result.numpy()[bs_idx] if len(result.size()) > 1 else result.numpy()
        audio_file_path = llm_bench_utils.output_file.output_gen_audio(speech, args, prompt_index, num, bs_idx, proc_id, '.wav')
        data, _ = sf.read(audio_file_path)
        result_md5_list.append(hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        out_size=result.numel(),
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=[tok_encode_time]
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        warm_up=(num == 0),
        tokenization_time=[tok_encode_time],
        batch_size=args['batch_size'],
        prompt_idx=prompt_index,
        tts=tts_hook
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
    if tts_hook is not None:
        tts_hook.clear_statistics()


def run_text_to_speech_generation_genai(
    input_text, num, model, processor, vocoder, args, iter_data_list, md5_list, prompt_index, tts_hook, model_precision, proc_id, mem_consumption
):
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()

    input_data = processor(text=input_text)
    num_input_tokens = len(input_data['input_ids'])

    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(num_input_tokens, args['batch_size'])
        log.info(out_str)

    start = time.perf_counter()
    additional_args = {"speaker_embeddings": ov.Tensor(args['speaker_embeddings'].numpy())} if args.get('speaker_embeddings') is not None else {}
    generation_result = model.generate(input_text_list, **additional_args)
    end = time.perf_counter()

    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()

    generation_time = end - start

    perf_metrics = generation_result.perf_metrics
    tokenization_time = [perf_metrics.get_tokenization_duration().mean]

    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        speech = generation_result.speeches[bs_idx].data[0]
        audio_file_path = llm_bench_utils.output_file.output_gen_audio(speech, args, prompt_index, num, bs_idx, proc_id, '.wav')
        data, _ = sf.read(audio_file_path)
        result_md5_list.append(hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest())

    md5_list[num][prompt_index] = result_md5_list

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=num_input_tokens * args['batch_size'],
        out_size=perf_metrics.num_generated_samples,
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        tokenization_time=tokenization_time,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )
    log.debug(f'[{num}]Throughput: {perf_metrics.throughput.mean:.4f}')
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")


def run_text_2_speech_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    model, processor, vocoder, pretrain_time, use_genai = FW_UTILS[framework].create_text_2_speech_model(model_path, device, mem_consumption, **args)
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_text_list = get_text_prompt(args)
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

    tts_hook = None
    if framework == "ov" and not use_genai:
        tts_hook = TTSHook()
        tts_hook.new_encoder(model)
        tts_hook.new_decoder(model)
        tts_hook.new_postnet(model)
        tts_hook.new_vocoder(model)

    if use_genai:
        gen_fn = run_text_to_speech_generation_genai
    else:
        gen_fn = run_text_to_speech_generation_optimum

    proc_id = os.getpid()
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, prompt_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                p_idx = prompt_idx_list[idx]
                if num == 0:
                    metrics_print.print_unicode(f'[warm-up][P{p_idx}] Input text: {input_text}', f'[warm-up][P{p_idx}] Unable print input text')
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                gen_fn(input_text, num, model, processor, vocoder, args, iter_data_list, md5_list,
                       p_idx, tts_hook, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                if num == 0:
                    metrics_print.print_unicode(f'[warm-up][P{p_idx}] Input text: {input_text}', f'[warm-up][P{p_idx}] Unable print input text')
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                gen_fn(input_text, num, model, processor, vocoder, args, iter_data_list, md5_list,
                       prompt_idx_list[idx], tts_hook, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    return iter_data_list, pretrain_time, iter_timestamp
