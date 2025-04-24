# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import datetime
import numpy as np
from pathlib import Path
import hashlib
import logging as log
import llm_bench_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
import llm_bench_utils.parse_json_data as parse_json_data
from llm_bench_utils.hook_forward_whisper import WhisperHook

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}
whisper_hook = WhisperHook()

DEFAULT_OUTPUT_TOKEN_SIZE = 1000


def run_speech_2_txt_generation(input_param, args, md5_list, iter_data_list):
    result_md5_list = []
    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    pipe = input_param['pipe']
    raw_speech = input_param['raw_speech']
    num = input_param['iter_idx']
    speech_id = input_param['speech_idx']
    mem_consumption = input_param['mem_consumption']
    processor = input_param['processor']
    use_genai = input_param['use_genai']
    speech_language = input_param['speech_param'].get('language', "<|en|>")
    ret_timestamps = input_param['speech_param'].get('timestamp', True)
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']

    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()
    if use_genai:
        start = time.perf_counter()
        result_text = pipe.generate(
            raw_speech,
            max_new_tokens=max_gen_tokens,
            # 'task' and 'language' parameters are supported for multilingual models only
            language=speech_language,
            task="translate",
            return_timestamps=ret_timestamps
        )
        end = time.perf_counter()
        perf_metrics = result_text.perf_metrics
        first_token_time = perf_metrics.get_ttft().mean
        second_tokens_durations = (
            np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
            - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1])
        ).tolist()
        tm_list = (np.array([first_token_time] + second_tokens_durations) / 1000).tolist()
        tm_infer_list = (np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000).tolist()
        result_text = result_text.texts[0]
    else:
        from optimum.intel.utils.import_utils import is_transformers_version
        additional_args = {}
        if is_transformers_version(">=", "4.51"):
            additional_args["use_model_defaults"] = False
        start = time.perf_counter()
        result_text = pipe(
            raw_speech,
            generate_kwargs={"task": 'translate', "language": speech_language, **additional_args},
            return_timestamps=ret_timestamps
        )["text"]
        end = time.perf_counter()
        tm_list = whisper_hook.get_time_list()
        tm_infer_list = whisper_hook.get_time_infer_list()
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    if tm_infer_list is not None:
        log.debug('latency of all infers:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
    generation_time = end - start
    out_data = processor.tokenizer(result_text, return_tensors='pt')
    out_tokens = out_data['input_ids'] if 'input_ids' in out_data else out_data
    out_token_size = out_tokens[0].numel()

    result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {speech_id : result_md5_list}
    else:
        md5_list[num][speech_id] = result_md5_list
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        out_size=out_token_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=speech_id,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        tms=tm_list,
        tms_infer=tm_infer_list,
        warm_up=(num == 0),
        prompt_idx=speech_id,
        whisper=whisper_hook
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][speech_id]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{speech_id}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=result_text, prompt_idx=speech_id)
            if num == 1:
                # if the device is CPU, throw exception
                if args['devices'].lower().startswith('cpu') is True:
                    assert (result_md5_list == prev_md5)
            else:
                # throw exception
                assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=result_text, prompt_idx=speech_id)
    if whisper_hook is not None:
        whisper_hook.clear_statistics()


def run_speech_2_txt_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    iter_data_list = []
    speech_file_list = get_speech_files(args)
    if args['prompt_index'] is None:
        speech_idx_list = [prompt_idx for prompt_idx, speech_data in enumerate(speech_file_list)]
        speech_list = speech_file_list
    else:
        speech_idx_list = []
        speech_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(speech_file_list):
                speech_list.append(speech_file_list[i])
                speech_idx_list.append(i)
    if len(speech_list) == 0:
        raise RuntimeError('==Failure speech list is empty ==')
    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, speech file nums: {len(speech_file_list)}, speech idx: {speech_idx_list}')
    pipe, processor, pretrain_time, use_genai = FW_UTILS[framework].create_speech_2_txt_model(model_path, device, mem_consumption, **args)
    md5_list = {num : {} for num in range(num_iters + 1)}
    iter_timestamp = model_utils.init_timestamp(num_iters, speech_list, speech_idx_list)
    input_param = {
        'pipe': pipe,
        'mem_consumption': mem_consumption,
        'processor': processor,
        'use_genai': use_genai
    }
    if framework == "ov" and use_genai is False:
        whisper_hook.new_text_encoder(pipe)
        whisper_hook.new_text_encoder_request(pipe)
        whisper_hook.new_generate(pipe)
        whisper_hook.new_text_sample(pipe)
    for num in range(num_iters + 1):
        for idx, speech_param in enumerate(speech_list):
            p_idx = speech_idx_list[idx]
            raw_speech = model_utils.read_wav(speech_param['media'], processor.feature_extractor.sampling_rate)
            input_param['speech_idx'] = p_idx
            input_param['speech_param'] = speech_param
            input_param['iter_idx'] = num
            input_param['raw_speech'] = raw_speech
            iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
            run_speech_2_txt_generation(input_param, args, md5_list, iter_data_list)
            iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
            prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
            log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    metrics_print.print_average(iter_data_list, speech_idx_list, 1, True)

    return iter_data_list, pretrain_time, iter_timestamp


def get_speech_files(args):
    speech_file_list = []
    output_data_list, is_json_data = model_utils.get_param_from_file(args, 'media')
    if is_json_data is True:
        speech_param_list = parse_json_data.parse_speech_json_data(output_data_list)
        if len(speech_param_list) > 0:
            for speech_file in speech_param_list:
                if args['prompt_file'] is not None and len(args['prompt_file']) > 0:
                    speech_file['media'] = os.path.join(os.path.dirname(args['prompt_file'][0]), speech_file['media'].replace('./', ''))
                    speech_file['media'] = Path(speech_file['media'])
                speech_file_list.append(speech_file)
    else:
        speech_file_list.append({'media': output_data_list[0]})
    return speech_file_list
