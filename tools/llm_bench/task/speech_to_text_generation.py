# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import numpy as np
from pathlib import Path
import hashlib
import logging as log
import llm_bench_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
import llm_bench_utils.parse_json_data as parse_json_data

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}


def run_speech_2_txt_generation(raw_speech, pipe, args, num, md5_list, speech_id,
                                iter_data_list, mem_consumption, processor):
    result_md5_list = []
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    start = time.perf_counter()
    result_text = pipe.generate(
        raw_speech,
        max_new_tokens=1000,
        # 'task' and 'language' parameters are supported for multilingual models only
        language="<|en|>",
        task="transcribe",
        return_timestamps=True
    )
    end = time.perf_counter()
    tm_list = np.array(result_text.perf_metrics.raw_metrics.m_durations) / 1000 / 1000
    result_text = result_text.texts[0]

    generation_time = end - start
    out_data = processor.tokenizer(result_text, return_tensors='pt')
    out_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    out_tokens = out_data['input_ids'] if 'input_ids' in out_data else out_data
    out_token_size = out_tokens[0].numel()

    result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {speech_id : result_md5_list}
    else:
        md5_list[num][speech_id] = result_md5_list
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        out_size=out_token_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=speech_id,
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
        prompt_idx=speech_id
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
    ov_model, processor, pretrain_time = FW_UTILS[framework].create_genai_speech_2_txt_model(model_path, device, **args)
    pipe = ov_model
    md5_list = {num : {} for num in range(num_iters + 1)}
    for num in range(num_iters + 1):
        for idx, speech_file in enumerate(speech_list):
            raw_speech = model_utils.read_wav(speech_file['media'], processor.feature_extractor.sampling_rate)
            run_speech_2_txt_generation(raw_speech, pipe, args, num, md5_list, speech_idx_list[idx],
                                        iter_data_list, mem_consumption, processor)
    metrics_print.print_average(iter_data_list, speech_idx_list, 1, True)

    return iter_data_list, pretrain_time


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
