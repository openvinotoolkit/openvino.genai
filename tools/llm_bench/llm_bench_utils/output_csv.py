# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import csv
import numpy as np
import copy
from pathlib import Path
import llm_bench_utils.output_json as output_json
from llm_bench_utils.memory_monitor import MemoryDataSummarizer, MemoryUnit


def output_comments(result, use_case, writer):
    for key in result.keys():
        result[key] = ''
    writer.writerow(result)

    comment_list = []
    if use_case == 'text_gen' or use_case == 'code_gen':
        comment_list.append('input_size: Input token size')
        comment_list.append('output_size: Text/Code generation models: generated text token size')
        comment_list.append("infer_count: Limit the Text/Code generation models' output token size")
        comment_list.append('latency: Text/Code generation models: ms/token. Output token size / generation time')
        comment_list.append('1st_latency: Text/Code generation models: First token latency')
        comment_list.append('2nd_avg_latency: Text/Code generation models: Other tokens (exclude first token) latency')
        comment_list.append('1st_infer_latency: Text/Code generation models: First inference latency')
        comment_list.append('2nd_infer_avg_latency: Text/Code generation models: Other inferences (exclude first inference) latency')
        comment_list.append('result_md5: MD5 of generated text')
        comment_list.append('prompt_idx: Index of prompts')
    elif use_case == 'image_gen':
        comment_list.append("infer_count: Tex2Image models' Inference(or Sampling) step size")
        comment_list.append('1st_latency: First step latency of unet')
        comment_list.append('2nd_avg_latency: Other steps latency of unet(exclude first step)')
        comment_list.append('1st_infer_latency: Same as 1st_latency')
        comment_list.append('2nd_infer_avg_latency: Same as 2nd_avg_latency')
        comment_list.append('prompt_idx: Index of prompts')
    elif use_case == 'ldm_super_resolution':
        comment_list.append("infer_count: Tex2Image models' Inference(or Sampling) step size")
        comment_list.append('1st_latency: First step latency of unet')
        comment_list.append('2nd_avg_latency: Other steps latency of unet(exclude first step)')
        comment_list.append('1st_infer_latency: Same as 1st_latency')
        comment_list.append('2nd_infer_avg_latency: Same as 2nd_avg_latency')
        comment_list.append('prompt_idx: Image Index')
    comment_list.append('tokenization_time: Tokenizer encode time')
    comment_list.append('detokenization_time: Tokenizer decode time')
    comment_list.append('pretrain_time: Total time of load model and compile model')
    comment_list.append('generation_time: Time for one interaction. (e.g. The duration of  answering one question or generating one picture)')
    comment_list.append('iteration=0: warm-up; iteration=avg: average (exclude warm-up);iteration=mini: minimum value (exclude warm-up);'
                        'iteration=median: median value (exclude warm-up);')
    comment_list.append(
        'max_rss_mem/max_sys_mem: max rss/system memory consumption during iteration;'
    )
    comment_list.append(
        'max_increase_rss_mem/max_increase_sys_mem: max increase of rss/system memory during iteration;'
    )
    comment_list.append(
        'initial_rss_mem/initial_sys_mem: rss/system memory state at start;'
    )
    comment_list.append(
        'compile_max_rss_mem/compile_max_sys_mem: max rss/system memory consumption on compilation phase;'
    )
    comment_list.append(
        'compile_max_increase_rss_mem/compile_max_increase_sys_mem: max increase of rss/system memory on compilation phase;'
    )

    for comments in comment_list:
        result['iteration'] = comments
        writer.writerow(result)


def output_avg_min_median(iter_data_list):
    prompt_idxs = []
    for iter_data in iter_data_list:
        prompt_idxs.append(iter_data['prompt_idx'])
    prompt_idxs = list(set(prompt_idxs))
    result = {}
    for prompt_idx in prompt_idxs:
        same_prompt_datas = []
        for iter_data in iter_data_list:
            if iter_data['prompt_idx'] == prompt_idx and iter_data['iteration'] > 0:
                same_prompt_datas.append(iter_data)
        key_word = ['input_size', 'infer_count', 'generation_time', 'output_size', 'latency', 'first_token_latency', 'other_tokens_avg_latency',
                    'first_token_infer_latency', 'other_tokens_infer_avg_latency', 'tokenization_time', 'detokenization_time']
        if len(same_prompt_datas) > 0:
            iters_idx = ['avg', 'mini', 'median']
            result[prompt_idx] = [copy.deepcopy(same_prompt_datas[0]) for i in range(3)]
            for i in range(len(iters_idx)):
                result[prompt_idx][i]['iteration'] = iters_idx[i]
            for key in key_word:
                values = []
                for prompt in same_prompt_datas:
                    if prompt[key] != '':
                        values.append(prompt[key])
                if len(values) > 0:
                    result[prompt_idx][0][key] = np.mean(values)
                    result[prompt_idx][1][key] = np.min(values)
                    result[prompt_idx][2][key] = np.median(values)
    return result


def gen_data_to_csv(result: dict, iter_data: dict, pretrain_time: int, iter_timestamp: dict,
                    memory_data_collector: MemoryDataSummarizer | None, mem_unit: MemoryUnit):
    generation_time = iter_data['generation_time']
    latency = iter_data['latency']
    first_latency = iter_data['first_token_latency']
    other_latency = iter_data['other_tokens_avg_latency']
    first_token_infer_latency = iter_data['first_token_infer_latency']
    other_token_infer_latency = iter_data['other_tokens_infer_avg_latency']
    rss_mem = iter_data['max_rss_mem_consumption']
    sys_mem = iter_data['max_sys_mem_consumption']
    rss_mem_increase = iter_data['max_rss_mem_increase']
    sys_mem_increase = iter_data['max_sys_mem_increase']
    token_time = iter_data['tokenization_time']
    detoken_time = iter_data['detokenization_time']
    result['iteration'] = str(iter_data['iteration'])
    result['pretrain_time(s)'] = pretrain_time
    result['input_size'] = iter_data['input_size']
    result['infer_count'] = iter_data['infer_count']
    result['generation_time(s)'] = round(generation_time, 5) if generation_time != '' else generation_time
    result['output_size'] = iter_data['output_size']
    result['latency(ms)'] = round(latency, 5) if latency != '' else latency
    result['result_md5'] = iter_data['result_md5']
    if first_latency < 0:
        result['1st_latency(ms)'] = 'NA'
    else:
        result['1st_latency(ms)'] = round(first_latency, 5) if first_latency != '' else first_latency
    if other_latency < 0:
        result['2nd_avg_latency(ms)'] = 'NA'
    else:
        result['2nd_avg_latency(ms)'] = round(other_latency, 5) if other_latency != '' else other_latency
    if first_token_infer_latency < 0:
        result['1st_infer_latency(ms)'] = 'NA'
    else:
        result['1st_infer_latency(ms)'] = round(first_token_infer_latency, 5) if first_token_infer_latency != '' else first_token_infer_latency
    if other_token_infer_latency < 0:
        result['2nd_infer_avg_latency(ms)'] = 'NA'
    else:
        result['2nd_infer_avg_latency(ms)'] = round(other_token_infer_latency, 5) if other_token_infer_latency != '' else other_token_infer_latency
    result[f'max_rss_mem({mem_unit.value})'] = round(rss_mem, 5) if rss_mem != '' else rss_mem
    result[f'max_sys_mem({mem_unit.value})'] = round(sys_mem, 5) if sys_mem != '' else sys_mem
    result[f'max_increase_rss_mem({mem_unit.value})'] = round(rss_mem_increase, 5) if rss_mem_increase != '' else rss_mem_increase
    result[f'max_increase_sys_mem({mem_unit.value})'] = round(sys_mem_increase, 5) if sys_mem_increase != '' else sys_mem_increase
    result['prompt_idx'] = iter_data['prompt_idx']
    result['tokenization_time'] = round(token_time, 5) if token_time != '' else token_time
    result['detokenization_time'] = round(detoken_time, 5) if detoken_time != '' else detoken_time
    result['start'], result['end'] = output_json.get_timestamp(iter_data['iteration'], iter_data['prompt_idx'], iter_timestamp)
    result = result | output_json.get_pre_gen_memory_data(memory_data_collector, print_unit=mem_unit)


def write_result(report_file, model, framework, device, model_args, iter_data_list, pretrain_time, model_precision, iter_timestamp, memory_data_collector):
    mem_unit = memory_data_collector.memory_monitor.memory_unit if memory_data_collector else MemoryDataSummarizer.DEF_MEM_UNIT
    header = [
        'iteration',
        'model',
        'framework',
        'device',
        'pretrain_time(s)',
        f'initial_sys_mem({mem_unit.value})',
        f'initial_rss_mem({mem_unit.value})',
        f'compile_max_rss_mem({mem_unit.value})',
        f'compile_max_sys_mem({mem_unit.value})',
        f'compile_max_increase_rss_mem({mem_unit.value})',
        f'compile_max_increase_sys_mem({mem_unit.value})',
        'input_size',
        'infer_count',
        'generation_time(s)',
        'output_size',
        'latency(ms)',
        '1st_latency(ms)',
        '2nd_avg_latency(ms)',
        'precision',
        f'max_rss_mem({mem_unit.value})',
        f'max_sys_mem({mem_unit.value})',
        f'max_increase_rss_mem({mem_unit.value})',
        f'max_increase_sys_mem({mem_unit.value})',
        'prompt_idx',
        '1st_infer_latency(ms)',
        '2nd_infer_avg_latency(ms)',
        'num_beams',
        'batch_size',
        'tokenization_time',
        'detokenization_time',
        'result_md5',
        'start',
        'end'
    ]
    out_file = Path(report_file)

    if len(iter_data_list) > 0:
        with open(out_file, 'w+', newline='') as f:
            writer = csv.DictWriter(f, header)
            writer.writeheader()
            result = {}
            result['model'] = model
            result['framework'] = framework
            result['device'] = device
            result['pretrain_time(s)'] = round(pretrain_time, 5)
            result['precision'] = model_precision
            result['num_beams'] = model_args['num_beams']
            result['batch_size'] = model_args['batch_size']
            for i in range(len(iter_data_list)):
                iter_data = iter_data_list[i]
                pre_time = '' if i > 0 else result['pretrain_time(s)']
                mem_data_collector = None if i > 0 else memory_data_collector
                gen_data_to_csv(result, iter_data, pre_time, iter_timestamp, mem_data_collector, mem_unit)
                writer.writerow(result)

            res_data = output_avg_min_median(iter_data_list)

            for key in res_data.keys():
                for data in res_data[key]:
                    gen_data_to_csv(result, data, '', iter_timestamp, None, mem_unit)
                    writer.writerow(result)
            output_comments(result, model_args['use_case'], writer)
