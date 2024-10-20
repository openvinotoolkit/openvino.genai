# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def gen_iterate_data(
    iter_idx='',
    loop_idx='',
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
    loop_data=None
):
    iter_data = {}
    iter_data['iteration'] = iter_idx
    iter_data['loop_idx'] = loop_idx
    iter_data['input_size'] = in_size
    iter_data['infer_count'] = infer_count
    iter_data['output_size'] = out_size
    iter_data['generation_time'] = gen_time
    iter_data['latency'] = latency
    iter_data['result_md5'] = res_md5
    iter_data['max_rss_mem_consumption'] = max_rss_mem
    iter_data['max_shared_mem_consumption'] = max_shared_mem
    iter_data['max_uss_mem_consumption'] = max_uss_mem
    iter_data['prompt_idx'] = prompt_idx
    iter_data['tokenization_time'] = tokenization_time[0] if len(tokenization_time) > 0 else ''
    iter_data['detokenization_time'] = tokenization_time[1] if len(tokenization_time) > 1 else ''

    if loop_data is not None:
        iter_data['enc_token_latency'] = loop_data['enc_token_time']
        iter_data['enc_infer_latency'] = loop_data['enc_infer_time']
        iter_data['first_token_latency'] = loop_data['dec_1st_token_time']
        iter_data['other_tokens_avg_latency'] = loop_data['dec_2nd_tokens_time']
        iter_data['first_token_infer_latency'] = loop_data['dec_1st_infer_time']
        iter_data['other_tokens_infer_avg_latency'] = loop_data['dec_2nd_infers_time']
    else:
        iter_data['enc_token_latency'] = -1
        iter_data['enc_infer_latency'] = -1
        iter_data['first_token_latency'] = -1
        iter_data['other_tokens_avg_latency'] = -1
        iter_data['first_token_infer_latency'] = -1
        iter_data['other_tokens_infer_avg_latency'] = -1

    return iter_data


def gen_json_data(
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
    input_loop_data=None
):
    json_data = {}
    loop_data_list = []
    json_data['iteration'] = iter_idx
    json_data['input_size'] = in_size
    json_data['infer_count'] = infer_count
    json_data['output_size'] = out_size
    json_data['generation_time'] = gen_time
    json_data['latency'] = latency
    json_data['result_md5'] = res_md5
    json_data['max_rss_mem_consumption'] = max_rss_mem
    json_data['max_shared_mem_consumption'] = max_shared_mem
    json_data['max_uss_mem_consumption'] = max_uss_mem
    json_data['prompt_idx'] = prompt_idx
    json_data['tokenization_time'] = tokenization_time[0] if len(tokenization_time) > 0 else ''
    json_data['detokenization_time'] = tokenization_time[1] if len(tokenization_time) > 1 else ''

    if input_loop_data is not None:
        for data in input_loop_data:
            loop_data = {}
            enc_data = {
                'token_latency': data['enc_token_time'],
                'infer_latency': data['enc_infer_time']
            }
            loop_data['encoder'] = enc_data
            loop_data['first_latency'] = data['dec_1st_token_time']
            loop_data['second_avg_latency'] = data['dec_2nd_tokens_time']
            loop_data['first_infer_latency'] = data['dec_1st_infer_time']
            loop_data['second_infer_avg_latency'] = data['dec_2nd_infers_time']
            loop_data_list.append(loop_data)
    json_data['loop'] = loop_data_list

    return json_data
