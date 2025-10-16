# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def gen_iterate_data(
    iter_idx='',
    in_size='',
    infer_count='',
    out_size='',
    gen_time='',
    latency='',
    res_md5='',
    max_rss_mem='',
    max_rss_mem_increase='',
    max_sys_mem='',
    max_sys_mem_increase='',
    prompt_idx='',
    tokenization_time=[],
    mm_embeddings_preparation_time=''
):
    iter_data = {}
    iter_data['iteration'] = iter_idx
    iter_data['input_size'] = in_size
    iter_data['infer_count'] = infer_count
    iter_data['output_size'] = out_size
    iter_data['generation_time'] = gen_time
    iter_data['latency'] = latency
    iter_data['result_md5'] = res_md5
    iter_data['first_token_latency'] = -1
    iter_data['other_tokens_avg_latency'] = -1
    iter_data['first_token_infer_latency'] = -1
    iter_data['other_tokens_infer_avg_latency'] = -1
    iter_data['max_rss_mem_consumption'] = max_rss_mem
    iter_data['max_rss_mem_increase'] = max_rss_mem_increase
    iter_data['max_sys_mem_consumption'] = max_sys_mem
    iter_data['max_sys_mem_increase'] = max_sys_mem_increase
    iter_data['prompt_idx'] = prompt_idx
    iter_data['tokenization_time'] = tokenization_time[0] if len(tokenization_time) > 0 else ''
    iter_data['detokenization_time'] = tokenization_time[1] if len(tokenization_time) > 1 else ''
    iter_data["mm_embeddings_preparation_time"] = mm_embeddings_preparation_time
    return iter_data


def embed_iterate_data(
    iter_idx='',
    in_size='',
    infer_count='',
    total_time='',
    latency='',
    max_rss_mem='',
    max_rss_mem_increase='',
    max_sys_mem='',
    max_sys_mem_increase='',
    prompt_idx='',
    tokenization_time=[],
):
    iter_data = {}
    iter_data['iteration'] = iter_idx
    iter_data['input_size'] = in_size
    iter_data['infer_count'] = infer_count
    iter_data['generation_time'] = total_time
    iter_data['latency'] = latency
    iter_data['first_token_latency'] = -1
    iter_data['other_tokens_avg_latency'] = -1
    iter_data['first_token_infer_latency'] = -1
    iter_data['other_tokens_infer_avg_latency'] = -1
    iter_data['max_rss_mem_consumption'] = max_rss_mem
    iter_data['max_rss_mem_increase'] = max_rss_mem_increase
    iter_data['max_sys_mem_consumption'] = max_sys_mem
    iter_data['max_sys_mem_increase'] = max_sys_mem_increase
    iter_data['prompt_idx'] = prompt_idx
    iter_data['tokenization_time'] = tokenization_time[0] if len(tokenization_time) > 0 else ''
    iter_data['detokenization_time'] = ''
    iter_data['result_md5'] = ''
    iter_data['output_size'] = ''
    return iter_data
