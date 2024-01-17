# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log


def print_metrics(
        iter_num, iter_data, tms=None, tms_infer=None, warm_up=False, max_rss_mem=-1, max_shared_mem=-1,
        stable_diffusion=None, tokenization_time=None
):
    if tms is None:
        tms = []
    if tms_infer is None:
        tms_infer = []
    iter_str = str(iter_num)
    if warm_up:
        iter_str = 'warm-up'
    output_str = ''
    if iter_data['input_size'] != '':
        output_str += 'Input token size: {}, '.format(iter_data['input_size'])
    if iter_data['output_size'] != '':
        output_str += 'Output size: {}, '.format(iter_data['output_size'])
    if iter_data['infer_count'] != '':
        output_str += 'Infer count: {}, '.format(iter_data['infer_count'])
    if tokenization_time:
        output_str += 'Tokenization Time: {:.2f}ms, '.format(tokenization_time[0])
        if len(tokenization_time) > 1:
            output_str += 'Detokenization Time: {:.2f}ms, '.format(tokenization_time[1])
    if iter_data['generation_time'] != '':
        output_str += 'Generation Time: {:.2f}s, '.format(iter_data['generation_time'])
    if iter_data['latency'] != '':
        output_str += 'Latency: {:.2f} ms/token'.format(iter_data['latency'])
    if output_str != '':
        output_str = ' '.join(['[{}]'.format(iter_str), output_str])
        log.info(output_str)
    if len(tms) > 0:
        iter_data['first_token_latency'] = tms[0] * 1000 if len(tms) > 0 else -1
        iter_data['other_tokens_avg_latency'] = sum(tms[1:]) / (len(tms) - 1) * 1000 if len(tms) > 1 else -1
        log.info(
            f"[{iter_str}] First token latency: {iter_data['first_token_latency']:.2f} ms/token, "
            f"other tokens latency: {iter_data['other_tokens_avg_latency']:.2f} ms/token, len of tokens: {len(tms)}",
        )
    if len(tms_infer) > 0:
        iter_data['first_token_infer_latency'] = tms_infer[0] * 1000 if len(tms_infer) > 0 else -1
        iter_data['other_tokens_infer_avg_latency'] = sum(tms_infer[1:]) / (len(tms_infer) - 1) * 1000 if len(tms_infer) > 1 else -1
        log.info(
            f"[{iter_str}] First infer latency: {iter_data['first_token_infer_latency']:.2f} ms/infer, "
            f"other infers latency: {iter_data['other_tokens_infer_avg_latency']:.2f} ms/infer, inference count: {len(tms_infer)}",
        )
    if stable_diffusion is not None:
        print_stable_diffusion_infer_latency(iter_str, iter_data, stable_diffusion)
    output_str = ''
    if max_rss_mem != '' and max_rss_mem > -1:
        output_str += 'Max rss memory cost: {:.2f}MBytes, '.format(max_rss_mem)
    if max_shared_mem != '' and max_shared_mem > -1:
        output_str += 'max shared memory cost: {:.2f}MBytes'.format(max_shared_mem)
    if output_str != '':
        output_str = ' '.join(['[{}]'.format(iter_str), output_str])
        log.info(output_str)
    if iter_data['result_md5'] != '':
        log.info(f"[{iter_str}] Result MD5:{iter_data['result_md5']}")


def print_generated(iter_num, warm_up=False, generated=None):
    iter_str = str(iter_num)
    if warm_up:
        iter_str = 'warm-up'
    if generated is not None:
        log.info(f'[{iter_str}] Generated: {generated}')


def print_stable_diffusion_infer_latency(iter_str, iter_data, stable_diffusion):
    iter_data['first_token_latency'] = stable_diffusion.get_1st_unet_latency()
    iter_data['other_tokens_avg_latency'] = stable_diffusion.get_2nd_unet_latency()
    iter_data['first_token_infer_latency'] = iter_data['first_token_latency']
    iter_data['other_tokens_infer_avg_latency'] = iter_data['other_tokens_avg_latency']
    log.info(f"[{iter_str}] First step of unet latency: {iter_data['first_token_latency']:.2f} ms/step, "
             f"other steps of unet latency: {iter_data['other_tokens_avg_latency']:.2f} ms/step",)
    log.info(f"[{iter_str}] Text encoder latency: {stable_diffusion.get_text_encoder_latency():.2f} ms/step, "
             f"unet latency: {stable_diffusion.get_unet_latency():.2f} ms/step, "
             f"vae decoder latency: {stable_diffusion.get_vae_decoder_latency():.2f} ms/step, "
             f"text encoder step count: {stable_diffusion.get_text_encoder_step_count()}, "
             f"unet step count: {stable_diffusion.get_unet_step_count()}, "
             f"vae decoder step count: {stable_diffusion.get_vae_decoder_step_count()}",)


def print_ldm_unet_vqvae_infer_latency(iter_num, iter_data, tms=None, warm_up=False):
    iter_str = str(iter_num)
    if warm_up:
        iter_str = 'warm-up'
    len_tms = len(tms)
    iter_data['first_token_latency'] = tms[0] * 1000 if len_tms > 0 else -1
    iter_data['other_tokens_avg_latency'] = sum(tms[1:(len_tms - 1)]) / (len_tms - 2) * 1000 if len_tms > 2 else 0
    iter_data['first_token_infer_latency'] = iter_data['first_token_latency']
    iter_data['other_tokens_infer_avg_latency'] = iter_data['other_tokens_avg_latency']

    log.info(f"[{iter_str}] First step of unet latency: {iter_data['first_token_latency']:.2f} ms/step, "
             f"other steps of unet latency: {iter_data['other_tokens_avg_latency']:.2f} ms/step",)
    if len_tms > 1:
        log.info(f"[{iter_str}] Unet latency: {(sum(tms[0:(len_tms - 1)]) / (len_tms - 1)) * 1000:.2f} ms/step, "
                 f"vqvae decoder latency: {tms[len_tms - 1] * 1000:.2f} ms/step, "
                 f"unet step count: {len_tms - 1}, "
                 f"vqvae decoder step count: 1",)


def print_average(iter_data_list):
    if len(iter_data_list) <= 1:
        # 1st iteration is the warm-up iteration
        return
    total_generation_time = 0
    total_num_tokens = 0
    warm_up_iters = 0
    for iter_data in iter_data_list:
        if iter_data['iteration'] == 0:
            # Exclude the warm-up iteration
            warm_up_iters = warm_up_iters + 1
            continue
        if iter_data['generation_time'] != '':
            total_generation_time += iter_data['generation_time']
        if iter_data['output_size'] != '':
            total_num_tokens += iter_data['output_size']

    total_iters = len(iter_data_list) - warm_up_iters

    if total_iters > 0:
        log.info('<<< Warm-up iteration is excluded. >>>')
        out_str = '[Total] Iterations: {}'.format(total_iters)
        if total_num_tokens > 0:
            out_str += ', Output size: {} tokens'.format(total_num_tokens)
        if total_generation_time > 0:
            avg_per_iter_time = total_generation_time / total_iters
            out_str += '\n[ INFO ] [Average] Iteration time: {:.2f}s'.format(avg_per_iter_time)
            if total_num_tokens > 0:
                avg_per_token_time = total_generation_time * 1000 / total_num_tokens
                out_str += ', Latency: {:.2f} ms/token'.format(avg_per_token_time)
        log.info(out_str)
