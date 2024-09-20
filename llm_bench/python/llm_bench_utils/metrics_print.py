# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log


def print_metrics(
        iter_num, iter_data, tms=None, tms_infer=None, warm_up=False, max_rss_mem=-1, max_shared_mem=-1,
        max_uss_mem=-1, stable_diffusion=None, tokenization_time=None, batch_size=1
):
    iter_str = str(iter_num)
    if warm_up:
        iter_str = 'warm-up'
    output_str = ''
    latency_unit = 'token'
    if batch_size > 1:
        latency_unit = '{}tokens'.format(batch_size)
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
        output_str += 'Latency: {:.2f} ms/{}'.format(iter_data['latency'], latency_unit)
    if output_str != '':
        output_str = ' '.join(['[{}]'.format(iter_str), output_str])
        log.info(output_str)
    if tms is not None:
        iter_data['first_token_latency'] = tms[0] * 1000 if len(tms) > 0 else -1
        iter_data['other_tokens_avg_latency'] = sum(tms[1:]) / (len(tms) - 1) * 1000 if len(tms) > 1 else -1
        first_token_latency = 'NA' if iter_data['first_token_latency'] == -1 else f"{iter_data['first_token_latency']:.2f} ms/{latency_unit}"
        other_token_latency = 'NA' if iter_data['other_tokens_avg_latency'] == -1 else f"{iter_data['other_tokens_avg_latency']:.2f} ms/{latency_unit}"
        log.info(
            f"[{iter_str}] First token latency: {first_token_latency}, "
            f"other tokens latency: {other_token_latency}, len of tokens: {len(tms)} * {batch_size}",
        )
        if len(tms) == 0:
            log.warning(f'[{iter_str}] No hook data output for first token latency and other tokens latency')
    if tms_infer is not None:
        iter_data['first_token_infer_latency'] = tms_infer[0] * 1000 if len(tms_infer) > 0 else -1
        iter_data['other_tokens_infer_avg_latency'] = sum(tms_infer[1:]) / (len(tms_infer) - 1) * 1000 if len(tms_infer) > 1 else -1
        first_infer_latency = 'NA' if iter_data['first_token_infer_latency'] == -1 else f"{iter_data['first_token_infer_latency']:.2f} ms/infer"
        other_infer_latency = 'NA' if iter_data['other_tokens_infer_avg_latency'] == -1 else f"{iter_data['other_tokens_infer_avg_latency']:.2f} ms/infer"
        log.info(
            f"[{iter_str}] First infer latency: {first_infer_latency}, "
            f"other infers latency: {other_infer_latency}, inference count: {len(tms_infer)}",
        )
        if len(tms_infer) == 0:
            log.warning(f'[{iter_str}] No hook data output for first infer latency and other infers latency')
    if stable_diffusion is not None:
        print_stable_diffusion_infer_latency(iter_str, iter_data, stable_diffusion)
    output_str = ''
    if max_rss_mem != '' and max_rss_mem > -1:
        output_str += 'Max rss memory cost: {:.2f}MBytes, '.format(max_rss_mem)
    if max_uss_mem != '' and max_uss_mem > -1:
        output_str += 'max uss memory cost: {:.2f}MBytes, '.format(max_uss_mem)
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
        try:
            log.info(f'[{iter_str}] Generated: {generated}')
        except UnicodeError:
            try:
                utf8_generated = generated.encode(encoding="utf-8", errors="replace").decode()
                log.info(f'[{iter_str}] Generated: {utf8_generated}')
            except Exception:
                log.warning(f"[{iter_str}] Unable print generated")


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

    first_token_latency = 'NA' if iter_data['first_token_latency'] == -1 else f"{iter_data['first_token_latency']:.2f} ms/step"
    other_token_latency = 'NA' if iter_data['other_tokens_avg_latency'] == -1 else f"{iter_data['other_tokens_avg_latency']:.2f} ms/step"
    log.info(f"[{iter_str}] First step of unet latency: {first_token_latency}, "
             f"other steps of unet latency: {other_token_latency}",)
    if len_tms > 1:
        log.info(f"[{iter_str}] Unet latency: {(sum(tms[0:(len_tms - 1)]) / (len_tms - 1)) * 1000:.2f} ms/step, "
                 f"vqvae decoder latency: {tms[len_tms - 1] * 1000:.2f} ms/step, "
                 f"unet step count: {len_tms - 1}, "
                 f"vqvae decoder step count: 1",)


def output_avg_statis_tokens(prompt_dict, prompt_idx_list, iter_data_list, batch_size, is_text_gen):
    for p_idx in prompt_idx_list:
        avg_1st_token_latency = 0
        avg_2nd_tokens_latency = 0
        avg_2nd_token_tput = 0
        avg_input_size = 0
        index_num = 0
        for iter_data in iter_data_list:
            # Exclude the warm-up iteration
            if iter_data['iteration'] == 0:
                continue
            if iter_data['prompt_idx'] == p_idx:
                avg_1st_token_latency += iter_data['first_token_latency'] if iter_data['first_token_latency'] != '' else 0
                avg_2nd_tokens_latency += iter_data['other_tokens_avg_latency'] if iter_data['other_tokens_avg_latency'] != '' else 0
                avg_input_size += iter_data['input_size'] if iter_data['input_size'] != '' else 0
                index_num = index_num + 1
        if index_num > 0:
            avg_1st_token_latency = avg_1st_token_latency / index_num
            avg_2nd_tokens_latency = avg_2nd_tokens_latency / index_num
            avg_input_size = int(avg_input_size / index_num)
            if avg_2nd_tokens_latency > 0:
                avg_2nd_token_tput = (1 / avg_2nd_tokens_latency) * batch_size * 1000
            latency_unit = 'token' if is_text_gen is True else 'step'
            if batch_size > 1:
                if is_text_gen is True:
                    latency_unit = '{}tokens'.format(batch_size)
                else:
                    latency_unit = '{}steps'.format(batch_size)
            avg_1st_token_latency = 'NA' if avg_1st_token_latency < 0 else f'{avg_1st_token_latency:.2f} ms/{latency_unit}'
            avg_2nd_tokens_latency = 'NA' if avg_2nd_tokens_latency < 0 else f'{avg_2nd_tokens_latency:.2f} ms/{latency_unit}'
            avg_2nd_token_tput = 'NA' if avg_2nd_tokens_latency == 'NA' else f'{avg_2nd_token_tput:.2f} {latency_unit}s/s'
            if is_text_gen is True:
                prompt_dict[p_idx] = '\n[ INFO ] [Average] Prompt[{}] Input token size: {}, 1st token lantency: {}, ' \
                    '2nd token lantency: {}, 2nd tokens throughput: {}' \
                    .format(p_idx, avg_input_size, avg_1st_token_latency, avg_2nd_tokens_latency, avg_2nd_token_tput)
            else:
                prompt_dict[p_idx] = '\n[ INFO ] [Average] Prompt[{}] 1st step of unet latency: {}, ' \
                    '2nd steps of unet latency: {}, 2nd steps throughput: {}' \
                    .format(p_idx, avg_1st_token_latency, avg_2nd_tokens_latency, avg_2nd_token_tput)


def print_average(iter_data_list, prompt_idx_list, batch_size, is_text_gen=False):
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
        prompt_dict = {}
        output_avg_statis_tokens(prompt_dict, prompt_idx_list, iter_data_list, batch_size, is_text_gen)
        log.info('<<< Warm-up iteration is excluded. >>>')
        out_str = '[Total] Iterations: {}'.format(total_iters)
        for prompt_key in prompt_dict:
            out_str += prompt_dict[prompt_key]
        log.info(out_str)
