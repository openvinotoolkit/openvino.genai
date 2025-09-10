# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log


def print_metrics(iter_num, iter_data, tms=None, tms_infer=None, warm_up=False,
                  stable_diffusion=None, tokenization_time=None, batch_size=1,
                  prompt_idx=-1, whisper=None, text_emb=None, latency_unit=None,
                  tts=None, cb_metric=None, text_rerank=None):
    iter_str = str(iter_num)
    if warm_up:
        iter_str = 'warm-up'
    output_str = ''
    latency_unit = 'token' if latency_unit is None else latency_unit
    prefix = f'[{iter_str}][P{prompt_idx}]'
    if batch_size > 1:
        latency_unit = '{}{}s'.format(batch_size, latency_unit)
    if iter_data['input_size'] != '':
        output_str += 'Input token size: {}, '.format(iter_data['input_size'])
    if iter_data.get('output_size', '') != '':
        output_str += 'Output size: {}, '.format(iter_data['output_size'])
    if iter_data['infer_count'] != '':
        output_str += 'Infer count: {}, '.format(iter_data['infer_count'])
    if tokenization_time:
        output_str += 'Tokenization Time: {:.2f}ms, '.format(tokenization_time[0])
        if len(tokenization_time) > 1:
            output_str += 'Detokenization Time: {:.2f}ms, '.format(tokenization_time[1])
    if iter_data.get('mm_embeddings_preparation_time', '') != '':
        output_str += ' Multimodal Embeddings Preparation Time: {:.2f}ms, '.format(iter_data['mm_embeddings_preparation_time'])
    if iter_data.get('generation_time', '') != '':
        output_str += 'Generation Time: {:.2f}s, '.format(iter_data['generation_time'])
    if iter_data.get('total_time', '') != '':
        output_str += 'Total Time: {:.4f}s, '.format(iter_data["total_time"])
    if iter_data['latency'] != '':
        output_str += 'Latency: {:.4f} ms/{}'.format(iter_data['latency'], latency_unit)
    if output_str != '':
        output_str = ' '.join([prefix, output_str])
        log.info(output_str)
    if tms is not None:
        iter_data['first_token_latency'] = tms[0] * 1000 if len(tms) > 0 else -1
        iter_data['other_tokens_avg_latency'] = sum(tms[1:]) / (len(tms) - 1) * 1000 if len(tms) > 1 else -1
        first_token_latency = 'NA' if iter_data['first_token_latency'] == -1 else f"{iter_data['first_token_latency']:.2f} ms"
        other_token_latency = 'NA' if iter_data['other_tokens_avg_latency'] == -1 else f"{iter_data['other_tokens_avg_latency']:.2f} ms/{latency_unit}"
        if text_emb is not None:
            log.info(
                f"{prefix} First iteration latency: {first_token_latency}, "
                f'other iterations latency: {other_token_latency}, len of input tokens: {iter_data["input_size"]} * {batch_size}',
            )
        elif text_rerank is not None:
            log.info(
                f"{prefix} First iteration latency: {first_token_latency}, "
                f'other iteration latency: {other_token_latency}, len of input tokens: {iter_data["input_size"]}, '
                f'texts number: {text_rerank.get("texts_num", -1)}',
            )
        else:
            log.info(
                f"{prefix} First token latency: {first_token_latency}, "
                f'other tokens latency: {other_token_latency}, len of input tokens: {iter_data["input_size"]} * {batch_size}',
            )
        if len(tms) == 0:
            log.warning(f'{prefix} No hook data output for first token latency and other tokens latency')
    if tms_infer is not None:
        iter_data['first_token_infer_latency'] = tms_infer[0] * 1000 if len(tms_infer) > 0 else -1
        iter_data['other_tokens_infer_avg_latency'] = sum(tms_infer[1:]) / (len(tms_infer) - 1) * 1000 if len(tms_infer) > 1 else -1
        first_infer_latency = 'NA' if iter_data['first_token_infer_latency'] == -1 else f"{iter_data['first_token_infer_latency']:.2f} ms"
        other_infer_latency = 'NA' if iter_data['other_tokens_infer_avg_latency'] == -1 else f"{iter_data['other_tokens_infer_avg_latency']:.2f} ms/infer"
        log.info(
            f'{prefix} First infer latency: {first_infer_latency}, '
            f'other infers latency: {other_infer_latency}, inference count: {len(tms_infer)}',
        )
        if len(tms_infer) == 0:
            log.warning(f'{prefix} No hook data output for first infer latency and other infers latency')
    if cb_metric and "avg_cache_usage" in cb_metric and "max_cache_usage" in cb_metric:
        log.info(
            f'Running average of the KV cache usage {cb_metric["avg_cache_usage"]:.2f}%, '
            f'max KV cache usage: {cb_metric["max_cache_usage"]:.2f}%',
        )
    if stable_diffusion is not None:
        print_stable_diffusion_infer_latency(iter_str, iter_data, stable_diffusion, prompt_idx)
    if whisper is not None:
        print_whisper_infer_latency(iter_str, whisper, prompt_idx)
    if tts is not None:
        print_tts_latency(iter_str, tts, prompt_idx)
    output_str = ''
    if iter_data['max_rss_mem_consumption'] != '' and iter_data['max_rss_mem_consumption'] > -1:
        output_str += f"Max rss memory cost: {iter_data['max_rss_mem_consumption']:.2f}MBytes, "
    if iter_data['max_rss_mem_increase'] != '' and iter_data['max_rss_mem_increase'] > -1:
        output_str += f"rss memory increase: {iter_data['max_rss_mem_increase']:.2f}MBytes, "
    if iter_data['max_sys_mem_consumption'] != '' and iter_data['max_sys_mem_consumption'] > -1:
        output_str += f"max system memory memory cost: {iter_data['max_sys_mem_consumption']:.2f}MBytes, "
    if iter_data['max_sys_mem_increase'] != '' and iter_data['max_sys_mem_increase'] > -1:
        output_str += f"system memory increase: {iter_data['max_sys_mem_increase']:.2f}MBytes "
    if output_str != '':
        output_str = ' '.join([prefix, output_str])
        log.info(output_str)
    if iter_data.get('result_md5', '') != '':
        log.info(f"{prefix} Result MD5:{iter_data['result_md5']}")


def print_generated(iter_num, warm_up=False, generated=None, prompt_idx=-1):
    iter_str = str(iter_num)
    if warm_up:
        iter_str = 'warm-up'
    prefix = f'[{iter_str}][P{prompt_idx}]'
    if generated is not None:
        print_unicode(f'{prefix} Generated: {generated}', '{prefix} Unable print generated')


def print_unicode(text, on_error="Unable print", loglevel="info"):
    log_fn = getattr(log, loglevel)
    try:
        log_fn(text)
    except (UnicodeError, UnicodeEncodeError, UnicodeDecodeError):
        try:
            utf8_text = text.encode(encoding="utf-8", errors="replace").decode()
            log_fn(utf8_text)
        except Exception:
            log.warning(on_error)


def print_stable_diffusion_infer_latency(iter_str, iter_data, stable_diffusion=None, prompt_idx=-1):
    if (len(stable_diffusion.raw_metrics.unet_inference_durations) != 0):
        main_model_name = "unet"
        first_token_latency, other_tokens_latency = stable_diffusion.get_first_and_other_unet_infer_duration()
    else:
        main_model_name = "transformer"
        first_token_latency, other_tokens_latency = stable_diffusion.get_first_and_other_trans_infer_duration()
    iter_data['first_token_latency'] = first_token_latency
    iter_data['other_tokens_avg_latency'] = other_tokens_latency
    iter_data['first_token_infer_latency'] = iter_data['first_token_latency']
    iter_data['other_tokens_infer_avg_latency'] = iter_data['other_tokens_avg_latency']

    prefix = f'[{iter_str}][P{prompt_idx}]'
    log.info(f"{prefix} First step of {main_model_name} latency: {iter_data['first_token_latency']:.2f} ms, "
             f"other steps of {main_model_name} latency: {iter_data['other_tokens_avg_latency']:.2f} ms/step, "
             f"{main_model_name} step count: {len(stable_diffusion.raw_metrics.unet_inference_durations)}")

    log_str = f"{prefix} "
    if (len(stable_diffusion.get_text_encoder_infer_duration().keys())):
        log_str += "Text encoder latency: "
        for text_enc_name, duration in stable_diffusion.get_text_encoder_infer_duration().items():
            log_str += f"{text_enc_name} {duration:.2f} ms/step, "
    else:
        log_str += "Text encoder latency: N/A "

    log_str += (f"vae decoder latency: {stable_diffusion.get_vae_decoder_infer_duration():.2f} ms/step, ")

    if hasattr(stable_diffusion, 'get_text_encoder_step_count'):
        log_str += f"text encoder step count: {stable_diffusion.get_text_encoder_step_count()}, "

    if hasattr(stable_diffusion, 'get_vae_decoder_step_count'):
        log_str += f"vae decoder step count: {stable_diffusion.get_vae_decoder_step_count()}, "
    else:
        log_str += "vae decoder step count: 1 "

    log.info(log_str)


def print_ldm_unet_vqvae_infer_latency(iter_num, iter_data, tms=None, warm_up=False, prompt_idx=-1):
    iter_str = str(iter_num)
    if warm_up:
        iter_str = 'warm-up'
    len_tms = len(tms)
    iter_data['first_token_latency'] = tms[0] * 1000 if len_tms > 0 else -1
    iter_data['other_tokens_avg_latency'] = sum(tms[1:(len_tms - 1)]) / (len_tms - 2) * 1000 if len_tms > 2 else 0
    iter_data['first_token_infer_latency'] = iter_data['first_token_latency']
    iter_data['other_tokens_infer_avg_latency'] = iter_data['other_tokens_avg_latency']

    first_token_latency = 'NA' if iter_data['first_token_latency'] == -1 else f"{iter_data['first_token_latency']:.2f} ms"
    other_token_latency = 'NA' if iter_data['other_tokens_avg_latency'] == -1 else f"{iter_data['other_tokens_avg_latency']:.2f} ms/step"
    prefix = f'[{iter_str}][P{prompt_idx}]'
    log.info(f"{prefix} First step of unet latency: {first_token_latency}, "
             f"other steps of unet latency: {other_token_latency}",)
    if len_tms > 1:
        log.info(f"{prefix} Unet latency: {(sum(tms[0:(len_tms - 1)]) / (len_tms - 1)) * 1000:.2f} ms/step, "
                 f"vqvae decoder latency: {tms[len_tms - 1] * 1000:.2f} ms/step, "
                 f"unet step count: {len_tms - 1}, "
                 f"vqvae decoder step count: 1",)


def output_avg_statis_tokens(prompt_dict, prompt_idx_list, iter_data_list, batch_size, is_text_gen, is_embed, loop_idx, latency_unit=None):
    for p_idx in prompt_idx_list:
        avg_1st_token_latency = 0
        avg_2nd_tokens_latency = 0
        avg_2nd_token_tput = 0
        avg_input_size = 0
        index_num = 0

        if latency_unit is None:
            latency_unit = 'token' if is_text_gen else 'step'
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
            tput_unit = latency_unit
            if batch_size > 1:
                latency_unit = '{}{}s'.format(batch_size, latency_unit)
            avg_1st_token_latency = 'NA' if avg_1st_token_latency < 0 else f'{avg_1st_token_latency:.2f} ms'
            avg_2nd_tokens_latency = 'NA' if avg_2nd_tokens_latency < 0 else f'{avg_2nd_tokens_latency:.2f} ms/{latency_unit}'
            avg_2nd_token_tput = 'NA' if avg_2nd_tokens_latency == 'NA' else f'{avg_2nd_token_tput:.2f} {tput_unit}s/s'
            prefix = f'[ INFO ] [Average] P[{p_idx}]L[{loop_idx}]' if loop_idx != -1 else f'[ INFO ] [Average] P[{p_idx}]'
            if is_text_gen:
                output_info = ''
                if avg_input_size > 0:
                    output_info += f' Input token size: {avg_input_size},'
                prompt_dict[p_idx] = '\n{}{} 1st token latency: {}, ' \
                    '2nd token latency: {}, 2nd tokens throughput: {}' \
                    .format(prefix, output_info, avg_1st_token_latency, avg_2nd_tokens_latency, avg_2nd_token_tput)
            elif is_embed:
                output_info = ''
                if avg_input_size > 0:
                    output_info += f' Input token size: {avg_input_size},'
                prompt_dict[p_idx] = '\n{}{} 1st iteration latency: {}, ' \
                    '2nd iteration latency: {}, 2nd iteration throughput: {}' \
                    .format(prefix, output_info, avg_1st_token_latency, avg_2nd_tokens_latency, avg_2nd_token_tput)
            else:
                prompt_dict[p_idx] = '\n{} 1st step of unet/transformer latency: {}, ' \
                    '2nd steps of unet/transformer latency: {}, 2nd steps throughput: {}' \
                    .format(prefix, avg_1st_token_latency, avg_2nd_tokens_latency, avg_2nd_token_tput)


def print_average(iter_data_list, prompt_idx_list, batch_size, is_text_gen=False, is_embed=False, loop_idx=-1, latency_unit=None):
    if len(iter_data_list) <= 1:
        # 1st iteration is the warm-up iteration
        return
    warm_up_iters = 0
    for iter_data in iter_data_list:
        if iter_data['iteration'] == 0:
            # Exclude the warm-up iteration
            warm_up_iters = warm_up_iters + 1
            continue
    total_iters = len(iter_data_list) - warm_up_iters

    if total_iters > 0:
        prompt_dict = {}
        output_avg_statis_tokens(prompt_dict, prompt_idx_list, iter_data_list, batch_size, is_text_gen, is_embed, loop_idx, latency_unit)
        log.info('<<< Warm-up iteration is excluded. >>>')
        out_str = '[Total] Iterations: {}'.format(total_iters)
        for prompt_key in prompt_dict:
            out_str += prompt_dict[prompt_key]
        log.info(out_str)


def print_whisper_infer_latency(iter_str, whisper, prompt_idx=-1):
    log.debug(f'{whisper.print_whisper_latency(iter_str, prompt_idx)}')


def print_tts_latency(iter_str, tts_hook, prompt_idx=-1):
    log.debug(tts_hook.print_tts_latency(iter_str, prompt_idx))
