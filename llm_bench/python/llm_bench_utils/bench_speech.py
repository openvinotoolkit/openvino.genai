# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
from pathlib import Path
import hashlib
import logging as log
from transformers import pipeline
import llm_bench_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.hook_forward_whisper import WhisperHook

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

whisper_hook = WhisperHook()


def run_speech_2txt_generation(pipe, args, num, md5_list, prompt_id, audio_prompt, iter_data_list, json_data_list, mem_consumption):
    result_md5_list = []
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    inputs = model_utils.get_audio(audio_prompt['prompt'], pipe.feature_extractor.sampling_rate)
    start = time.perf_counter()
    result_text = pipe(inputs, generate_kwargs={"task": 'translate'}, return_timestamps=True)["text"]
    end = time.perf_counter()
    generation_time = end - start

    out_data = pipe.tokenizer(result_text, return_tensors='pt')
    out_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    out_tokens = out_data['input_ids'] if 'input_ids' in out_data else out_data
    out_token_size = out_tokens[0].numel()

    result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_id : result_md5_list}
    else:
        md5_list[num][prompt_id] = result_md5_list
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    latency_list = whisper_hook.get_whisper_latency()
    for loop_idx, data in enumerate(latency_list):
        iter_data = gen_output_data.gen_iterate_data(
            iter_idx=num,
            loop_idx=loop_idx,
            out_size=out_token_size,
            gen_time=generation_time,
            res_md5=result_md5_list,
            max_rss_mem=max_rss_mem_consumption,
            max_shared_mem=max_shared_mem_consumption,
            max_uss_mem=max_uss_mem_consumption,
            prompt_idx=prompt_id,
            loop_data=data
        )
        iter_data_list.append(iter_data)
    json_data = gen_output_data.gen_json_data(
        iter_idx=num,
        out_size=out_token_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_id,
        input_loop_data=latency_list
    )
    json_data_list.append(json_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        whisper=whisper_hook,
        prompt_idx=prompt_id
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_id]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_id}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=result_text, prompt_idx=prompt_id)
            if num == 1:
                # if the device is CPU, throw exception
                if args['devices'].lower().startswith('cpu') is True:
                    assert (result_md5_list == prev_md5)
            else:
                # throw exception
                assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=result_text, prompt_idx=prompt_id)
    whisper_hook.clear_statistics()


def run_speech_2txt_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    iter_data_list = []
    json_data_list = []
    input_audio_prompt_list = model_utils.get_audio_param_from_prompt_file(args)
    audios_prompt_list = []
    if len(input_audio_prompt_list) > 0:
        for audio_prompt in input_audio_prompt_list:
            if args['prompt'] is None and args['prompt_file'] is None:
                raise RuntimeError('==Failure image is empty ==')
            elif args['prompt_file'] is not None and len(args['prompt_file']) > 0:
                audio_prompt['prompt'] = os.path.join(os.path.dirname(args['prompt_file'][0]), audio_prompt['prompt'].replace('./', ''))
            audio_prompt['prompt'] = Path(audio_prompt['prompt'])
            audios_prompt_list.append(audio_prompt)
    if args['prompt_index'] is None:
        prompt_idx_list = [prompt_idx for prompt_idx, input_audio in enumerate(audios_prompt_list)]
        audio_list = audios_prompt_list
    else:
        prompt_idx_list = []
        audio_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(audios_prompt_list):
                audio_list.append(audios_prompt_list[i])
                prompt_idx_list.append(i)
    if len(audio_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f'Benchmarking iter nums(exclude warm-up): {num_iters}, prompt nums: {len(input_audio_prompt_list)}, prompt idx: {prompt_idx_list}')
    ov_model, processor, pretrain_time = FW_UTILS[framework].create_speech_2txt_model(model_path, device, **args)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=ov_model,
        chunk_length_s=30,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )
    if framework == "ov":
        whisper_hook.new_text_encoder(pipe)
        whisper_hook.new_text_encoder_request(pipe)
        whisper_hook.new_generate(pipe)
        whisper_hook.new_text_sample(pipe)
    md5_list = {num : {} for num in range(num_iters + 1)}
    for num in range(num_iters + 1):
        for idx, audio_prompt in enumerate(audio_list):
            run_speech_2txt_generation(pipe, args, num, md5_list, prompt_idx_list[idx], audio_prompt, iter_data_list, json_data_list, mem_consumption)
    metrics_print.print_average(iter_data_list, prompt_idx_list, 1, True, 0)

    return iter_data_list, pretrain_time, json_data_list
