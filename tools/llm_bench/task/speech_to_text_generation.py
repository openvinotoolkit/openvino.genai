# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import datetime
import numpy as np
import hashlib
import logging as log
import llm_bench_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import BenchPrompter
from llm_bench_utils.hook_forward_whisper import WhisperHook

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}
whisper_hook = WhisperHook()

DEFAULT_OUTPUT_TOKEN_SIZE = 1000


def run_speech_2_txt_generation(input_param, args, md5_list, iter_data_list):
    result_md5_list = []
    pipe = input_param['pipe']
    raw_speech = input_param['raw_speech']
    num = input_param['iter_idx']
    speech_id = input_param['speech_idx']
    processor = input_param['processor']
    use_genai = input_param['use_genai']
    speech_language = input_param['speech_param'].get('language', "<|en|>")
    ret_timestamps = input_param['speech_param'].get('timestamp', True)
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']

    mem_consumption = input_param["mem_consumption"]
    mem_consumption.start(num)
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

        wm = perf_metrics.whisper_raw_metrics
        enc_ms = (
            [v / 1000 for v in wm.encode_inference_durations]
            if getattr(wm, "encode_inference_durations", None) is not None
            else []
        )
        dec_ms = (
            [v / 1000 for v in wm.decode_inference_durations]
            if getattr(wm, "decode_inference_durations", None) is not None
            else []
        )
        smp_ms = (
            [v / 1000 for v in perf_metrics.raw_metrics.sampling_durations]
            if getattr(perf_metrics.raw_metrics, "sampling_durations", None) is not None
            else []
        )
        whisper_genai_metrics = {
            "tokenization_ms": perf_metrics.get_tokenization_duration().mean,
            "features_extraction_ms": perf_metrics.get_features_extraction_duration().mean
            if getattr(perf_metrics, "get_features_extraction_duration", None) is not None
            else -1,
            "encode_first_ms": enc_ms[0] if enc_ms else -1,
            "decode_first_ms": dec_ms[0] if dec_ms else -1,
            "decode_other_avg_ms": (sum(dec_ms[1:]) / len(dec_ms[1:])) if len(dec_ms) > 1 else -1,
            "sampling_avg_ms": (sum(smp_ms) / len(smp_ms)) if smp_ms else -1,
            "detokenization_ms": perf_metrics.get_detokenization_duration().mean,
        }
        result_text = result_text.texts[0]
    else:
        whisper_genai_metrics = None
        additional_args = model_utils.setup_gen_config_use_custom_args()
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
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        out_size=out_token_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        prompt_idx=speech_id,
        **memory_metrics,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        tms=tm_list,
        tms_infer=tm_infer_list,
        warm_up=(num == 0),
        prompt_idx=speech_id,
        whisper=whisper_hook,
        whisper_genai=whisper_genai_metrics,
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
    # Build the prompt schedule via BenchPrompter, which:
    #   - reads and parses the speech prompt file (JSON or plain path)
    #   - resolves media paths relative to the prompt file
    #   - honours args['prompt_index'] for selective benchmarking
    #   - handles both subsequent=False (iter-major) and subsequent=True
    #     (prompt-major) scheduling in a single unified iter_schedule() loop
    # NOTE: the raw waveform is loaded lazily inside the loop (not at
    #       prompt-construction time) because decoding depends on the model's
    #       feature-extractor sampling rate, which is only known after the
    #       model has been loaded.
    prompter = BenchPrompter(args)
    speech_idx_list = prompter.active_indices
    speech_list = prompter.active_items

    log.info(
        f"Benchmarking iter nums(exclude warm-up): {num_iters}, "
        f"speech file nums: {len(speech_list)}, speech idx: {speech_idx_list}"
    )
    mem_consumption.update_marker("model")
    pipe, processor, pretrain_time, use_genai = FW_UTILS[framework].create_speech_2_txt_model(
        model_path, device, mem_consumption, **args
    )
    md5_list = {num: {} for num in range(num_iters + 1)}
    iter_timestamp = model_utils.init_timestamp(num_iters, speech_list, speech_idx_list)
    input_param = {
        "pipe": pipe,
        "mem_consumption": mem_consumption,
        "processor": processor,
        "use_genai": use_genai,
    }
    if framework == "ov" and use_genai is False:
        whisper_hook.new_text_encoder(pipe)
        whisper_hook.new_text_encoder_request(pipe)
        whisper_hook.new_generate(pipe)
        whisper_hook.new_text_sample(pipe)
    mem_consumption.activate_cooldown("after model compilation")
    iter_data_list = []
    for num, p_idx, prompt in prompter.iter_schedule(num_iters):
        mem_consumption.update_marker(f"step-{num}-{p_idx}")
        prefix = prompter.get_prefix(num, p_idx)
        prompt.introduce_in_stdout(num, prefix)
        # Load audio waveform here (inside the loop) so that a fresh array is
        # used for every iteration, and because read_wav requires the model's
        # sampling rate which is only available after model creation.
        raw_speech = model_utils.read_wav(prompt["audio"], processor.feature_extractor.sampling_rate)
        input_param["speech_idx"] = p_idx
        input_param["speech_param"] = prompt  # BenchPrompt dict carries language/timestamp
        input_param["iter_idx"] = num
        input_param["raw_speech"] = raw_speech
        iter_timestamp[num][p_idx]["start"] = datetime.datetime.now().isoformat()
        run_speech_2_txt_generation(input_param, args, md5_list, iter_data_list)
        if iter_data_list:
            iter_data_list[-1]["prompt_repr"] = repr(prompt)
        iter_timestamp[num][p_idx]["end"] = datetime.datetime.now().isoformat()
        log.info(
            f"{prefix} start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}"
        )
    metrics_print.print_average(iter_data_list, speech_idx_list, 1, True)
    return iter_data_list, pretrain_time, iter_timestamp
