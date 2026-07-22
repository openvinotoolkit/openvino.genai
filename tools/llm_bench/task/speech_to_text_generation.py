# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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
from llm_bench_utils.hook_forward_whisper import ASRHook

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}
asr_hook = ASRHook()

DEFAULT_OUTPUT_TOKEN_SIZE = 1000


def run_speech_2_txt_generation(input_param, args, md5_list, iter_data_list):
    result_md5_list = []
    pipe = input_param["pipe"]
    raw_speech = input_param["raw_speech"]
    num = input_param["iter_idx"]
    speech_id = input_param["speech_idx"]
    processor = input_param["processor"]
    use_genai = input_param["use_genai"]
    use_case = args["use_case"]

    default_language = "English" if use_case.model_type in ["qwen3-asr"] else "<|en|>"
    speech_language = input_param["speech_param"].get("language", default_language)
    ret_timestamps = input_param["speech_param"].get("timestamp", True)
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]

    perf_kwargs = {}
    mem_consumption = input_param["mem_consumption"]
    mem_consumption.start(num)
    if use_genai:
        generation_config = pipe.get_generation_config()
        generation_config.max_new_tokens = max_gen_tokens
        generation_config.language = speech_language
        generation_config.return_timestamps = ret_timestamps
        if use_case.model_type in ["whisper"]:
            generation_config.task = "translate"

        start = time.perf_counter()
        result_text = pipe.generate(raw_speech, generation_config)
        end = time.perf_counter()
        generation_time = end - start

        perf_metrics = result_text.perf_metrics
        first_token_time = perf_metrics.get_ttft().mean
        second_tokens_durations = (
            np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
            - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1])
        ).tolist()
        tm_list = (np.array([first_token_time] + second_tokens_durations) / 1000).tolist()
        tm_infer_list = (np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000).tolist()

        wm = getattr(perf_metrics, "asr_raw_metrics", None)
        if wm is None:
            wm = getattr(perf_metrics, "whisper_raw_metrics", None)
        enc_ms = (
            [v / 1000 for v in wm.encode_inference_durations]
            if wm is not None and getattr(wm, "encode_inference_durations", None) is not None
            else []
        )
        dec_ms = (
            [v / 1000 for v in wm.decode_inference_durations]
            if wm is not None and getattr(wm, "decode_inference_durations", None) is not None
            else []
        )
        smp_ms = (
            [v / 1000 for v in perf_metrics.raw_metrics.sampling_durations]
            if getattr(perf_metrics, "raw_metrics", None) is not None
            and getattr(perf_metrics.raw_metrics, "sampling_durations", None) is not None
            else []
        )
        asr_genai_metrics = {
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
        asr_genai_metrics = None
        additional_args = model_utils.setup_gen_config_use_custom_args()
        generate_kwargs = {
            "max_new_tokens": max_gen_tokens,
            **additional_args,
        }
        if use_case.model_type == "whisper":
            generate_kwargs.update({"task": "translate", "language": speech_language})
        else:
            generate_kwargs["language"] = speech_language

        start = time.perf_counter()
        output = pipe(raw_speech, generate_kwargs=generate_kwargs, return_timestamps=ret_timestamps)
        end = time.perf_counter()

        if isinstance(output, dict) and "perf_metrics" in output:
            perf_kwargs["tokenization_time"] = (
                output["perf_metrics"]["preprocess_time"],
                output["perf_metrics"]["detokenization_time"],
            )
            generation_time = output["perf_metrics"]["generation_time"]
        else:
            generation_time = end - start

        result_text = output["text"]
        tm_list = asr_hook.get_time_list()
        tm_infer_list = asr_hook.get_time_infer_list()

    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    if tm_infer_list is not None:
        log.debug('latency of all infers:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    out_data = tokenizer(result_text, return_tensors="pt")
    out_tokens = out_data["input_ids"] if "input_ids" in out_data else out_data
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
        **perf_kwargs,
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
        whisper=asr_hook,
        whisper_genai=asr_genai_metrics,
        **perf_kwargs,
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

    if asr_hook is not None:
        asr_hook.clear_statistics()


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
    mem_consumption.update_marker("model")
    pipe, processor, pretrain_time, use_genai = FW_UTILS[framework].create_speech_2_txt_model(model_path, device, mem_consumption, **args)
    md5_list = {num : {} for num in range(num_iters + 1)}
    iter_timestamp = model_utils.init_timestamp(num_iters, speech_list, speech_idx_list)
    input_param = {
        "pipe": pipe,
        "mem_consumption": mem_consumption,
        "processor": processor,
        "use_genai": use_genai,
    }

    if framework == "ov" and use_genai is False:
        asr_hook.new_text_encoder(pipe)
        asr_hook.new_text_encoder_request(pipe)
        asr_hook.new_generate(pipe)
        asr_hook.new_text_sample(pipe)

    sampling_rate = processor.feature_extractor.sampling_rate if hasattr(processor, "feature_extractor") else 16000
    mem_consumption.activate_cooldown("after model compilation")
    for num in range(num_iters + 1):
        for idx, speech_param in enumerate(speech_list):
            p_idx = speech_idx_list[idx]
            mem_consumption.update_marker(f"step-{num}-{p_idx}")
            raw_speech = model_utils.read_wav(speech_param["media"], sampling_rate)
            input_param["speech_idx"] = p_idx
            input_param["speech_param"] = speech_param
            input_param["iter_idx"] = num
            input_param["raw_speech"] = raw_speech
            iter_timestamp[num][p_idx]["start"] = datetime.datetime.now().isoformat()
            run_speech_2_txt_generation(input_param, args, md5_list, iter_data_list)
            iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
            prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
            log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    metrics_print.print_average(iter_data_list, speech_idx_list, 1, True)

    return iter_data_list, pretrain_time, iter_timestamp


def get_speech_files(args):
    speech_file_list = []
    speech_args = dict(args)
    if args.get("media") is None and args.get("prompt_file") is None:
        default_prompt_file = Path(__file__).resolve().parents[1] / "prompts" / "speech_to_text_default.jsonl"
        speech_args["prompt_file"] = [str(default_prompt_file)]
        log.info(f"Default speech prompt file is used: {default_prompt_file}")

    output_data_list, is_json_data = model_utils.get_param_from_file(speech_args, "media")
    if is_json_data is True:
        speech_param_list = parse_json_data.parse_speech_json_data(output_data_list)
        if len(speech_param_list) > 0:
            for speech_file in speech_param_list:
                if speech_args["prompt_file"] is not None and len(speech_args["prompt_file"]) > 0:
                    speech_file["media"] = model_utils.resolve_media_file_path(
                        speech_file.get("media"), speech_args["prompt_file"][0]
                    )
                    if not str(speech_file["media"]).startswith(("http://", "https://")):
                        speech_file["media"] = Path(speech_file["media"])
                speech_file_list.append(speech_file)
    else:
        speech_file_list.append({'media': output_data_list[0]})
    return speech_file_list
