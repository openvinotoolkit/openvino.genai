# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import datetime
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
import numpy as np
import hashlib
from transformers import set_seed
import llm_bench_utils.output_file
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import extract_prompt_data
from llm_bench_utils.prompt_utils import get_vlm_prompt
from .text_generation import (
    update_md5_list,
    print_generated_output,
    update_chat_iteration_with_memory_info,
    save_input_data_to_file,
)
from itertools import zip_longest
from llm_bench_utils.memory_monitor import MemoryMonitorHandler


DEFAULT_OUTPUT_TOKEN_SIZE = 512
FW_UTILS = {
    'pt': llm_bench_utils.pt_utils,
    'ov': llm_bench_utils.ov_utils
}


def run_visual_language_generation_optimum(
    inputs: list,
    num: int,
    model: object,
    processor: dict,
    args: dict,
    iter_data_list: list,
    md5_list: list,
    prompt_index: int,
    bench_hook: object,
    model_precision: str,
    proc_id: int,
    mem_consumption: MemoryMonitorHandler,
):
    set_seed(args["seed"])
    if args["batch_size"] != 1:
        log.warning("Only batch size 1 available for benchmarking")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    decim_frames = args["video_frames"]
    prompts, images, videos = extract_prompt_data(inputs, decim_frames, False)
    save_input_data_to_file(prompts, args, model_precision, prompt_index, num, proc_id)
    tok_encode_start = time.perf_counter()

    prefix = "[warm-up]" if num == 0 else "[{}]".format(num)
    log.info(f"{prefix}[P{prompt_index}] Input image nums: {len(images)}")
    log.info(f"{prefix}[P{prompt_index}] Input video nums: {len(videos)}")
    input_data = model.preprocess_inputs(image=images[0] if images else None,
                                         video=videos[0] if videos else None,
                                         text=prompts[0], **processor)

    # ===== Tokenization =====
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000

    # Remove `token_type_ids` from inputs
    input_tokens = input_data["input_ids"] if "input_ids" in input_data else input_data
    input_token_size = input_tokens[0].numel()

    # ===== Prepare Additional Args =====
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]
    additional_args = model_utils.setup_gen_config_use_custom_args()

    if args["infer_count"] is not None and args["end_token_stopping"] is False:
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
        additional_args["eos_token_id"] = None

    # ===== Generation =====
    mem_consumption.start(num)
    start = time.perf_counter()
    result = model.generate(
        **input_data,
        max_new_tokens=int(max_gen_tokens),
        num_beams=args["num_beams"],
        use_cache=True,
        do_sample=False,
        **additional_args,
    )
    end = time.perf_counter()
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    # ===== Detokenization =====
    tok_decode_start = time.perf_counter()
    generated_text = processor["tokenizer"].batch_decode(result[:, input_data["input_ids"].shape[1]:], skip_special_tokens=True)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000

    # ===== Performance Data Collection and Print Results =====
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_token_size = len(result[bs_idx]) - input_data["input_ids"][bs_idx].numel()
        num_tokens += generated_token_size
        if generated_token_size > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    update_md5_list(md5_list, num, result_md5_list, prompt_index)

    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")

    tm_list = []
    tm_infer_list = []
    tm_mm_embeddings = ""
    if bench_hook is not None:
        tm_list = bench_hook.get_time_list()
        tm_mm_embeddings = np.mean(bench_hook.get_mm_embeddings_time_list()) * 1000 * 1000
        log.debug('latency of all tokens:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
        tm_infer_list = bench_hook.get_time_infer_list()
        log.debug('latency of all infers:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
        if args['num_beams'] == 1 and generated_token_size != len(tm_infer_list):
            log.warning(f'Output token size({generated_token_size}) is not equal to infer count({len(tm_infer_list)})')

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_infer_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time),
        mm_embeddings_preparation_time=tm_mm_embeddings,
        **memory_metrics,
    )
    iter_data_list.append(iter_data)

    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )

    print_generated_output(
        prompt_index, num, result_md5_list, md5_list, generated_text, enable_prompt_permutations=False
    )
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()
        bench_hook.clear_mm_embeddins_time_list()


# ===== GenAI Utils =====
def genai_generation_config_setup(model: object, max_gen_tokens: int, args: dict):
    gen_config = model.get_generation_config()
    gen_config.max_new_tokens = max_gen_tokens
    gen_config.num_beams = args["num_beams"]
    gen_config.do_sample = False
    gen_config.ignore_eos = True
    if args["pruning_ratio"] is not None:
        gen_config.pruning_ratio = args["pruning_ratio"]
    if args["relevance_weight"] is not None:
        gen_config.relevance_weight = args["relevance_weight"]

    return gen_config


def run_visual_language_generation_genai(
    inputs: list,
    num: int,
    model: object,
    processor: object,
    args: dict,
    iter_data_list: list,
    md5_list: list,
    prompt_index: int,
    streamer: object,
    model_precision: str,
    proc_id: int,
    mem_consumption: MemoryMonitorHandler,
):
    if args["batch_size"] != 1:
        log.warning("Only batch size 1 available for benchmarking")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    decim_frames = args["video_frames"]
    prompts, images, videos = extract_prompt_data(inputs, decim_frames, True)
    save_input_data_to_file(prompts, args, model_precision, prompt_index, num, proc_id)

    # ===== Setup Generation Config And Additional Args =====
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]
    gen_config = genai_generation_config_setup(model, max_gen_tokens, args)

    kwargs = {}
    prefix = "[warm-up]" if num == 0 else "[{}]".format(num)
    log.info(f"{prefix}[P{prompt_index}] Input image nums: {len(images)}")
    log.info(f"{prefix}[P{prompt_index}] Input video nums: {len(videos)}")

    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos

    # ===== Generation =====
    mem_consumption.start(num)
    start = time.perf_counter()
    generation_result = model.generate(prompts[0], generation_config=gen_config, **kwargs)
    end = time.perf_counter()
    generation_time = end - start
    generated_text = generation_result.texts
    perf_metrics = generation_result.perf_metrics
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    # ===== Performance Data Collection and Print =====
    result_md5_list = []
    generated_text_len = perf_metrics.get_num_generated_tokens()
    if generated_text_len > max_gen_tokens:
        log.error("Output token size is over max output token size!")

    result_text = generated_text[0]
    if args["output_dir"] is not None:
        llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, 0, proc_id)

    result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    update_md5_list(md5_list, num, result_md5_list, prompt_index)

    per_token_time = ""
    if generated_text_len > 0:
        per_token_time = generation_time * 1000 / (generated_text_len / args['batch_size'])
    else:
        log.warning("No generated tokens")
    first_token_time = perf_metrics.get_ttft().mean - perf_metrics.raw_metrics.tokenization_durations[-1] / 1000
    second_tokens_durations = (
        np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
        - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1])
    ).tolist()

    tm_list = np.array([first_token_time] + second_tokens_durations) / 1000
    log.debug("latency of all tokens:")
    [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]
    tokenization_time = (
        np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000,
        np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000,
    )

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=args['batch_size'] * perf_metrics.get_num_input_tokens(),
        infer_count=len(tm_list),
        out_size=generated_text_len,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time,
        mm_embeddings_preparation_time=perf_metrics.get_prepare_embeddings_duration().mean,
        **memory_metrics,
    )
    iter_data_list.append(iter_data)

    inference_durations = np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list.tolist(),
        inference_durations.tolist(),
        warm_up=(num == 0),
        tokenization_time=tokenization_time,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )
    print_generated_output(
        prompt_index, num, result_md5_list, md5_list, generated_text, enable_prompt_permutations=False
    )


def run_visual_language_generation_chat_genai(
    inputs: list,
    num: int,
    model: object,
    processor: dict,
    args: dict,
    iter_data_list: list,
    md5_list: list,
    chat_index: int,
    streamer: object,
    model_precision: str,
    proc_id: int,
    mem_consumption: MemoryMonitorHandler,
):
    import openvino_genai

    if args["batch_size"] != 1:
        log.warning("Only batch size 1 available for benchmarking")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    decim_frames = args["video_frames"]

    prompts, images, videos = extract_prompt_data(inputs, decim_frames, True)
    if not isinstance(prompts[0], list):
        prompts[0] = [prompts[0]] * args["chat_iter"]
    if len(images) > 0 and not isinstance(images[0], list):
        images = [images] * args["chat_iter"]
    if len(videos) > 0 and not isinstance(videos[0], list):
        videos = [videos] * args["chat_iter"]
    save_input_data_to_file(["; ".join(prompts[0])], args, model_precision, chat_index, num, proc_id, is_chat=True)

    # ===== Setup Generation Config And Additional Args =====
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]
    gen_config = genai_generation_config_setup(model, max_gen_tokens, args)
    gen_config.ignore_eos = False

    mem_consumption.start(num)

    # ===== Chat Generation =====
    chat_history = openvino_genai.ChatHistory()
    chat_iter_data_list = []
    for prompt_index, (prompt, input_images, input_videos) in enumerate(zip_longest(prompts[0], images, videos)):
        # ===== Configure kwargs with media content and chat history =====
        kwargs = {}
        prefix = "[warm-up]" if num == 0 else "[{}]".format(num)
        log.info(
            f"{prefix}[C{chat_index}][P{prompt_index}] Input image nums: {len(input_images) if input_images else 0}"
        )
        log.info(
            f"{prefix}[C{chat_index}][P{prompt_index}] Input video nums: {len(input_videos) if input_videos else 0}"
        )

        if input_images:
            kwargs["images"] = input_images
        if input_videos:
            kwargs["videos"] = input_videos

        chat_history.append({"role": "user", "content": prompt})

        # ===== Generation =====
        start = time.perf_counter()
        decoded_results = model.generate(chat_history, generation_config=gen_config, **kwargs)
        end = time.perf_counter()
        generation_time = end - start
        chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})

        # ===== Performance Data Collection and Print Results =====
        perf_metrics = decoded_results.perf_metrics
        generated_text_len = perf_metrics.get_num_generated_tokens()
        if generated_text_len > max_gen_tokens:
            log.error("Output token size is over max output token size!")

        per_token_time = ""
        if generated_text_len > 0:
            per_token_time = generation_time * 1000 / generated_text_len
        else:
            log.warning("No generated tokens")

        first_token_time = perf_metrics.get_ttft().mean - perf_metrics.raw_metrics.tokenization_durations[-1] / 1000
        second_tokens_durations = (
            np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
            - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1])
        ).tolist()
        tm_list = np.array([first_token_time] + second_tokens_durations) / 1000
        log.debug("latency of all tokens:")
        [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]
        tokenization_time = (
            np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000,
            np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000,
        )

        result_md5_list = []
        generated_text = "; ".join(str(replica) for replica in chat_history.get_messages())
        result_md5_list.append(hashlib.new("md5", generated_text.encode(), usedforsecurity=False).hexdigest())
        update_md5_list(md5_list, num, result_md5_list, prompt_index, chat_index)
        print_generated_output(
            prompt_index,
            num,
            result_md5_list,
            md5_list,
            [generated_text],
            enable_prompt_permutations=False,
            chat_prompts_num=len(prompts[0]),
            chat_idx=chat_index,
        )

        iter_data = gen_output_data.gen_iterate_data(
            iter_idx=num,
            in_size=perf_metrics.get_num_input_tokens(),
            infer_count=len(tm_list),
            out_size=generated_text_len,
            gen_time=generation_time,
            latency=per_token_time,
            res_md5=result_md5_list,
            prompt_idx=prompt_index,
            tokenization_time=tokenization_time,
            mm_embeddings_preparation_time=perf_metrics.get_prepare_embeddings_duration().mean,
            chat_idx=chat_index,
        )
        chat_iter_data_list.append(iter_data)

        inference_durations = np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000
        metrics_print.print_metrics(
            num,
            iter_data,
            tm_list.tolist(),
            inference_durations.tolist(),
            warm_up=(num == 0),
            tokenization_time=tokenization_time,
            batch_size=args["batch_size"],
            prompt_idx=prompt_index,
            chat_idx=chat_index,
        )

    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)
    update_chat_iteration_with_memory_info(chat_iter_data_list, memory_metrics)

    metrics_print.print_memory_info(num, iter_data, chat_index)
    iter_data_list.extend(chat_iter_data_list)

    # ===== Print Generated =====
    generated_text = "; ".join(str(replica) for replica in chat_history.get_messages())
    if args["output_dir"] is not None:
        llm_bench_utils.output_file.output_gen_text(
            generated_text, args, model_precision, chat_index, num, batchsize_idx=0, proc_id=proc_id, is_chat=True
        )


def run_visual_language_generation_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    mem_consumption.update_marker("model")
    outs = FW_UTILS[framework].create_image_text_gen_model(model_path, device, mem_consumption, **args)
    model, processor, pretrain_time, bench_hook, use_genai = outs
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_image_text_list = get_vlm_prompt(args)
    if args["prompt_index"] is None:
        iteration_idx_list = list(range(0, len(input_image_text_list)))
        image_text_list = input_image_text_list
    else:
        iteration_idx_list = []
        image_text_list = []
        for i in args["prompt_index"]:
            if 0 <= i < len(input_image_text_list):
                image_text_list.append(input_image_text_list[i])
                iteration_idx_list.append(i)
    if len(input_image_text_list) == 0:
        raise RuntimeError("==Failure prompts is empty ==")
    chat_info = f", chat iteration {args['chat_iter']}" if args.get("chat_iter") else ""
    log.info(
        f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
        f"prompt nums: {len(image_text_list)}, prompt idx: {iteration_idx_list}{chat_info}"
    )

    is_chat_mode = isinstance(input_image_text_list[0].get("prompt"), list) or args["chat_iter"]
    if use_genai:
        if is_chat_mode:
            gen_fn = run_visual_language_generation_chat_genai
        else:
            gen_fn = run_visual_language_generation_genai
    else:
        if is_chat_mode:
            raise RuntimeError("Chat mode for VLM is only supported with GenAI framework. Please use --genai flag.")
        gen_fn = run_visual_language_generation_optimum

    proc_id = os.getpid()
    mem_consumption.activate_cooldown("after model compilation")
    iter_alias = "C" if is_chat_mode else "P"
    iter_timestamp = model_utils.init_timestamp(num_iters, image_text_list, iteration_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(image_text_list):
                mem_consumption.update_marker(f"step-{num}-{idx}")
                p_idx = iteration_idx_list[idx]
                if num == 0:
                    prefix = f"[warm-up][{iter_alias}{p_idx}] Input text: {input_text}"
                    metrics_print.print_unicode(prefix, max_output=metrics_print.MAX_INPUT_TXT_IN_LOG)
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                gen_fn(
                    input_text,
                    num,
                    model,
                    processor,
                    args,
                    iter_data_list,
                    md5_list,
                    iteration_idx_list[idx],
                    bench_hook,
                    model_precision,
                    proc_id,
                    mem_consumption,
                )
                iter_timestamp[num][p_idx]["end"] = datetime.datetime.now().isoformat()
                prefix = f"[warm-up][{iter_alias}{p_idx}]" if num == 0 else f"[{num}][{iter_alias}{p_idx}]"
                log.info(f"{prefix} start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(image_text_list):
            p_idx = iteration_idx_list[idx]
            for num in range(num_iters + 1):
                mem_consumption.update_marker(f"step-{num}-{idx}")
                if num == 0:
                    prefix = f"[warm-up][{iter_alias}{p_idx}] Input text: {input_text}"
                    metrics_print.print_unicode(prefix, max_output=metrics_print.MAX_INPUT_TXT_IN_LOG)
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                gen_fn(
                    input_text,
                    num,
                    model,
                    processor,
                    args,
                    iter_data_list,
                    md5_list,
                    iteration_idx_list[idx],
                    bench_hook,
                    model_precision,
                    proc_id,
                    mem_consumption,
                )
                iter_timestamp[num][p_idx]["end"] = datetime.datetime.now().isoformat()
                prefix = f"[warm-up][{iter_alias}{p_idx}]" if num == 0 else f"[{num}][{iter_alias}{p_idx}]"
                log.info(f"{prefix} start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    metrics_print.print_average(iter_data_list, iteration_idx_list, args["batch_size"], True, chat_mode=is_chat_mode)
    return iter_data_list, pretrain_time, iter_timestamp
