# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import hashlib
import datetime
import numpy as np
import logging as log

from typing import Any, Union
from transformers import set_seed, Cache

import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.output_file
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data

from llm_bench_utils.prompt_utils import get_text_prompt
from llm_bench_utils.memory_monitor import MemoryMonitorHandler
from task.text_generation import (
    save_input_data_to_file,
    setup_additional_optimum_args,
    calc_generated_token_size_optimum,
    print_generated_output,
    genai_generation_config_setup,
)


FW_UTILS = {"pt": llm_bench_utils.pt_utils, "ov": llm_bench_utils.ov_utils}

DEFAULT_OUTPUT_TOKEN_SIZE = 512


# ===== Common Utils =====
def get_chat_input_data(input_text: str | list, args: dict):
    # if prompts are set as list, let's use it
    # if prompt is set as single string, let's create chat where prompt will repeat chat_iter times
    input_data = input_text
    if not isinstance(input_text, list):
        if args.get("chat_iter"):
            input_data = [input_text] * args["chat_iter"]
        else:
            raise RuntimeError("Chat mode can't be started due to incompatible input prompts")
    return input_data


# ===== Optimum-intel/Transformers Utils =====


def find_common_prefix_length(new_tokens: list, tokenized_history: list) -> int:
    kv_cache_len = min(len(new_tokens), len(tokenized_history))
    prefix_len = kv_cache_len
    for idx in range(kv_cache_len):
        if new_tokens[idx] != tokenized_history[idx]:
            prefix_len = idx
            break

    return prefix_len


def get_kv_cache_seq_len(model: Any, past_key_values: Union[tuple, "Cache", None], tokenized_chat_hist: list) -> int:
    past_key_values_len = 0
    if past_key_values is None:
        return past_key_values_len

    if "transformers" in str(type(model)):
        from transformers import Cache

        if isinstance(past_key_values, Cache):
            if hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() is not None:
                past_key_values_len = past_key_values.get_seq_length()
        elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            past_key_values_len = past_key_values[0][0].shape[-2]
    else:
        # kv_cache doesn't include last generated token, len(kv_cache) == len(output tokens) - 1
        past_key_values_len = len(tokenized_chat_hist) - 1

    return past_key_values_len


def trim_kv_cache(
    model, past_key_values: Union[tuple, "Cache", None], prefix_len: int, kv_axes_pos: int = 2
) -> Union[tuple, "Cache", None]:
    if past_key_values is None:
        return None

    if prefix_len == 0:
        if "transformers" in str(type(model)):
            return None
        else:
            model.request.reset_state()
            return [None]

    if "transformers" in str(type(model)):
        from transformers import Cache

        if isinstance(past_key_values, Cache):
            if hasattr(past_key_values, "crop"):
                past_key_values.crop(max_length=prefix_len)
        elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            trimmed = tuple((k[..., :prefix_len, :], v[..., :prefix_len, :]) for k, v in past_key_values)
            past_key_values = trimmed
    else:
        model._past_length = prefix_len
        import openvino as ov

        states = model.request.query_state()
        for state in states:
            old_tensor = state.state
            # [BATCH_SIZE, num_kv_heads, seq_len, head_size]
            data = np.array(old_tensor.data)
            slices = [slice(None)] * data.ndim
            slices[kv_axes_pos] = slice(None, prefix_len)
            trimmed_tensor = data[tuple(slices)]

            new_tensor = ov.Tensor(trimmed_tensor)
            state.state = new_tensor

    return past_key_values


def get_kv_axes_pos(model: Any) -> int:
    # sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
    # therefore usually seq_length_axis = 2
    kv_pos = 2

    # "ReadValue" node is KV cache representation in stateful model
    kv_node_type_name = "ReadValue"

    for op in model.get_ops():
        # check input size, as in LoRA adapters case it could be 0
        if op.get_type_name() != kv_node_type_name or op.get_input_size() < 1:
            continue

        # Shape example: [-1,4,0,64]
        shape = op.get_input_partial_shape(0)
        if shape.rank.is_dynamic or shape.rank.get_length() != 4:
            # kv cache should have 4 dimensions
            continue

        for i in range(shape.rank.get_length()):
            # Find axis = 0. This would be sequence length axis.
            if shape[i] == 0:
                kv_pos = i
                break

    return kv_pos


def update_chat_iteration_with_memory_info(chat_iter_data_list: list, memory_metrics: dict):
    for chat_iter_data in chat_iter_data_list:
        mem_vals = {
            key: val
            for key, val in gen_output_data.gen_iterate_data(**memory_metrics).items()
            if (val != "" and val != -1)
        }
        chat_iter_data.update(**mem_vals)


def run_text_generation_chat_optimum(
    input_text: str,
    iter_num: int,
    model: object,
    tokenizer: object,
    args: dict,
    iter_data_list: list,
    md5_list: list,
    chat_index: int,
    bench_hook: object,
    tokens_len: int,
    streaming: bool,
    model_precision: str,
    proc_id: int,
    mem_consumption: MemoryMonitorHandler,
    prefix: str,
    kv_axes_pos: int = 2,
):
    set_seed(args["seed"])

    if args["batch_size"] != 1:
        log.warning("Only batch size 1 available for benchmarking of chat mode. Fallback to batch_size = 1.")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    input_data = get_chat_input_data(input_text, args)
    save_input_data_to_file(input_data, args, model_precision, chat_index, iter_num, proc_id, is_chat=True)

    # ===== Prepare Backend Specific Args =====
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]
    additional_args = setup_additional_optimum_args(args, model, tokenizer, streaming, tokens_len)

    # ===== Chat Conversation ======
    chat_history = []
    chat_iter_data_list = []

    # transformers manage kv_cache via past_key_values
    # for optimum-intel statefull model ((),)
    past_key_values = None
    tokenized_history: list = []

    past_key_values = None
    prefix_len = 0
    full_chat = args.get("full_chat")

    mem_consumption.start(iter_num)
    for prompt_index, prompt in enumerate(input_data):
        chat_history.append({"role": "user", "content": prompt})

        # ===== Tokenization =====
        templated_input_text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        tok_encode_start = time.perf_counter()
        tokenized_chat = tokenizer(templated_input_text, return_tensors="pt")
        tok_encode_end = time.perf_counter()
        tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
        tokenized_chat.pop("token_type_ids", None)

        full_input_ids = tokenized_chat["input_ids"] if "input_ids" in tokenized_chat else tokenized_chat
        full_input_ids_list = full_input_ids[0].tolist()

        # === Align KV Cache and Tokenized History for models supporting past_key_values ===
        if len(tokenized_history) > 0 and not full_chat:
            prefix_len = find_common_prefix_length(full_input_ids_list, tokenized_history)
            if prefix_len < len(tokenized_history):
                past_key_values = trim_kv_cache(model, past_key_values, prefix_len, kv_axes_pos)
            tokenized_history = tokenized_history[:prefix_len]
        else:
            prefix_len = 0

        num_new_token_input_size = len(full_input_ids_list) - prefix_len

        if past_key_values is not None:
            if "transformers" in str(type(model)):
                additional_args["past_key_values"] = past_key_values
            else:
                # for optimum-intel stateful model past_key_values are not used explicitly, instead they are handled inside the model
                # to avoid taking into account past_key_values, will set it to [None]
                additional_args["past_key_values"] = [None]

        # ===== Generation =====
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        start = time.perf_counter()
        result = model.generate(
            **tokenized_chat,
            max_new_tokens=int(max_gen_tokens),
            num_beams=args["num_beams"],
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
            **additional_args,
        )
        end = time.perf_counter()
        log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
        generation_time = end - start
        input_token_size = full_input_ids[0].numel()
        answer_result = result.sequences[0][input_token_size:]

        # Update the KV cache state for the next turn.
        new_past_key_values = getattr(result, "past_key_values", None)
        if new_past_key_values is not None and not full_chat:
            past_key_values = new_past_key_values
            tokenized_history = full_input_ids_list + answer_result.tolist()
            actual_cache_len = get_kv_cache_seq_len(model, past_key_values, tokenized_history)
            if actual_cache_len > 0 and len(tokenized_history) != actual_cache_len:
                tokenized_history = tokenized_history[:actual_cache_len]
        else:
            past_key_values = None
            tokenized_history = []

        # ===== Detokenization =====
        tok_decode_start = time.perf_counter()
        generated_text = tokenizer.batch_decode([answer_result])
        tok_decode_end = time.perf_counter()
        tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
        chat_history.append({"role": "assistant", "content": generated_text[0]})

        # ===== Performance Data Collection and Print Results =====
        batch_idx = 0
        generated_token_size = calc_generated_token_size_optimum(
            result.sequences, batch_idx, model, full_input_ids, input_token_size, args["model_name"]
        )
        if generated_token_size > max_gen_tokens:
            log.error("Output token size is over max output token size!")

        result_md5_list = []
        generated_text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        result_md5_list.append(hashlib.new("md5", generated_text.encode(), usedforsecurity=False).hexdigest())
        if len(md5_list[iter_num]) == 0:
            md5_list[iter_num] = {}

        if chat_index not in md5_list[iter_num]:
            md5_list[iter_num][chat_index] = {}
        md5_list[iter_num][chat_index][prompt_index] = result_md5_list
        print_generated_output(
            prompt_index,
            iter_num,
            result_md5_list,
            md5_list,
            ["\n" + generated_text],
            enable_prompt_permutations=False,
            chat_prompts_num=len(input_data),
            chat_idx=chat_index,
        )

        per_token_time = ""
        if generated_token_size > 0:
            per_token_time = generation_time * 1000 / generated_token_size
        else:
            log.warning("No generated tokens")

        tm_list = []
        tm_infer_list = []
        if bench_hook is not None:
            tm_list = bench_hook.get_time_list()
            log.debug("latency of all tokens:")
            [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]
            tm_infer_list = bench_hook.get_time_infer_list()
            log.debug("latency of all infers:")
            [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
            if args["num_beams"] == 1 and generated_token_size != len(tm_infer_list):
                log.warning(
                    f"Output token size({generated_token_size}) is not equal to infer count({len(tm_infer_list)})"
                )
        iter_data = gen_output_data.gen_iterate_data(
            iter_idx=iter_num,
            in_size=num_new_token_input_size,
            infer_count=len(tm_infer_list),
            out_size=generated_token_size,
            gen_time=generation_time,
            latency=per_token_time,
            res_md5=result_md5_list,
            prompt_idx=prompt_index,
            tokenization_time=(tok_encode_time, tok_decode_time),
            chat_idx=chat_index,
        )
        chat_iter_data_list.append(iter_data)
        metrics_print.print_metrics(
            iter_num,
            iter_data,
            tm_list,
            tm_infer_list,
            warm_up=(iter_num == 0),
            tokenization_time=(tok_encode_time, tok_decode_time),
            batch_size=args["batch_size"],
            prompt_idx=prompt_index,
            chat_idx=chat_index,
        )

        if bench_hook is not None:
            bench_hook.clear_time_list()
            bench_hook.clear_time_infer_list()

    memory_metrics = mem_consumption.iter_stop_and_collect_data(iter_num)
    update_chat_iteration_with_memory_info(chat_iter_data_list, memory_metrics)

    metrics_print.print_memory_info(iter_num, chat_iter_data_list[-1], chat_index)

    # === Save perf data ===
    iter_data_list.extend(chat_iter_data_list)

    if args["output_dir"] is not None:
        llm_bench_utils.output_file.output_gen_text(
            chat_history, args, model_precision, chat_index, iter_num, batchsize_idx=0, proc_id=proc_id, is_chat=True
        )


def run_text_generation_genai_chat_mode(
    input_text: str | list,
    iter_num: int,
    model: object,
    tokenizer: object,
    args: dict,
    iter_data_list: list,
    md5_list: list,
    chat_index: int,
    streamer: object,
    tokens_len: int,
    streaming: bool,
    model_precision: str,
    proc_id: int,
    mem_consumption: MemoryMonitorHandler,
    prefix: str,
    kv_axes_pos: int = None,
):
    import openvino_genai

    if args["batch_size"] != 1:
        log.warning("Only batch size 1 available for benchmarking of chat mode. Fallback to batch_size = 1.")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    input_data = get_chat_input_data(input_text, args)
    save_input_data_to_file(input_data, args, model_precision, chat_index, iter_num, proc_id, is_chat=True)

    # ===== Setup Generation Config and Additional Args =====
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]
    gen_config = genai_generation_config_setup(model, max_gen_tokens, args)
    gen_config.ignore_eos = False
    if hasattr(gen_config, "apply_chat_template"):
        gen_config.apply_chat_template = True
    additional_args = (
        {"adapters": openvino_genai.AdapterConfig()}
        if (args["empty_lora"] and (gen_config.adapters is not None))
        else {}
    )

    # ===== Chat Conversation =====
    chat_history = openvino_genai.ChatHistory()
    chat_iter_data_list = []
    chat_token_size = 0
    num_input_size = 0
    tokenizer = model.get_tokenizer()

    mem_consumption.start(iter_num)
    for prompt_index, prompt in enumerate(input_data):
        chat_history.append({"role": "user", "content": prompt})

        # ===== Generation =====
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        start = time.perf_counter()
        decoded_results = model.generate(
            chat_history,
            generation_config=gen_config,
            streamer=streamer,
            **additional_args,
        )
        end = time.perf_counter()
        log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
        generation_time = end - start
        chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})

        # ===== Performance Data Collection and Print Results =====
        perf_metrics = decoded_results.perf_metrics

        # For GenAI it's impossible to calculate precise token size, but we can calulate approximate number
        num_full_chat_input_tokens = perf_metrics.get_num_input_tokens()
        if chat_token_size == 0 and args.get("full_chat", False):
            num_input_size = num_full_chat_input_tokens
        else:
            num_input_size = num_full_chat_input_tokens - chat_token_size
        chat_token_size = num_full_chat_input_tokens + perf_metrics.get_num_generated_tokens()

        per_token_time = ""
        num_tokens = perf_metrics.get_num_generated_tokens()
        if num_tokens > max_gen_tokens:
            log.error("Output token size is over max output token size!")
        if num_tokens > 0:
            per_token_time = generation_time * 1000 / num_tokens
        else:
            log.warning("No generated tokens")
        first_token_time = perf_metrics.get_ttft().mean - perf_metrics.raw_metrics.tokenization_durations[-1] / 1000
        second_tokens_durations = (np.array(perf_metrics.raw_metrics.m_durations) / 1000).tolist()
        tm_list = (np.array([first_token_time] + second_tokens_durations) / 1000).tolist()
        inference_durations = (np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000).tolist()
        log.debug("latency of all tokens:")
        [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]

        tokenization_time = []
        tokenization_time.append(np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000)
        tokenization_time.append(np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000)

        result_md5_list = []
        generated_text = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True)
        result_md5_list.append(hashlib.new("md5", generated_text.encode(), usedforsecurity=False).hexdigest())
        if len(md5_list[iter_num]) == 0:
            md5_list[iter_num] = {}
        if chat_index not in md5_list[iter_num]:
            md5_list[iter_num][chat_index] = {}
        md5_list[iter_num][chat_index][prompt_index] = result_md5_list
        print_generated_output(
            prompt_index,
            iter_num,
            result_md5_list,
            md5_list,
            [generated_text],
            enable_prompt_permutations=False,
            chat_prompts_num=len(input_data),
            chat_idx=chat_index,
        )

        cache_usage = None
        if hasattr(model, "get_metrics"):
            pipeline_metrics = model.get_metrics()
            if hasattr(pipeline_metrics, "avg_cache_usage") and hasattr(pipeline_metrics, "max_cache_usage"):
                cache_usage = {
                    "avg_cache_usage": pipeline_metrics.avg_cache_usage,
                    "max_cache_usage": pipeline_metrics.max_cache_usage,
                }
        iter_data = gen_output_data.gen_iterate_data(
            iter_idx=iter_num,
            in_size=num_input_size,
            infer_count=len(tm_list),
            out_size=num_tokens,
            gen_time=generation_time,
            latency=per_token_time,
            res_md5=result_md5_list,
            prompt_idx=prompt_index,
            tokenization_time=tokenization_time,
            chat_idx=chat_index,
        )
        chat_iter_data_list.append(iter_data)

        metrics_print.print_metrics(
            iter_num,
            iter_data,
            tm_list,
            inference_durations,
            warm_up=(iter_num == 0),
            tokenization_time=tokenization_time,
            batch_size=args["batch_size"],
            prompt_idx=prompt_index,
            cb_metric=cache_usage,
            chat_idx=chat_index,
        )

    memory_metrics = mem_consumption.iter_stop_and_collect_data(iter_num)
    update_chat_iteration_with_memory_info(chat_iter_data_list, memory_metrics)
    metrics_print.print_memory_info(iter_num, chat_iter_data_list[-1], chat_index)

    # === Save perf data ===
    iter_data_list.extend(chat_iter_data_list)

    if args["output_dir"] is not None:
        llm_bench_utils.output_file.output_gen_text(
            chat_history.get_messages(),
            args,
            model_precision,
            chat_index,
            iter_num,
            batchsize_idx=0,
            proc_id=proc_id,
            is_chat=True,
        )


def run_text_generation_benchmark(
    model_path, framework, device, tokens_len, streaming, args, num_iters, mem_consumption
):
    mem_consumption.update_marker("model")
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_gen_model(
        model_path, device, mem_consumption, **args
    )
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num: {} for num in range(num_iters + 1)}
    input_text_list = get_text_prompt(args)
    if args["prompt_index"] is None:
        inputs_idx_list = [prompt_idx for prompt_idx, input_text in enumerate(input_text_list)]
        text_list = input_text_list
    else:
        inputs_idx_list = []
        text_list = []
        for i in args["prompt_index"]:
            if 0 <= i < len(input_text_list):
                text_list.append(input_text_list[i])
                inputs_idx_list.append(i)

    if len(input_text_list) == 0 or any(len(chat_prompts_list) == 0 for chat_prompts_list in input_text_list):
        raise RuntimeError("==Failure prompts is empty ==")
    chat_info = f", chat iteration {args['chat_iter']}" if args.get("chat_iter") else ""
    log.info(
        f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
        f"input nums: {len(text_list)}, input idx: {inputs_idx_list}{chat_info}"
    )

    # if num_iters == 0, just output warm-up data
    if use_genai:
        text_gen_fn = run_text_generation_genai_chat_mode
    else:
        text_gen_fn = run_text_generation_chat_optimum

    kv_axes_pos = -1
    if "optimum" in str(type(model)):
        kv_axes_pos = get_kv_axes_pos(model.model)

    proc_id = os.getpid()
    iter_alias = "C"
    mem_consumption.activate_cooldown("after model compilation")
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, inputs_idx_list)
    if args["subsequent"] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                p_idx = inputs_idx_list[idx]
                # Set the logger prefix for current iteration and prompt index
                prefix = f"[warm-up][{iter_alias}{p_idx}]" if num == 0 else f"[{num}][{iter_alias}{p_idx}]"
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                if num == 0:
                    metrics_print.print_unicode(
                        f"[warm-up][{iter_alias}{p_idx}] Input text: {input_text}",
                        f"[warm-up][{iter_alias}{p_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][p_idx]["start"] = datetime.datetime.now().isoformat()
                text_gen_fn(
                    input_text,
                    num,
                    model,
                    tokenizer,
                    args,
                    iter_data_list,
                    md5_list,
                    p_idx,
                    bench_hook,
                    tokens_len,
                    streaming,
                    model_precision,
                    proc_id,
                    mem_consumption,
                    prefix,
                    kv_axes_pos,
                )
                iter_timestamp[num][p_idx]["end"] = datetime.datetime.now().isoformat()
                log.info(
                    f"{prefix} start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}"
                )
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = inputs_idx_list[idx]
            for num in range(num_iters + 1):
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                # Set the logger prefix for current iteration and prompt index
                prefix = f"[warm-up][{iter_alias}{p_idx}]" if num == 0 else f"[{num}][{iter_alias}{p_idx}]"
                if num == 0:
                    metrics_print.print_unicode(
                        f"[warm-up][{iter_alias}{p_idx}] Input text: {input_text}",
                        f"[warm-up][{iter_alias}{p_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][p_idx]["start"] = datetime.datetime.now().isoformat()
                text_gen_fn(
                    input_text,
                    num,
                    model,
                    tokenizer,
                    args,
                    iter_data_list,
                    md5_list,
                    p_idx,
                    bench_hook,
                    tokens_len,
                    streaming,
                    model_precision,
                    proc_id,
                    mem_consumption,
                    prefix,
                    kv_axes_pos,
                )
                iter_timestamp[num][p_idx]["end"] = datetime.datetime.now().isoformat()
                log.info(
                    f"{prefix} start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}"
                )

    metrics_print.print_average(iter_data_list, inputs_idx_list, args["batch_size"], True, chat_mode=True)
    return iter_data_list, pretrain_time, iter_timestamp
