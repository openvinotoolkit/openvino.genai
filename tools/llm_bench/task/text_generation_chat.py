# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import hashlib
import datetime
import numpy as np
import logging as log

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union
from transformers import set_seed, Cache

import llm_bench_utils.ov_utils
from llm_bench_utils.ov_utils import get_genai_chunk_streamer
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


def update_chat_iteration_with_memory_info(chat_iter_data_list: list, memory_metrics: dict):
    for chat_iter_data in chat_iter_data_list:
        mem_vals = {
            key: val
            for key, val in gen_output_data.gen_iterate_data(**memory_metrics).items()
            if (val != "" and val != -1)
        }
        chat_iter_data.update(**mem_vals)


# ===== Optimum-intel/Transformers Utils =====
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


@dataclass
class ChatIterationResult:
    input_size: int
    output_size: int
    generation_time: float
    infer_count: int
    tm_list: list
    tm_infer_list: list
    tokenization_time: Any
    rendered_chat: str
    cache_usage: dict | None = None


class TextGenerationChatAdapter(ABC):
    @abstractmethod
    def run_chat_iteration(self, prompt: str, prefix: str, bench_hook: object) -> ChatIterationResult:
        pass

    @abstractmethod
    def get_messages(self):
        pass

    @abstractmethod
    def init_chat(self):
        pass

    @abstractmethod
    def clear_chat(self):
        pass


class OptimumTextGenerationChatAdapter(TextGenerationChatAdapter):
    def __init__(
        self,
        model: object,
        tokenizer: object,
        args: dict,
        tokens_len: int,
        streaming: bool,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.kv_axes_pos = 2
        if "optimum" in str(type(model)):
            self.kv_axes_pos = get_kv_axes_pos(model.model)

        self.max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]
        self.streaming = streaming
        self.tokens_len = tokens_len
        self.full_chat = args.get("full_chat")

        self.generation_args = None
        self.chat_history = []
        # transformers manage kv_cache via past_key_values
        # for optimum-intel statefull model ((),)
        self.past_key_values = None
        self.tokenized_history: list = []

        set_seed(args["seed"])

    def init_chat(self):
        self.clear_chat()
        self.generation_args = setup_additional_optimum_args(
            self.args, self.model, self.tokenizer, self.streaming, self.tokens_len
        )

    def clear_chat(self):
        self.generation_args = None
        self.chat_history = []
        self.past_key_values = None
        self.tokenized_history: list = []

    def find_common_prefix_length(self, new_tokens: list) -> int:
        kv_cache_len = min(len(new_tokens), len(self.tokenized_history))
        prefix_len = kv_cache_len
        for idx in range(kv_cache_len):
            if new_tokens[idx] != self.tokenized_history[idx]:
                prefix_len = idx
                break

        return prefix_len

    def get_kv_cache_seq_len(self) -> int:
        past_key_values_len = 0
        if self.past_key_values is None:
            return past_key_values_len

        if "transformers" in str(type(self.model)):
            if isinstance(self.past_key_values, Cache):
                if (
                    hasattr(self.past_key_values, "get_seq_length")
                    and self.past_key_values.get_seq_length() is not None
                ):
                    past_key_values_len = self.past_key_values.get_seq_length()
            elif isinstance(self.past_key_values, (tuple, list)) and len(self.past_key_values) > 0:
                past_key_values_len = self.past_key_values[0][0].shape[-2]
        else:
            # kv_cache doesn't include last generated token, len(kv_cache) == len(output tokens) - 1
            past_key_values_len = len(self.tokenized_history) - 1

        return past_key_values_len

    def trim_kv_cache(self, prefix_len: int) -> Union[tuple, "Cache", None]:
        if self.past_key_values is None:
            return None

        if prefix_len == 0:
            if "transformers" in str(type(self.model)):
                return None
            else:
                self.model.request.reset_state()
                return [None]

        if "transformers" in str(type(self.model)):
            if isinstance(self.past_key_values, Cache):
                if hasattr(self.past_key_values, "crop"):
                    self.past_key_values.crop(max_length=prefix_len)
            elif isinstance(self.past_key_values, (tuple, list)) and len(self.past_key_values) > 0:
                trimmed = tuple((k[..., :prefix_len, :], v[..., :prefix_len, :]) for k, v in self.past_key_values)
                self.past_key_values = trimmed
        else:
            self.model._past_length = prefix_len
            import openvino as ov

            states = self.model.request.query_state()
            for state in states:
                old_tensor = state.state
                # [BATCH_SIZE, num_kv_heads, seq_len, head_size]
                data = np.array(old_tensor.data)
                slices = [slice(None)] * data.ndim
                slices[self.kv_axes_pos] = slice(None, prefix_len)
                trimmed_tensor = data[tuple(slices)]

                new_tensor = ov.Tensor(trimmed_tensor)
                state.state = new_tensor

    def run_chat_iteration(self, prompt: str, prefix: str, bench_hook: object) -> ChatIterationResult:
        self.chat_history.append({"role": "user", "content": prompt})

        # ===== Tokenization =====
        templated_input_text = self.tokenizer.apply_chat_template(
            self.chat_history, tokenize=False, add_generation_prompt=True
        )
        tok_encode_start = time.perf_counter()
        tokenized_chat = self.tokenizer(templated_input_text, return_tensors="pt")
        tok_encode_end = time.perf_counter()
        tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
        tokenized_chat.pop("token_type_ids", None)

        full_input_ids = tokenized_chat["input_ids"] if "input_ids" in tokenized_chat else tokenized_chat
        full_input_ids_list = full_input_ids[0].tolist()

        # === Align KV Cache and Tokenized History for models supporting past_key_values ===
        prefix_len = 0
        if len(self.tokenized_history) > 0 and not self.full_chat:
            prefix_len = self.find_common_prefix_length(full_input_ids_list)
            if prefix_len < len(self.tokenized_history):
                self.trim_kv_cache(prefix_len)
            self.tokenized_history = self.tokenized_history[:prefix_len]

        num_new_token_input_size = len(full_input_ids_list) - prefix_len

        if self.past_key_values is not None:
            if "transformers" in str(type(self.model)):
                self.generation_args["past_key_values"] = self.past_key_values
            else:
                # for optimum-intel stateful model past_key_values are not used explicitly, instead they are handled inside the model
                # to avoid taking into account past_key_values, will set it to [None]
                self.generation_args["past_key_values"] = [None]

        # ===== Generation =====
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        start = time.perf_counter()
        result = self.model.generate(
            **tokenized_chat,
            max_new_tokens=int(self.max_gen_tokens),
            num_beams=self.args["num_beams"],
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
            **self.generation_args,
        )
        end = time.perf_counter()
        log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())

        generation_time = end - start
        input_token_size = full_input_ids[0].numel()
        answer_result = result.sequences[0][input_token_size:]

        # Update the KV cache state for the next turn.
        new_past_key_values = getattr(result, "past_key_values", None)
        if new_past_key_values is not None and not self.full_chat:
            self.past_key_values = new_past_key_values
            self.tokenized_history = full_input_ids_list + answer_result.tolist()
            actual_cache_len = self.get_kv_cache_seq_len()
            if actual_cache_len > 0 and len(self.tokenized_history) != actual_cache_len:
                self.tokenized_history = self.tokenized_history[:actual_cache_len]
        else:
            self.past_key_values = None
            self.tokenized_history = []

        # ===== Detokenization =====
        tok_decode_start = time.perf_counter()
        generated_text = self.tokenizer.batch_decode([answer_result])
        tok_decode_end = time.perf_counter()
        tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
        self.chat_history.append({"role": "assistant", "content": generated_text[0]})

        # ===== Performance Data Collection and Print Results =====
        batch_idx = 0
        generated_token_size = calc_generated_token_size_optimum(
            result.sequences, batch_idx, self.model, full_input_ids, input_token_size, self.args["model_name"]
        )
        if generated_token_size > self.max_gen_tokens:
            log.error("Output token size is over max output token size!")

        rendered_chat = self.tokenizer.apply_chat_template(
            self.chat_history, tokenize=False, add_generation_prompt=True
        )

        tm_list = []
        tm_infer_list = []
        if bench_hook is not None:
            tm_list = list(bench_hook.get_time_list())
            log.debug("latency of all tokens:")
            [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]
            tm_infer_list = bench_hook.get_time_infer_list()
            log.debug("latency of all infers:")
            [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
            if self.args["num_beams"] == 1 and generated_token_size != len(tm_infer_list):
                log.warning(
                    f"Output token size({generated_token_size}) is not equal to infer count({len(tm_infer_list)})"
                )

        return ChatIterationResult(
            input_size=num_new_token_input_size,
            output_size=generated_token_size,
            generation_time=generation_time,
            infer_count=len(tm_infer_list),
            tm_list=tm_list,
            tm_infer_list=tm_infer_list,
            tokenization_time=(tok_encode_time, tok_decode_time),
            rendered_chat=rendered_chat,
        )

    def get_messages(self):
        return self.chat_history


class GenAITextGenerationChatAdapter(TextGenerationChatAdapter):
    def __init__(self, model: object, args: dict, streaming: bool, tokens_len: int):
        self.model = model
        self.args = args
        self.streaming = streaming
        self.tokenizer = model.get_tokenizer()
        self.max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args["infer_count"] is None else args["infer_count"]
        self.tokens_len = tokens_len

        self.gen_config = None
        self.adapters_args = None
        self.chat_history = None
        self.chat_iter_data_list = []
        self.chat_token_size = 0
        self.num_input_size = 0
        self.streamer = None

    def setup_generation_config(self):
        gen_config = genai_generation_config_setup(self.model, self.max_gen_tokens, self.args)
        gen_config.ignore_eos = False
        if hasattr(gen_config, "apply_chat_template"):
            gen_config.apply_chat_template = True

        return gen_config

    def init_chat(self):
        import openvino_genai

        self.clear_chat()

        self.gen_config = self.setup_generation_config()
        self.adapters_args = (
            {"adapters": openvino_genai.AdapterConfig()}
            if (self.args["empty_lora"] and (self.gen_config.adapters is not None))
            else {}
        )
        if self.streaming:
            self.streamer = get_genai_chunk_streamer()(self.tokenizer, self.tokens_len)
        self.chat_history = openvino_genai.ChatHistory()

    def clear_chat(self):
        self.gen_config = None
        self.adapters_args = None
        self.chat_history = None
        self.chat_iter_data_list = []
        self.chat_token_size = 0
        self.num_input_size = 0
        self.streamer = None

    def run_chat_iteration(self, prompt: str, prefix: str, bench_hook: object) -> ChatIterationResult:
        self.chat_history.append({"role": "user", "content": prompt})

        # ===== Generation =====
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        start = time.perf_counter()
        decoded_results = self.model.generate(
            self.chat_history,
            generation_config=self.gen_config,
            streamer=self.streamer,
            **self.adapters_args,
        )
        end = time.perf_counter()
        log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
        generation_time = end - start
        self.chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})

        perf_metrics = decoded_results.perf_metrics
        # For GenAI it's impossible to calculate precise token size, but we can calulate approximate number
        num_full_chat_input_tokens = perf_metrics.get_num_input_tokens()
        if self.chat_token_size == 0 and self.args.get("full_chat", False):
            num_input_size = num_full_chat_input_tokens
        else:
            num_input_size = num_full_chat_input_tokens - self.chat_token_size
        num_tokens = perf_metrics.get_num_generated_tokens()
        self.chat_token_size = num_full_chat_input_tokens + num_tokens

        if num_tokens > self.max_gen_tokens:
            log.error("Output token size is over max output token size!")

        first_token_time = perf_metrics.get_ttft().mean - perf_metrics.raw_metrics.tokenization_durations[-1] / 1000
        second_tokens_durations = (np.array(perf_metrics.raw_metrics.m_durations) / 1000).tolist()
        tm_list = (np.array([first_token_time] + second_tokens_durations) / 1000).tolist()
        inference_durations = (np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000).tolist()
        log.debug("latency of all tokens:")
        [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]

        tokenization_time = []
        tokenization_time.append(np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000)
        tokenization_time.append(np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000)

        rendered_chat = self.tokenizer.apply_chat_template(self.chat_history, add_generation_prompt=True)

        cache_usage = None
        if hasattr(self.model, "get_metrics"):
            pipeline_metrics = self.model.get_metrics()
            if hasattr(pipeline_metrics, "avg_cache_usage") and hasattr(pipeline_metrics, "max_cache_usage"):
                cache_usage = {
                    "avg_cache_usage": pipeline_metrics.avg_cache_usage,
                    "max_cache_usage": pipeline_metrics.max_cache_usage,
                }

        return ChatIterationResult(
            input_size=num_input_size,
            output_size=num_tokens,
            generation_time=generation_time,
            infer_count=len(tm_list),
            tm_list=tm_list,
            tm_infer_list=inference_durations,
            tokenization_time=tokenization_time,
            rendered_chat=rendered_chat,
            cache_usage=cache_usage,
        )

    def get_messages(self):
        if self.chat_history is None:
            return []
        return self.chat_history.get_messages()


def run_text_generation_chat_common(
    pipeline: TextGenerationChatAdapter,
    input_text: str | list,
    iter_num: int,
    args: dict,
    iter_data_list: list,
    md5_list: dict,
    chat_index: int,
    bench_hook: object,
    model_precision: str,
    proc_id: int,
    mem_consumption: MemoryMonitorHandler,
    prefix: str,
):
    if args["batch_size"] != 1:
        log.warning("Batch size is not applicable for chat scenario. Parameter will be ignored and set to 1.")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    input_data = get_chat_input_data(input_text, args)
    save_input_data_to_file(input_data, args, model_precision, chat_index, iter_num, proc_id, is_chat=True)

    # ===== Prepare Config, Additional Args and Chat Managing Variables =====
    pipeline.init_chat()

    # ===== Chat Iterations =====
    chat_iter_data_list = []
    mem_consumption.start(iter_num)
    for prompt_index, prompt in enumerate(input_data):
        chat_iteration_result = pipeline.run_chat_iteration(prompt, prefix, bench_hook)

        result_md5_list = []
        result_md5_list.append(
            hashlib.new("md5", chat_iteration_result.rendered_chat.encode(), usedforsecurity=False).hexdigest()
        )
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
            ["\n" + chat_iteration_result.rendered_chat],
            enable_prompt_permutations=False,
            chat_prompts_num=len(input_data),
            chat_idx=chat_index,
        )

        per_token_time = ""
        if chat_iteration_result.output_size > 0:
            per_token_time = chat_iteration_result.generation_time * 1000 / chat_iteration_result.output_size
        else:
            log.warning("No generated tokens")

        iter_data = gen_output_data.gen_iterate_data(
            iter_idx=iter_num,
            in_size=chat_iteration_result.input_size,
            infer_count=chat_iteration_result.infer_count,
            out_size=chat_iteration_result.output_size,
            gen_time=chat_iteration_result.generation_time,
            latency=per_token_time,
            res_md5=result_md5_list,
            prompt_idx=prompt_index,
            tokenization_time=chat_iteration_result.tokenization_time,
            chat_idx=chat_index,
        )
        chat_iter_data_list.append(iter_data)

        metrics_print.print_metrics(
            iter_num,
            iter_data,
            chat_iteration_result.tm_list,
            chat_iteration_result.tm_infer_list,
            warm_up=(iter_num == 0),
            tokenization_time=chat_iteration_result.tokenization_time,
            batch_size=args["batch_size"],
            prompt_idx=prompt_index,
            cb_metric=chat_iteration_result.cache_usage,
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
            pipeline.get_messages(),
            args,
            model_precision,
            chat_index,
            iter_num,
            batchsize_idx=0,
            proc_id=proc_id,
            is_chat=True,
        )

    pipeline.clear_chat()


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
        pipeline = GenAITextGenerationChatAdapter(model=model, args=args, streaming=streaming, tokens_len=tokens_len)
    else:
        pipeline = OptimumTextGenerationChatAdapter(
            model=model, tokenizer=tokenizer, args=args, tokens_len=tokens_len, streaming=streaming
        )

    proc_id = os.getpid()
    iter_alias = "C"
    mem_consumption.activate_cooldown("after model compilation")
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, inputs_idx_list)
    if args["subsequent"] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                chat_idx = inputs_idx_list[idx]
                # Set the logger prefix for current iteration and prompt index
                prefix = f"[warm-up][{iter_alias}{chat_idx}]" if num == 0 else f"[{num}][{iter_alias}{chat_idx}]"
                mem_consumption.update_marker(f"step-{num}-{chat_idx}")
                if num == 0:
                    metrics_print.print_unicode(
                        f"[warm-up][{iter_alias}{chat_idx}] Input text: {input_text}",
                        f"[warm-up][{iter_alias}{chat_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][chat_idx]["start"] = datetime.datetime.now().isoformat()
                run_text_generation_chat_common(
                    pipeline,
                    input_text,
                    num,
                    args,
                    iter_data_list,
                    md5_list,
                    chat_idx,
                    bench_hook,
                    model_precision,
                    proc_id,
                    mem_consumption,
                    prefix,
                )
                iter_timestamp[num][chat_idx]["end"] = datetime.datetime.now().isoformat()
                log.info(
                    f"{prefix} start: {iter_timestamp[num][chat_idx]['start']}, end: {iter_timestamp[num][chat_idx]['end']}"
                )
    else:
        for idx, input_text in enumerate(text_list):
            chat_idx = inputs_idx_list[idx]
            for num in range(num_iters + 1):
                mem_consumption.update_marker(f"step-{num}-{chat_idx}")
                # Set the logger prefix for current iteration and prompt index
                prefix = f"[warm-up][{iter_alias}{chat_idx}]" if num == 0 else f"[{num}][{iter_alias}{chat_idx}]"
                if num == 0:
                    metrics_print.print_unicode(
                        f"[warm-up][{iter_alias}{chat_idx}] Input text: {input_text}",
                        f"[warm-up][{iter_alias}{chat_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][chat_idx]["start"] = datetime.datetime.now().isoformat()
                run_text_generation_chat_common(
                    pipeline,
                    input_text,
                    num,
                    args,
                    iter_data_list,
                    md5_list,
                    chat_idx,
                    bench_hook,
                    model_precision,
                    proc_id,
                    mem_consumption,
                    prefix,
                )
                iter_timestamp[num][chat_idx]["end"] = datetime.datetime.now().isoformat()
                log.info(
                    f"{prefix} start: {iter_timestamp[num][chat_idx]['start']}, end: {iter_timestamp[num][chat_idx]['end']}"
                )

    metrics_print.print_average(iter_data_list, inputs_idx_list, args["batch_size"], True, chat_mode=True)
    return iter_data_list, pretrain_time, iter_timestamp
