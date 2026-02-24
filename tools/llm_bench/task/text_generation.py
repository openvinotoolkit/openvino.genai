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
import threading
import llm_bench_utils.metrics_print as metrics_print
from transformers import set_seed
from llm_bench_utils.ov_utils import get_genai_chunk_streamer, OptimumChunkStreamer
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import get_text_prompt
from llm_bench_utils.memory_monitor import MemoryDataSummarizer

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_OUTPUT_TOKEN_SIZE = 512


# ===== Common Utils =====
def save_input_data_to_file(input_text_list: list, args: dict, model_precision: str, prompt_index: int, iter_num: int, proc_id: int):
    if args["output_dir"] is not None and iter_num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)


def print_generated_output(prompt_index: int, iter_num: int, result_md5_list: list, md5_list: list, generated_text: list,
                           enable_prompt_permutations: bool = False, chat_prompts_num: int = None, chat_idx: int = None):
    if iter_num > 0 and not enable_prompt_permutations:
        prev_md5 = md5_list[iter_num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{iter_num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {iter_num - 1} iteration {prev_md5}")
            metrics_print.print_generated(iter_num, warm_up=(iter_num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    else:
        if chat_prompts_num is None or chat_prompts_num - 1 == prompt_index:
            metrics_print.print_generated(iter_num, warm_up=(iter_num == 0), generated=generated_text[0], prompt_idx=prompt_index, chat_idx=chat_idx)


def print_input_info(args: dict, iter_num: int, input_token_size: int):
    if args['batch_size'] <= 1:
        return

    out_str = '[warm-up]' if iter_num == 0 else '[{}]'.format(iter_num)
    out_str += " Batch_size={}, ".format(args['batch_size'])
    out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
    if args['infer_count'] is not None:
        out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
    log.info(out_str)


def update_md5_list(md5_list: list, iter_num: int, result_md5_list: list, prompt_index: int, chat_index: int = None):
    if len(md5_list[iter_num]) == 0:
        md5_list[iter_num] = {}

    if chat_index is not None:
        if chat_index not in md5_list[iter_num]:
            md5_list[iter_num][chat_index] = {}
        md5_list[iter_num][chat_index][prompt_index] = result_md5_list
    else:
        md5_list[iter_num][prompt_index] = result_md5_list


def update_chat_iteration_with_memory_info(chat_iter_data_list: list, memory_metrics: dict):
    for iter in chat_iter_data_list:
        mem_vals = {key: val for key, val in gen_output_data.gen_iterate_data(**memory_metrics).items() if (val != '' and val != -1)}
        iter.update(**mem_vals)


# ===== Optimum Utils =====
def setup_additional_args_optimum(args: dict, model: object, tokenizer: object, streaming, tokens_len: int):
    from optimum.intel.utils.import_utils import is_transformers_version

    additional_args = model_utils.setup_gen_config_use_custom_args()

    # llama-3-8b-instruct's generation_config.json has 4096 max_length.
    # This is too small because test prompt may contain 4096 tokens which leaves no space for new tokens.
    # Override it to preserve max_new_tokens.
    additional_args["max_length"] = 2**64 - 1
    if streaming:
        additional_args["streamer"] = OptimumChunkStreamer(tokenizer, tokens_len=tokens_len)

    if args['infer_count'] is not None and args['end_token_stopping'] is False:
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
        additional_args["eos_token_id"] = None

    return additional_args


def calc_generated_token_size_optimum(result: list, batch_idx: int, model: object, 
                                      input_tokens: list, input_token_size: int, model_name: str):
    if 'sum' not in model_name and result[batch_idx][:input_token_size].equal(input_tokens[batch_idx]):
        generated_token_size = len(result[batch_idx]) - input_tokens[batch_idx].numel()
    else:
        generated_token_size = len(result[batch_idx])
    # Encoder-decoder models expect the `decoder_input_ids` to start with a special token
    # When counting the output length, subtract 1. The last token does not participate in inference.
    if model.config.is_encoder_decoder and result[batch_idx][0] == model.config.decoder_start_token_id:
        generated_token_size = generated_token_size - 1

    return generated_token_size


def get_chat_input_data(input_text: str | list, args: dict):
    # if prompts are set as list, let's use it
    # if prompt are set as single string, let's create chat where prompt will repeate chat_iter times
    input_data = input_text
    if not isinstance(input_text, list):
        input_data = [input_text] * args["chat_iter"]
    return input_data


# ===== GENERATION FUNCTIONS FOR OPTIMUM =====
def run_text_generation_optimum(input_text: str, num: int, model: object, tokenizer: object, args: dict, iter_data_list: list,
                                md5_list: list, prompt_index: int, bench_hook: object, tokens_len: int, streaming: bool, 
                                model_precision: str, proc_id: int, mem_consumption: MemoryDataSummarizer, prefix: str):
    set_seed(args['seed'])

    # ===== Prepare Input Data =====
    input_text_list = [input_text] * args['batch_size']
    save_input_data_to_file(input_text_list, args, model_precision, prompt_index, num, proc_id)

    if args["apply_chat_template"]:
        input_text_hist = [{'role': 'user', 'content': input_text}]
        templated_input_text = tokenizer.apply_chat_template(input_text_hist, tokenize=False, add_generation_prompt=True)
        input_text_list = [templated_input_text] * args['batch_size']

    # ===== Tokenization =====
    tok_encode_start = time.perf_counter()
    input_data = tokenizer(input_text_list, return_tensors='pt')
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    input_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
    input_token_size = input_tokens[0].numel()

    print_input_info(args, num, input_token_size)

    # ===== Prepare Additional Args =====
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    additional_args = setup_additional_args_optimum(args, model, tokenizer, streaming, tokens_len)

    # === Generate ===
    mem_consumption.start()
    log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
    start = time.perf_counter()
    result = model.generate(
        **input_data,
        max_new_tokens=int(max_gen_tokens),
        num_beams=args['num_beams'],
        use_cache=True,
        do_sample=False,
        **additional_args
    )
    end = time.perf_counter()
    log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    # === Detokenization ===
    tok_decode_start = time.perf_counter()
    generated_text = tokenizer.batch_decode(result)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000

    # ===== Performance Data Collection and Print Results =====
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_token_size = calc_generated_token_size_optimum(result, bs_idx, model, input_tokens, input_token_size, args["model_name"])
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
    if bench_hook is not None:
        tm_list = bench_hook.get_time_list()
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
    print_generated_output(prompt_index, num, result_md5_list, md5_list, generated_text, enable_prompt_permutations=False)
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()


def run_text_generation_chat_optimum(input_text: str | list, num: int, model: object, tokenizer: object, args: dict, iter_data_list: list,
                                     md5_list: list, chat_index: int, bench_hook: object, tokens_len: int, streaming: bool,
                                     model_precision: str, proc_id: int, mem_consumption: MemoryDataSummarizer, prefix: str):
    set_seed(args['seed'])

    if args["batch_size"] != 1:
        log.warning("Only batch size 1 available for benchmarking")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    input_data = get_chat_input_data(input_text, args)
    save_input_data_to_file(["; ".join(input_data)], args, model_precision, chat_index, num, proc_id)

    # ===== Prepare Additional Args =====
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    additional_args = setup_additional_args_optimum(args, model, tokenizer, streaming, tokens_len)

    mem_consumption.start()

    # ===== Chat Conversation ======
    chat_history = []
    chat_iter_data_list = []
    for prompt_index, prompt in enumerate(input_data):
        chat_history.append({'role': 'user', 'content': prompt})
        templated_input_text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

        # ===== Tokenization =====
        tok_encode_start = time.perf_counter()
        input_data = tokenizer(templated_input_text, return_tensors='pt')
        tok_encode_end = time.perf_counter()
        tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
        input_data.pop('token_type_ids', None)
        # Remove `token_type_ids` from inputs
        input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
        input_token_size = input_tokens[0].numel()

        # === Generation ===
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        start = time.perf_counter()
        result = model.generate(
            **input_data,
            max_new_tokens=int(max_gen_tokens),
            num_beams=args['num_beams'],
            use_cache=True,
            do_sample=False,
            **additional_args
        )
        end = time.perf_counter()
        log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
        generation_time = end - start
        answer_result = result[0][input_data["input_ids"].shape[-1]:]

        # ===== Detokenization =====
        tok_decode_start = time.perf_counter()
        generated_text = tokenizer.batch_decode([answer_result])
        tok_decode_end = time.perf_counter()
        tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
        chat_history.append({'role': 'assistant', 'content': generated_text[0]})

        # ===== Performance Data Collection and Print Results =====
        batch_idx = 0
        generated_token_size = calc_generated_token_size_optimum(result, batch_idx, model, input_tokens, input_token_size, args["model_name"])
        if generated_token_size > max_gen_tokens:
            log.error('Output token size is over max output token size!')

        result_md5_list = []
        generated_text = "; ".join(str(replica) for replica in chat_history)
        result_md5_list.append(hashlib.new("md5", generated_text.encode(), usedforsecurity=False).hexdigest())
        update_md5_list(md5_list, num, result_md5_list, prompt_index, chat_index=chat_index)
        print_generated_output(prompt_index, num, result_md5_list, md5_list, generated_text,
                               enable_prompt_permutations=False, chat_prompts_num=len(input_data), chat_idx=chat_index)

        per_token_time = ""
        if generated_token_size > 0:
            per_token_time = generation_time * 1000 / generated_token_size
        else:
            log.warning("No generated tokens")

        tm_list = []
        tm_infer_list = []
        if bench_hook is not None:
            tm_list = bench_hook.get_time_list()
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
            num,
            iter_data,
            tm_list,
            tm_infer_list,
            warm_up=(num == 0),
            tokenization_time=(tok_encode_time, tok_decode_time),
            batch_size=args['batch_size'],
            prompt_idx=prompt_index,
            chat_idx=chat_index
        )

    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)
    # === ADD MEMORY METRICS for all chat iterartion ====
    for iter in chat_iter_data_list:
        mem_vlas = { key: val for key, val in gen_output_data.gen_iterate_data(**memory_metrics).items() if (val != '' and val != -1) }
        iter.update(**mem_vlas)
    update_chat_iteration_with_memory_info(chat_iter_data_list, memory_metrics)
        
    # === Save perf data ===
    iter_data_list.extend(chat_iter_data_list)

    generated_text = "; ".join(str(replica) for replica in chat_history)
    if args["output_dir"] is not None:
        llm_bench_utils.output_file.output_gen_text(generated_text, args, model_precision, chat_index, num, batchsize_idx=0, proc_id=proc_id, is_chat=True)

    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()


# ===== GenAI Utils =====
def apply_chat_template_genai(args: dict, input_text: str, tokenizer: object):
    input_text_list = input_text
    if args['apply_chat_template']:
        input_text_hist = [{'role': 'user', 'content': input_text}]
        templated_input_text = tokenizer.apply_chat_template(input_text_hist, add_generation_prompt=True)
        input_text_list = [templated_input_text] * args['batch_size']
        if not args["disable_prompt_permutation"]:
            log.warning(
                "Enabled chat template applying and permutation of input prompt. "
                "It means that after applying the chat template prompt will be tokenized and mixed, so the structure of chat template will not be kept. "
                "If it is not expected, please specify --disable_prompt_permutation in your benchmarking command to disable this behavior"
            )
    return input_text_list


def genai_generation_config_setup(model: object, max_gen_tokens: int, args: dict):
    from openvino_genai import GenerationConfig
    gen_config = model.get_generation_config() if hasattr(model, 'get_generation_config') else GenerationConfig()
    gen_config.max_new_tokens = max_gen_tokens
    # llama-3-8b-instruct's generation_config.json has 4096 max_length.
    # This is too small because test prompt may contain 4096 tokens which leaves no space for new tokens.
    # Override it to preserve max_new_tokens.
    gen_config.max_length = 2**64 - 1
    gen_config.ignore_eos = True
    gen_config.rng_seed = args["seed"]
    gen_config.num_beams = args["num_beams"]
    gen_config.do_sample = False
    if gen_config.num_beams > 1:
        gen_config.frequency_penalty = 0
        gen_config.presence_penalty = 0
        gen_config.repetition_penalty = 1
    if hasattr(gen_config, 'apply_chat_template'):
        gen_config.apply_chat_template = False
    if args.get('draft_model', ''):
        config_info = "Speculative decoding config: "
        if args.get('num_assistant_tokens', None):
            gen_config.num_assistant_tokens = int(args['num_assistant_tokens'])
            config_info += f" num_assistant_tokens {gen_config.num_assistant_tokens}"
        if args.get('assistant_confidence_threshold', None):
            gen_config.assistant_confidence_threshold = float(args['assistant_confidence_threshold'])
            config_info += f" assistant_confidence_threshold {gen_config.assistant_confidence_threshold}"
        log.info(config_info)
    if args.get('max_ngram_size') and args.get('num_assistant_tokens'):
        config_info = "Prompt Lookup decoding config: "
        gen_config.max_ngram_size = int(args['max_ngram_size'])
        gen_config.num_assistant_tokens = int(args['num_assistant_tokens'])
        config_info += f"max_ngram_size {gen_config.max_ngram_size}, num_assistant_tokens {gen_config.num_assistant_tokens}"
        log.info(config_info)

    return gen_config


def genai_generate(streaming: bool, model: object, tokens_len: int, gen_config: object, empty_lora: bool, input_data: list, batch_size: int, prefix: str):
    import openvino_genai
    import openvino as ov
    cb_pipeline = isinstance(model, openvino_genai.ContinuousBatchingPipeline)
    if cb_pipeline:
        input_data = [ov.Tensor([input]) for input in input_data.input_ids.data]
        gen_config = [gen_config] * batch_size

    additional_args = {"adapters": openvino_genai.AdapterConfig()} if (empty_lora and (gen_config.adapters is not None)) else {}
    log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
    start = time.perf_counter()
    if streaming:
        text_print_streamer = get_genai_chunk_streamer()(model.get_tokenizer(), tokens_len)

        def token_printer():
            # Getting next elements from iterable will be blocked until a new token is available.
            for word in text_print_streamer:
                print(word, end='', flush=True)

        printer_thread = threading.Thread(target=token_printer, daemon=True)
        printer_thread.start()
        generation_result = model.generate(
            input_data,
            gen_config,
            streamer=text_print_streamer,
            **additional_args
        )
        printer_thread.join()
    else:
        generation_result = model.generate(input_data, gen_config, **additional_args)
    end = time.perf_counter()
    log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
    generated_tokens = []
    if cb_pipeline:
        for res in generation_result:
            generated_tokens.append(res.m_generation_ids[0])
        generated_tokens = np.array(generated_tokens)
    else:
        generated_tokens = np.array(generation_result.tokens)

    perf_metrics = generation_result[0].perf_metrics if cb_pipeline else generation_result.perf_metrics
    return generated_tokens, perf_metrics, end - start


def run_text_generation_genai(input_text: str, num: int, model: object, tokenizer: object, args: dict, iter_data_list: list, md5_list: list, prompt_index: int,
                              streamer: object, tokens_len: int, streaming: bool, model_precision: str, proc_id: int, mem_consumption: object, prefix: str):
    # ===== Prepare Input Data =====
    input_text_list = [input_text] * args['batch_size']
    save_input_data_to_file(input_text_list, args, model_precision, prompt_index, num, proc_id)

    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    input_text_list = apply_chat_template_genai(args, input_text, tokenizer)

    # ===== Tokenization =====
    tokenization_start = time.perf_counter()
    input_data = tokenizer.encode(input_text_list)
    tokenization_end = time.perf_counter()
    tokenization_time = [(tokenization_end - tokenization_start) * 1000]

    # ===== Prompt permutation and config setup =====
    enable_prompt_permutations = not args.get("disable_prompt_permutation", False)
    if enable_prompt_permutations:
        log.warning(
            "Enabled input prompt permutations. It means that generated results may vary on different steps. "
            "If it is not expected, please specify --disable_prompt_permutation in your benchmarking command to disable this behavior"
        )
        from openvino_genai import TokenizedInputs
        import openvino as ov

        input_ids = input_data.input_ids.data
        if tokenizer.get_bos_token_id() == -1:
            input_ids[:, 0] = num + 1
        else:
            if tokenizer.get_eos_token_id() != num + 1:
                input_ids[:, 1] = num + 1
            else:
                input_ids[:, 1] = num + 3
        attention_mask = input_data.attention_mask
        input_data = TokenizedInputs(input_ids=ov.Tensor(input_ids), attention_mask=attention_mask)
    num_input_tokens = input_data.input_ids.shape[1]
    print_input_info(args, num, num_input_tokens)
    gen_config = genai_generation_config_setup(model, max_gen_tokens, args)

    mem_consumption.start()

    # ===== Generate =====
    generated_tokens, perf_metrics, generation_time = genai_generate(streaming, model, tokens_len, gen_config,
                                                                     args["empty_lora"], input_data, args['batch_size'], prefix)

    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    # ===== Detokenization =====
    if streaming:
        tokenization_time.append(np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000)
        generated_text = tokenizer.decode(generated_tokens)
    else:
        detokenization_start = time.perf_counter()
        generated_text = tokenizer.decode(generated_tokens)
        detokenization_end = time.perf_counter()
        tokenization_time.append((detokenization_end - detokenization_start) * 1000)

    # ===== Performance Data Collection and Print Results =====
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_text_len = generated_tokens[bs_idx].shape[-1]
        num_tokens += generated_text_len
        if generated_text_len > max_gen_tokens:
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
    first_token_time = (perf_metrics.get_ttft().mean)
    second_tokens_durations = (np.array(perf_metrics.raw_metrics.m_durations) / 1000).tolist()
    tm_list = (np.array([first_token_time] + second_tokens_durations) / 1000).tolist()
    inference_durations = (np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000).tolist()
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]

    cache_usage = None
    if hasattr(model, 'get_metrics'):
        pipeline_metrics = model.get_metrics()
        if hasattr(pipeline_metrics, 'avg_cache_usage') and hasattr(pipeline_metrics, 'max_cache_usage'):
            cache_usage = {"avg_cache_usage": pipeline_metrics.avg_cache_usage, "max_cache_usage": pipeline_metrics.max_cache_usage}

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=num_input_tokens * args['batch_size'],
        infer_count=len(tm_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time,
        **memory_metrics,
    )
    iter_data_list.append(iter_data)

    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        inference_durations,
        warm_up=(num == 0),
        tokenization_time=tokenization_time,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index,
        cb_metric=cache_usage
    )

    print_generated_output(prompt_index, num, result_md5_list, md5_list, generated_text, enable_prompt_permutations)


def run_text_generation_genai_with_stream(input_text: str, num: int, model: object, tokenizer: object, args: dict, iter_data_list: list, md5_list: list,
                                          prompt_index: int, streamer: object, tokens_len: int, streaming: bool, model_precision: str, proc_id: int, mem_consumption: object, prefix: str):
    # ===== Prepare Input Data =====
    input_text_list = [input_text] * args['batch_size']
    save_input_data_to_file(input_text_list, args, model_precision, prompt_index, num, proc_id)

    # ===== Tokenization =====
    pipe_tokenizer = model.get_tokenizer()
    input_text_list = apply_chat_template_genai(args, input_text, pipe_tokenizer)
    tok_encode_start = time.perf_counter()
    input_data = pipe_tokenizer.encode(input_text_list)
    tok_encode_end = time.perf_counter()
    input_token_size = input_data.input_ids.shape[1]
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000

    print_input_info(args, num, input_token_size)

    # === Prepare Generation Config and Streamer ===
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    streamer.reset()
    gen_config = genai_generation_config_setup(model, max_gen_tokens, args)
    enable_prompt_permutations = not args.get("disable_prompt_permutation", False)
    if enable_prompt_permutations:
        log.warning(
            "Enabled input prompt permutations. It means that generated results may vary on different steps. "
            "If it is not expected, please specify --disable_prompt_permutation in your benchmarking command to disable this behavior"
        )
        from openvino_genai import TokenizedInputs
        import openvino as ov

        input_ids = input_data.input_ids.data
        input_ids[:, 0] = num + 1
        attention_mask = input_data.attention_mask
        input_data = TokenizedInputs(input_ids=ov.Tensor(input_ids), attention_mask=attention_mask)

    mem_consumption.start(num)

    # === Generate ===
    log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
    start = time.perf_counter()
    generated_tokens = model.generate(input_data, gen_config, streamer=streamer).tokens
    end = time.perf_counter()
    log.info("%s Text generation end: %s", prefix, datetime.datetime.now().isoformat())
    generation_time = end - start

    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    # ===== Detokenization =====
    tok_decode_start = time.perf_counter()
    generated_text = pipe_tokenizer.decode(generated_tokens)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000

    # ===== Performance Data Collection and Print Results =====
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_text_len = len(generated_tokens[bs_idx])
        num_tokens += generated_text_len
        if generated_text_len > max_gen_tokens:
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
    tm_list = streamer.get_time_list()
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time),
        **memory_metrics,
    )
    iter_data_list.append(iter_data)

    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        tms=tm_list,
        tms_infer=None,
        warm_up=(num == 0),
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )

    print_generated_output(prompt_index, num, result_md5_list, md5_list, generated_text, enable_prompt_permutations)
    streamer.reset()


def run_text_generation_genai_chat_mode(input_text: str | list, iter_num: int, model: object, tokenizer: object, args: dict, iter_data_list: list, md5_list: list,
                                        chat_index: int, streamer: object, tokens_len: int, streaming: bool, model_precision: str, proc_id: int, mem_consumption: MemoryDataSummarizer, prefix: str):
    import openvino_genai

    if args["batch_size"] != 1:
        log.warning("Only batch size 1 available for benchmarking")
        args["batch_size"] = 1

    # ===== Prepare Input Data =====
    input_data = get_chat_input_data(input_text, args)
    save_input_data_to_file(["; ".join(input_data)], args, model_precision, chat_index, iter_num, proc_id)

    # ===== Setup Generation Config and Additional Args =====
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    gen_config = genai_generation_config_setup(model, max_gen_tokens, args)
    gen_config.apply_chat_template = True
    additional_args = {"adapters": openvino_genai.AdapterConfig()} if (args["empty_lora"] and (gen_config.adapters is not None)) else {}

    mem_consumption.start()

    # ===== Chat Conversation =====
    chat_history = openvino_genai.ChatHistory()
    chat_iter_data_list = []
    for prompt_index, prompt in enumerate(input_data):
        chat_history.append({"role": "user", "content": prompt})

        # === Generation ===
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        start = time.perf_counter()
        decoded_results = model.generate(chat_history, gen_config, streamer, **additional_args)
        end = time.perf_counter()
        log.info("%s Text generation start: %s", prefix, datetime.datetime.now().isoformat())
        generation_time = end - start
        chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})

        # ===== Performance Data Collection and Print Results =====
        perf_metrics = decoded_results.perf_metrics
        per_token_time = ""
        num_tokens = perf_metrics.get_num_generated_tokens()
        if num_tokens > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        if num_tokens > 0:
            per_token_time = generation_time * 1000 / num_tokens
        else:
            log.warning("No generated tokens")
        first_token_time = (perf_metrics.get_ttft().mean)
        second_tokens_durations = (np.array(perf_metrics.raw_metrics.m_durations) / 1000).tolist()
        tm_list = (np.array([first_token_time] + second_tokens_durations) / 1000).tolist()
        inference_durations = (np.array(perf_metrics.raw_metrics.token_infer_durations) / 1000 / 1000).tolist()
        log.debug('latency of all tokens:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]

        tokenization_time = []
        tokenization_time.append(np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000)
        tokenization_time.append(np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000)

        result_md5_list = []
        generated_text = "; ".join(str(replica) for replica in chat_history.get_messages())
        result_md5_list.append(hashlib.new("md5", generated_text.encode(), usedforsecurity=False).hexdigest())
        update_md5_list(md5_list, iter_num, result_md5_list, prompt_index, chat_index=chat_index)
        print_generated_output(prompt_index, iter_num, result_md5_list, md5_list, generated_text,
                               enable_prompt_permutations=False, chat_prompts_num=len(input_data), chat_idx=chat_index)

        cache_usage = None
        if hasattr(model, 'get_metrics'):
            pipeline_metrics = model.get_metrics()
            if hasattr(pipeline_metrics, 'avg_cache_usage') and hasattr(pipeline_metrics, 'max_cache_usage'):
                cache_usage = {"avg_cache_usage": pipeline_metrics.avg_cache_usage, "max_cache_usage": pipeline_metrics.max_cache_usage}
        iter_data = gen_output_data.gen_iterate_data(
            iter_idx=iter_num,
            in_size=perf_metrics.get_num_input_tokens(),
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
            batch_size=args['batch_size'],
            prompt_idx=prompt_index,
            cb_metric=cache_usage,
            chat_idx=chat_index
        )

    memory_metrics = mem_consumption.iter_stop_and_collect_data(iter_num)
    # === ADD MEMORY METRICS for all chat iterartion ====
    for iter in chat_iter_data_list:
        mem_vlas = { key: val for key, val in gen_output_data.gen_iterate_data(**memory_metrics).items() if (val != '' and val != -1) }
        iter.update(**mem_vlas)

    update_chat_iteration_with_memory_info(chat_iter_data_list, memory_metrics)

    # === Save perf data ===
    iter_data_list.extend(chat_iter_data_list)

    generated_text = "; ".join(str(replica) for replica in chat_history.get_messages())
    if args["output_dir"] is not None:
        llm_bench_utils.output_file.output_gen_text(generated_text, args, model_precision, chat_index, iter_num, batchsize_idx=0, proc_id=proc_id, is_chat=True)


def run_text_generation_benchmark(model_path, framework, device, tokens_len, streaming, args, num_iters, mem_consumption):
    mem_consumption.update_marker("model")
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_gen_model(model_path, device, mem_consumption, **args)
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_text_list = get_text_prompt(args)
    if args['prompt_index'] is None:
        inputs_idx_list = [prompt_idx for prompt_idx, input_text in enumerate(input_text_list)]
        text_list = input_text_list
    else:
        inputs_idx_list = []
        text_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(input_text_list):
                text_list.append(input_text_list[i])
                inputs_idx_list.append(i)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
             f'input nums: {len(text_list)}, input idx: {inputs_idx_list}')

    # if num_iters == 0, just output warm-up data
    is_chat_mode = isinstance(input_text_list[0], list)
    if use_genai:
        if is_chat_mode:
            text_gen_fn = run_text_generation_genai_chat_mode
        else:
            if bench_hook is not None:
                text_gen_fn = run_text_generation_genai_with_stream
            else:
                text_gen_fn = run_text_generation_genai
    else:
        if is_chat_mode:
            text_gen_fn = run_text_generation_chat_optimum
        else:
            text_gen_fn = run_text_generation_optimum

    proc_id = os.getpid()
    iter_alias = "C" if is_chat_mode else "P"
    mem_consumption.activate_cooldown("after model compilation")
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, inputs_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                p_idx = inputs_idx_list[idx]
                # Set the logger prefix for current iteration and prompt index
                prefix = f"[warm-up][{iter_alias}{p_idx}]" if num == 0 else f"[{num}][{iter_alias}{p_idx}]"
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                if num == 0:
                    metrics_print.print_unicode(
                        f"{prefix} Input text: {input_text}",
                        f"{prefix} Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                            p_idx, bench_hook, tokens_len, streaming, model_precision, proc_id, mem_consumption, prefix)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                log.info(f"{prefix}[{iter_alias}{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = inputs_idx_list[idx]
            for num in range(num_iters + 1):
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                # Set the logger prefix for current iteration and prompt index
                prefix = f"[warm-up][{iter_alias}{p_idx}]" if num == 0 else f"[{num}][{iter_alias}{p_idx}]"
                if num == 0:
                    metrics_print.print_unicode(
                        f"{prefix} Input text: {input_text}",
                        f"{prefix} Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                    metrics_print.print_unicode(f'[warm-up][{iter_alias}{p_idx}] Input text: {input_text}', f'[warm-up][{iter_alias}{p_idx}] Unable print input text',
                                                max_output=metrics_print.MAX_INPUT_TXT_IN_LOG)
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                            inputs_idx_list[idx], bench_hook, model_precision, proc_id, mem_consumption, prefix)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                log.info(f"{prefix}[{iter_alias}{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    metrics_print.print_average(iter_data_list, inputs_idx_list, args['batch_size'], True, chat_mode=is_chat_mode)
    return iter_data_list, pretrain_time, iter_timestamp
