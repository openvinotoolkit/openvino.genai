# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
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
import llm_bench_utils.output_csv
from transformers import set_seed
from llm_bench_utils.ov_utils import get_genai_chunk_streamer, OptimumChunkStreamer
import llm_bench_utils.output_json
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import get_text_prompt

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_OUTPUT_TOKEN_SIZE = 512


def run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                        prompt_index, bench_hook, tokens_len, streaming, model_precision, proc_id, mem_consumption):
    from optimum.intel.utils.import_utils import is_transformers_version
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    if args["apply_chat_template"]:
        input_text_hist = [{'role': 'user', 'content': input_text}]
        templated_input_text = tokenizer.apply_chat_template(input_text_hist, tokenize=False, add_generation_prompt=True)
        input_text_list = [templated_input_text] * args['batch_size']

    tok_encode_start = time.perf_counter()
    input_data = tokenizer(input_text_list, return_tensors='pt')
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    input_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
    input_token_size = input_tokens[0].numel()
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)

    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    additional_args = {}
    if is_transformers_version(">=", "4.51"):
        additional_args["use_model_defaults"] = False
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    # llama-3-8b-instruct's generation_config.json has 4096 max_length.
    # This is too small because test prompt may contain 4096 tokens which leaves no space for new tokens.
    # Override it to preserve max_new_tokens.
    max_length = 2**64 - 1
    start = time.perf_counter()
    if streaming:
        if args['infer_count'] is not None and args['end_token_stopping'] is False:
            model.generation_config.eos_token_id = None
            model.config.eos_token_id = None
            result = model.generate(
                **input_data,
                max_new_tokens=int(max_gen_tokens),
                max_length=max_length,
                num_beams=args['num_beams'],
                use_cache=True,
                eos_token_id=None,
                do_sample=False,
                streamer=OptimumChunkStreamer(tokenizer, tokens_len=tokens_len),
                **additional_args
            )
        else:
            result = model.generate(
                **input_data,
                max_new_tokens=int(max_gen_tokens),
                max_length=max_length,
                num_beams=args['num_beams'],
                use_cache=True,
                do_sample=False,
                streamer=OptimumChunkStreamer(tokenizer, tokens_len=tokens_len),
                **additional_args
            )
    else:
        if args['infer_count'] is not None and args['end_token_stopping'] is False:
            model.generation_config.eos_token_id = None
            model.config.eos_token_id = None
            result = model.generate(
                **input_data,
                max_new_tokens=int(max_gen_tokens),
                max_length=max_length,
                num_beams=args['num_beams'],
                use_cache=True,
                eos_token_id=None,
                do_sample=False,
                **additional_args
            )
        else:
            result = model.generate(
                **input_data,
                max_new_tokens=int(max_gen_tokens),
                max_length=max_length,
                num_beams=args['num_beams'],
                use_cache=True,
                do_sample=False,
                **additional_args
            )
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()

    generation_time = end - start
    tok_decode_start = time.perf_counter()
    generated_text = tokenizer.batch_decode(result)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        if 'sum' not in args['model_name'] and result[bs_idx][:input_token_size].equal(input_tokens[bs_idx]):
            generated_token_size = len(result[bs_idx]) - input_tokens[bs_idx].numel()
        else:
            generated_token_size = len(result[bs_idx])
        # Encoder-decoder models expect the `decoder_input_ids` to start with a special token
        # When counting the output length, subtract 1. The last token does not participate in inference.
        if model.config.is_encoder_decoder and result[bs_idx][0] == model.config.decoder_start_token_id:
            generated_token_size = generated_token_size - 1
        num_tokens += generated_token_size
        if generated_token_size > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
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
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
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
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()


def genai_generate(streaming, model, tokens_len, gen_config, empty_lora, input_data, batch_size):
    import openvino_genai
    import openvino as ov
    cb_pipeline = isinstance(model, openvino_genai.ContinuousBatchingPipeline)
    if cb_pipeline:
        input_data = [ov.Tensor([input]) for input in input_data.input_ids.data]
        gen_config = [gen_config] * batch_size

    start = time.perf_counter()
    if streaming:
        text_print_streamer = get_genai_chunk_streamer()(model.get_tokenizer(), tokens_len)

        def token_printer():
            # Getting next elements from iterable will be blocked until a new token is available.
            for word in text_print_streamer:
                print(word, end='', flush=True)
        printer_thread = threading.Thread(target=token_printer, daemon=True)
        printer_thread.start()
        if (empty_lora and (gen_config.adapters is not None)):
            generation_result = model.generate(
                input_data,
                gen_config,
                streamer=text_print_streamer,
                adapters=openvino_genai.AdapterConfig()
            )
        else:
            generation_result = model.generate(
                input_data,
                gen_config,
                streamer=text_print_streamer
            )
        printer_thread.join()
    else:
        if (empty_lora and (gen_config.adapters is not None)):
            generation_result = model.generate(input_data, gen_config, adapters=openvino_genai.AdapterConfig())
        else:
            generation_result = model.generate(input_data, gen_config)
    end = time.perf_counter()
    generated_tokens = []
    if cb_pipeline:
        for res in generation_result:
            generated_tokens.append(res.m_generation_ids[0])
        generated_tokens = np.array(generated_tokens)
    else:
        generated_tokens = np.array(generation_result.tokens)

    perf_metrics = generation_result[0].perf_metrics if cb_pipeline else generation_result.perf_metrics
    return generated_tokens, perf_metrics, end - start


def run_text_generation_genai(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_index,
                              streamer, tokens_len, streaming, model_precision, proc_id, mem_consumption):
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()

    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    tokenizer = model.get_tokenizer()
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

    tokenization_start = time.perf_counter()
    input_data = tokenizer.encode(input_text_list)
    tokenization_end = time.perf_counter()
    tokenization_time = [(tokenization_end - tokenization_start) * 1000]

    enable_prompt_permutations = not args.get("disable_prompt_permutation", False)
    if enable_prompt_permutations:
        log.warning(
            "Enabled input prompt permutations. It means that generated results may vary on different steps. "
            "If it is not expected, please specify --disable_prompt_permutation in your benchmarking command to disable this behavior"
        )
        from openvino_genai import TokenizedInputs, GenerationConfig
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
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(num_input_tokens, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)
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
    generated_tokens, perf_metrics, generation_time = genai_generate(streaming, model, tokens_len, gen_config,
                                                                     args["empty_lora"], input_data, args['batch_size'])
    if streaming:
        tokenization_time.append(np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000)
        generated_text = tokenizer.decode(generated_tokens)
    else:
        detokenization_start = time.perf_counter()
        generated_text = tokenizer.decode(generated_tokens)
        detokenization_end = time.perf_counter()
        tokenization_time.append((detokenization_end - detokenization_start) * 1000)

    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()
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
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
    first_token_time = (perf_metrics.get_ttft().mean)
    second_tokens_durations = (
        np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
        - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1])
    ).tolist()

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
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time
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
    if num > 0 and not enable_prompt_permutations:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)


def run_text_generation_genai_with_stream(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                                          prompt_index, streamer, tokens_len, streaming, model_precision, proc_id, mem_consumption):
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    pipe_tokenizer = model.get_tokenizer()
    if args['apply_chat_template']:
        input_text_hist = [{'role': 'user', 'content': input_text}]
        templated_input_text = pipe_tokenizer.apply_chat_template(input_text_hist, add_generation_prompt=True)
        input_text_list = [templated_input_text] * args['batch_size']
        if not args["disable_prompt_permutation"]:
            log.warning(
                "Enabled chat template applying and permutation of input prompt. "
                "It means that after applying the chat template prompt will be tokenized and mixed, so the structure of chat template will not be kept. "
                "If it is not expected, please specify --disable_prompt_permutation in your benchmarking command to disable this behavior"
            )
    tok_encode_start = time.perf_counter()
    input_data = pipe_tokenizer.encode(input_text_list)
    tok_encode_end = time.perf_counter()
    input_token_size = input_data.input_ids.shape[1]
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)
    max_rss_mem_consumption = ''
    max_sys_mem_consumption = ''
    max_rss_mem_increase = ''
    max_sys_mem_increase = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    streamer.reset()
    gen_config = model.get_generation_config()
    gen_config.rng_seed = args["seed"]
    gen_config.max_new_tokens = max_gen_tokens
    gen_config.max_length = 2**64 - 1
    gen_config.num_beams = args["num_beams"]
    gen_config.do_sample = False
    if gen_config.num_beams > 1:
        gen_config.frequency_penalty = 0
        gen_config.presence_penalty = 0
        gen_config.repetition_penalty = 1
    gen_config.ignore_eos = True
    if hasattr(gen_config, 'apply_chat_template'):
        gen_config.apply_chat_template = False
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
    if args.get('draft_model', ''):
        config_info = "Speculative decoding config: "
        if args.get("num_assistant_tokens", None):
            gen_config.num_assistant_tokens = int(args["num_assistant_tokens"])
            config_info += f'num_assistant_tokens {args["num_assistant_tokens"]}'
        if args.get("assistant_confidence_threshold", None):
            gen_config.assistant_confidence_threshold = float(args["assistant_confidence_threshold"])
            config_info += f'assistant_confidence_threshold {args["assistant_confidence_threshold"]}'
        log.info(config_info)
    if args.get('max_ngram_size') and args.get('num_assistant_tokens'):
        config_info = "Prompt Lookup decoding config: "
        gen_config.max_ngram_size = int(args['max_ngram_size'])
        gen_config.num_assistant_tokens = int(args['num_assistant_tokens'])
        config_info += f"max_ngram_size {gen_config.max_ngram_size}, num_assistant_tokens {gen_config.num_assistant_tokens}"
        log.info(config_info)
    start = time.perf_counter()
    generated_tokens = model.generate(input_data, gen_config, streamer=streamer).tokens
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.stop_and_collect_data(f"{'P' + str(num) if num > 0 else 'warm-up'}_{proc_id}")
        max_rss_mem_consumption, max_rss_mem_increase, max_sys_mem_consumption, max_sys_mem_increase = mem_consumption.get_data()
    generation_time = end - start
    tok_decode_start = time.perf_counter()
    generated_text = pipe_tokenizer.decode(generated_tokens)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
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
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
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
        max_rss_mem=max_rss_mem_consumption,
        max_rss_mem_increase=max_rss_mem_increase,
        max_sys_mem=max_sys_mem_consumption,
        max_sys_mem_increase=max_sys_mem_increase,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
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
    if num > 0 and not enable_prompt_permutations:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    streamer.reset()


def run_text_generation_benchmark(model_path, framework, device, tokens_len, streaming, args, num_iters, mem_consumption):
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_gen_model(model_path, device, mem_consumption, **args)
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_text_list = get_text_prompt(args)
    if args['prompt_index'] is None:
        prompt_idx_list = [prompt_idx for prompt_idx, input_text in enumerate(input_text_list)]
        text_list = input_text_list
    else:
        prompt_idx_list = []
        text_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(input_text_list):
                text_list.append(input_text_list[i])
                prompt_idx_list.append(i)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
             f'prompt nums: {len(text_list)}, prompt idx: {prompt_idx_list}')

    # if num_iters == 0, just output warm-up data
    if not use_genai:
        text_gen_fn = run_text_generation
    elif bench_hook is not None:
        text_gen_fn = run_text_generation_genai_with_stream
    else:
        text_gen_fn = run_text_generation_genai

    proc_id = os.getpid()
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, prompt_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                p_idx = prompt_idx_list[idx]
                if num == 0:
                    metrics_print.print_unicode(f'[warm-up][P{p_idx}] Input text: {input_text}', f'[warm-up][P{p_idx}] Unable print input text')
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                            p_idx, bench_hook, tokens_len, streaming, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                if num == 0:
                    metrics_print.print_unicode(f'[warm-up][P{p_idx}] Input text: {input_text}', f'[warm-up][P{p_idx}] Unable print input text')
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                            prompt_idx_list[idx], bench_hook, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], True)
    return iter_data_list, pretrain_time, iter_timestamp
