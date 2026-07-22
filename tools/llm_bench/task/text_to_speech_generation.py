# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import hashlib
import datetime
import logging as log
import numpy as np
import soundfile as sf
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
from llm_bench_utils.hook_forward import TTSHook
import openvino as ov
from llm_bench_utils.tts_utils import (
    extract_audio_array,
    get_tts_sample_rate,
    kokoro_preprocess_once,
    kokoro_generate_from_preprocessed,
    resolve_kokoro_speaker_embedding,
    resolve_omni_generation_settings,
)
import llm_bench_utils.metrics_print as metrics_print
from transformers import set_seed
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import get_text_prompt

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}


def run_text_to_speech_generation_optimum(
    input_text, num, model, processor, vocoder, args, iter_data_list, md5_list, prompt_index, tts_hook, model_precision, proc_id, mem_consumption
):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(
                in_text, args, model_precision, prompt_index, bs_index, proc_id
            )
    is_kokoro_model = args.get("is_kokoro_model", False)
    sample_rate = get_tts_sample_rate(args)
    tok_encode_time = None
    kokoro_preprocessed_inputs = []
    if is_kokoro_model:
        tok_encode_start = time.perf_counter()
        input_token_size = len(input_text.split())
        kokoro_preprocessed_inputs.append(kokoro_preprocess_once(model, input_text, args))
        tok_encode_end = time.perf_counter()
        tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    else:
        tok_encode_start = time.perf_counter()
        input_data = processor(text=input_text_list, return_tensors="pt", padding=True, truncation=True)
        input_data.pop("token_type_ids", None)
        input_tokens = input_data["input_ids"] if "input_ids" in input_data else input_data
        input_token_size = input_tokens[0].numel()
        tok_encode_end = time.perf_counter()
        tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        log.info(out_str)

    mem_consumption.start(num)
    start = time.perf_counter()
    speeches = []
    if is_kokoro_model:
        for preprocessed_input in kokoro_preprocessed_inputs:
            speeches.append(kokoro_generate_from_preprocessed(model, preprocessed_input, args))
        out_size = sum(speech.size for speech in speeches)
    else:
        if vocoder:
            result = model.generate(input_tokens, speaker_embeddings=args.get("speaker_embeddings"), vocoder=vocoder)
        else:
            result = model.generate(input_tokens, speaker_embeddings=args.get("speaker_embeddings"))
        out_size = result.numel()
    end = time.perf_counter()
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        if is_kokoro_model:
            speech = speeches[bs_idx]
        else:
            speech = result.numpy()[bs_idx] if len(result.size()) > 1 else result.numpy()
        audio_file_path = llm_bench_utils.output_file.output_gen_audio(
            speech, args, prompt_index, num, bs_idx, proc_id, ".wav", samplerate=sample_rate
        )
        data, _ = sf.read(audio_file_path)
        result_md5_list.append(hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list

    tokenization_kwargs = {}
    if tok_encode_time is not None:
        tokenization_kwargs["tokenization_time"] = [tok_encode_time]

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        out_size=out_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        **tokenization_kwargs,
        **memory_metrics,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        warm_up=(num == 0),
        **tokenization_kwargs,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index,
        tts=tts_hook
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
    if tts_hook is not None:
        tts_hook.clear_statistics()


def run_text_to_speech_generation_genai(
    input_text, num, model, processor, vocoder, args, iter_data_list, md5_list, prompt_index, tts_hook, model_precision, proc_id, mem_consumption
):
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)

    mem_consumption.start(num)
    is_kokoro_model = args.get("is_kokoro_model", False)
    sample_rate = get_tts_sample_rate(args)
    if is_kokoro_model:
        num_input_tokens = len(input_text.split())
    else:
        input_data = processor(text=input_text)
        num_input_tokens = len(input_data["input_ids"])

    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(num_input_tokens, args['batch_size'])
        log.info(out_str)

    speeches = []
    perf_metrics = None
    if is_kokoro_model and args.get("speaker_embeddings") is None:
        args["speaker_embeddings"] = resolve_kokoro_speaker_embedding(
            model_path=args.get("model_path"),
            speech_voice=args.get("speech_voice", ""),
            speaker_embeddings=args.get("speaker_embeddings"),
            strict=True,
        )

    additional_args = (
        {
            "speaker_embedding": ov.Tensor(
                args["speaker_embeddings"].detach().cpu().numpy()
                if hasattr(args["speaker_embeddings"], "detach")
                else np.asarray(args["speaker_embeddings"], dtype=np.float32)
            ),
        }
        if args.get("speaker_embeddings") is not None
        else {}
    )

    if is_kokoro_model:
        additional_args["language"] = args.get("speech_language", "")

    start = time.perf_counter()
    generation_result = model.generate(input_text_list, **additional_args)
    end = time.perf_counter()
    generation_time = end - start

    perf_metrics = generation_result.perf_metrics
    for bs_idx in range(args["batch_size"]):
        speeches.append(extract_audio_array(generation_result.speeches[bs_idx].data))
    out_size = perf_metrics.num_generated_samples
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        speech = speeches[bs_idx]
        audio_file_path = llm_bench_utils.output_file.output_gen_audio(
            speech, args, prompt_index, num, bs_idx, proc_id, ".wav", samplerate=sample_rate
        )
        data, _ = sf.read(audio_file_path)
        result_md5_list.append(hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest())

    md5_list[num][prompt_index] = result_md5_list

    tokenization_time = None
    tokenization_duration = perf_metrics.get_tokenization_duration().mean
    if tokenization_duration > 0:
        tokenization_time = [tokenization_duration]
    tokenization_kwargs = {"tokenization_time": tokenization_time} if tokenization_time is not None else {}

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=num_input_tokens * args['batch_size'],
        out_size=out_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        **tokenization_kwargs,
        **memory_metrics,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        tokenization_time=tokenization_time,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )

    log.debug(f"[{num}]Throughput: {perf_metrics.throughput.mean:.4f}")
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")


def _save_omni_speech(waveform, args, prompt_index, num, bs_idx, proc_id, sample_rate):
    audio_array = extract_audio_array(waveform)
    audio_file_path = llm_bench_utils.output_file.output_gen_audio(
        audio_array, args, prompt_index, num, bs_idx, proc_id, ".wav", samplerate=sample_rate
    )
    data, _ = sf.read(audio_file_path)
    return audio_array, hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest()


def _record_omni_iter(
    num,
    args,
    iter_data_list,
    md5_list,
    prompt_index,
    result_md5_list,
    in_size,
    out_size,
    generation_time,
    tokenization_kwargs,
    memory_metrics,
):
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=in_size,
        out_size=out_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        **tokenization_kwargs,
        **memory_metrics,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        warm_up=(num == 0),
        **tokenization_kwargs,
        batch_size=args["batch_size"],
        prompt_idx=prompt_index,
    )
    md5_list[num][prompt_index] = result_md5_list
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(
                f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                f"is different from md5 of the {num - 1} iteration {prev_md5}"
            )


def run_omni_text_to_speech_generation_optimum(
    input_text,
    num,
    model,
    processor,
    vocoder,
    args,
    iter_data_list,
    md5_list,
    prompt_index,
    tts_hook,
    model_precision,
    proc_id,
    mem_consumption,
):
    settings = resolve_omni_generation_settings(args)
    set_seed(settings["seed"])
    if args["output_dir"] is not None and num == 0:
        llm_bench_utils.output_file.output_input_text(input_text, args, model_precision, prompt_index, 0, proc_id)

    sample_rate = get_tts_sample_rate(args)

    tok_encode_start = time.perf_counter()
    input_data = model.preprocess_inputs(text=input_text, processor=processor)
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000

    input_tokens = input_data["input_ids"]
    input_token_size = input_tokens[0].numel()

    generate_kwargs = {
        "return_audio": True,
        "speaker": settings["speaker"],
        "talker_seed": settings["seed"],
    }
    if settings["max_new_tokens"] is not None:
        generate_kwargs["max_new_tokens"] = settings["max_new_tokens"]
        generate_kwargs["talker_max_new_tokens"] = settings["max_new_tokens"]
    if settings["num_beams"] and settings["num_beams"] > 1:
        generate_kwargs["num_beams"] = settings["num_beams"]

    mem_consumption.start(num)
    start = time.perf_counter()
    _, waveform = model.generate(**input_data, **generate_kwargs)
    end = time.perf_counter()
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    if waveform is None:
        raise RuntimeError("Qwen3-Omni text_to_speech: talker did not produce a waveform")

    result_md5_list = []
    _, md5 = _save_omni_speech(waveform, args, prompt_index, num, 0, proc_id, sample_rate)
    result_md5_list.append(md5)

    out_size = int(np.asarray(waveform).size)
    tokenization_kwargs = {"tokenization_time": [tok_encode_time]}
    _record_omni_iter(
        num,
        args,
        iter_data_list,
        md5_list,
        prompt_index,
        result_md5_list,
        in_size=input_token_size,
        out_size=out_size,
        generation_time=generation_time,
        tokenization_kwargs=tokenization_kwargs,
        memory_metrics=memory_metrics,
    )


def run_omni_text_to_speech_generation_genai(
    input_text,
    num,
    model,
    processor,
    vocoder,
    args,
    iter_data_list,
    md5_list,
    prompt_index,
    tts_hook,
    model_precision,
    proc_id,
    mem_consumption,
):
    import openvino_genai

    if args["output_dir"] is not None and num == 0:
        llm_bench_utils.output_file.output_input_text(input_text, args, model_precision, prompt_index, 0, proc_id)

    sample_rate = get_tts_sample_rate(args)
    settings = resolve_omni_generation_settings(args)

    text_config = openvino_genai.GenerationConfig(os.path.join(str(args["model_path"]), "generation_config.json"))
    if settings["max_new_tokens"] is not None:
        text_config.max_new_tokens = settings["max_new_tokens"]
    text_config.do_sample = False
    text_config.num_beams = settings["num_beams"]
    text_config.rng_seed = settings["seed"]

    talker_speech_config = openvino_genai.OmniTalkerSpeechConfig(args["model_path"])
    talker_speech_config.return_audio = True
    talker_speech_config.speaker = settings["speaker"]
    talker_speech_config.rng_seed = settings["seed"]
    if settings["max_new_tokens"] is not None:
        talker_speech_config.max_new_tokens = settings["max_new_tokens"]

    mem_consumption.start(num)
    start = time.perf_counter()
    generation_result = model.generate(
        input_text,
        text_config=text_config,
        talker_speech_config=talker_speech_config,
    )
    end = time.perf_counter()
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    waveforms = generation_result.speech_result.waveforms
    if not waveforms:
        raise RuntimeError("Qwen3-Omni text_to_speech: OmniPipeline returned no speech waveforms")

    result_md5_list = []
    _, md5 = _save_omni_speech(waveforms[0], args, prompt_index, num, 0, proc_id, sample_rate)
    result_md5_list.append(md5)

    out_size = generation_result.speech_result.perf_metrics.num_generated_samples

    perf_metrics = generation_result.perf_metrics
    tokenization_duration = perf_metrics.get_tokenization_duration().mean
    tokenization_kwargs = {"tokenization_time": [tokenization_duration]} if tokenization_duration > 0 else {}

    _record_omni_iter(
        num,
        args,
        iter_data_list,
        md5_list,
        prompt_index,
        result_md5_list,
        in_size=perf_metrics.get_num_input_tokens(),
        out_size=out_size,
        generation_time=generation_time,
        tokenization_kwargs=tokenization_kwargs,
        memory_metrics=memory_metrics,
    )

    log.debug(
        "[%s][P%s] talker generation time: %.2fms",
        "warm-up" if num == 0 else num,
        prompt_index,
        generation_result.speech_result.perf_metrics.generation_time_ms,
    )


def run_text_2_speech_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    mem_consumption.update_marker("model")
    model, processor, vocoder, pretrain_time, use_genai = FW_UTILS[framework].create_text_2_speech_model(model_path, device, mem_consumption, **args)
    args["model_path"] = model_path
    if (args.get("is_kokoro_model", False) or args.get("is_omni_model", False)) and args.get("batch_size", 1) != 1:
        log.warning("Only batch size 1 available for benchmarking with kokoro / qwen3-omni models")
        args["batch_size"] = 1
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

    tts_hook = None
    is_omni_model = args.get("is_omni_model", False)
    if framework == "ov" and not use_genai and not args.get("is_kokoro_model", False) and not is_omni_model:
        tts_hook = TTSHook()
        tts_hook.new_encoder(model)
        tts_hook.new_decoder(model)
        tts_hook.new_postnet(model)
        tts_hook.new_vocoder(model)

    if is_omni_model:
        gen_fn = run_omni_text_to_speech_generation_genai if use_genai else run_omni_text_to_speech_generation_optimum
    elif use_genai:
        gen_fn = run_text_to_speech_generation_genai
    else:
        gen_fn = run_text_to_speech_generation_optimum

    proc_id = os.getpid()
    mem_consumption.activate_cooldown("after model compilation")
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, prompt_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                p_idx = prompt_idx_list[idx]
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                if num == 0:
                    metrics_print.print_unicode(f'[warm-up][P{p_idx}] Input text: {input_text}', f'[warm-up][P{p_idx}] Unable print input text',
                                                max_output=metrics_print.MAX_INPUT_TXT_IN_LOG)
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                gen_fn(input_text, num, model, processor, vocoder, args, iter_data_list, md5_list,
                       p_idx, tts_hook, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                if num == 0:
                    metrics_print.print_unicode(f'[warm-up][P{p_idx}] Input text: {input_text}', f'[warm-up][P{p_idx}] Unable print input text',
                                                max_output=metrics_print.MAX_INPUT_TXT_IN_LOG)
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                gen_fn(input_text, num, model, processor, vocoder, args, iter_data_list, md5_list,
                       prompt_idx_list[idx], tts_hook, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    return iter_data_list, pretrain_time, iter_timestamp
