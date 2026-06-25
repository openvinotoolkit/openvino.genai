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
from llm_bench_utils.tts_utils import KOKORO_SPEAKER_EMB_SHAPE, DEFAULT_KOKORO_VOICE
import openvino as ov
import llm_bench_utils.metrics_print as metrics_print
from transformers import set_seed
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import get_text_prompt

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}
KOKORO_SAMPLE_RATE = 24000
DEFAULT_TTS_SAMPLE_RATE = 16000


def _get_tts_sample_rate(args):
    return KOKORO_SAMPLE_RATE if args.get("is_kokoro_model", False) else DEFAULT_TTS_SAMPLE_RATE


def _extract_audio_array(output):
    try:
        import torch

        if isinstance(output, torch.Tensor):
            return output.detach().cpu().reshape(-1).numpy()
    except ImportError:
        pass

    if hasattr(output, "numpy"):
        return output.numpy().reshape(-1)

    return np.asarray(output, dtype=np.float32).reshape(-1)


def _kokoro_generate_once(model, input_text, args, use_genai):
    speaker_embeddings = args.get("speaker_embeddings")
    speech_language = args.get("speech_language", "")
    speech_voice = args.get("speech_voice", "")

    if use_genai:
        # Note: For GenAI Kokoro model, the speaker embedding is passed here, even if user specified --speech_voice.
        speaker_embedding_np = (
            speaker_embeddings.detach().cpu().numpy()
            if hasattr(speaker_embeddings, "detach")
            else np.asarray(speaker_embeddings)
        )
        generation_result = model.generate(
            input_text,
            speaker_embedding=ov.Tensor(speaker_embedding_np),
            language=speech_language,
        )
        if hasattr(generation_result, "speeches"):
            return _extract_audio_array(generation_result.speeches[0].data)
        return _extract_audio_array(generation_result)

    generation_result = model.generate(
        input_text,
        speaker_embeddings=speaker_embeddings,
        language=speech_language,
        voice=speech_voice,
    )
    return _extract_audio_array(generation_result)


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
    sample_rate = _get_tts_sample_rate(args)
    tok_encode_start = time.perf_counter()
    if is_kokoro_model:
        input_token_size = len(input_text.split())
    else:
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
        for _ in range(args["batch_size"]):
            speeches.append(_kokoro_generate_once(model, input_text, args, use_genai=False))
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
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        out_size=out_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        tokenization_time=[tok_encode_time],
        **memory_metrics,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        warm_up=(num == 0),
        tokenization_time=[tok_encode_time],
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
    sample_rate = _get_tts_sample_rate(args)
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

    start = time.perf_counter()
    speeches = []
    perf_metrics = None
    if is_kokoro_model:
        for _ in range(args["batch_size"]):
            speeches.append(_kokoro_generate_once(model, input_text, args, use_genai=True))
        out_size = sum(speech.size for speech in speeches)
        tokenization_time = [0]
    else:
        additional_args = (
            {
                "speaker_embedding": ov.Tensor(
                    args["speaker_embeddings"].detach().cpu().numpy()
                    if hasattr(args["speaker_embeddings"], "detach")
                    else np.asarray(args["speaker_embeddings"], dtype=np.float32)
                )
            }
            if args.get("speaker_embeddings") is not None
            else {}
        )
        generation_result = model.generate(input_text_list, **additional_args)
        perf_metrics = generation_result.perf_metrics
        tokenization_time = [perf_metrics.get_tokenization_duration().mean]
        out_size = perf_metrics.num_generated_samples
        for bs_idx in range(args["batch_size"]):
            speeches.append(generation_result.speeches[bs_idx].data[0])
    end = time.perf_counter()
    generation_time = end - start
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

    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=num_input_tokens * args['batch_size'],
        out_size=out_size,
        gen_time=generation_time,
        res_md5=result_md5_list,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time,
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
    if perf_metrics is not None:
        log.debug(f"[{num}]Throughput: {perf_metrics.throughput.mean:.4f}")
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")


def run_text_2_speech_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    mem_consumption.update_marker("model")
    model, processor, vocoder, pretrain_time, use_genai = FW_UTILS[framework].create_text_2_speech_model(model_path, device, mem_consumption, **args)
    # For Kokoro with GenAI, the pipeline requires speaker_embedding as an ov.Tensor.
    # Auto-resolve from the model's voices/ directory when not provided via CLI.
    if args.get("is_kokoro_model") and use_genai and args.get("speaker_embeddings") is None:
        from llm_bench_utils.model_utils import get_speaker_embeddings

        voice = args.get("speech_voice", "").strip() or DEFAULT_KOKORO_VOICE
        voice_file = model_path / "voices" / f"{voice}.bin"
        if not voice_file.exists():
            raise RuntimeError(
                f"Kokoro voice file not found: {voice_file}. "
                f"Pass --speaker_embeddings directly or ensure '{voice}' exists under {model_path / 'voices'}/."
            )
        log.info(f"Loading Kokoro speaker embedding from {voice_file}")
        args["speaker_embeddings"] = get_speaker_embeddings(str(voice_file), expected_shape=KOKORO_SPEAKER_EMB_SHAPE)
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
    if framework == "ov" and not use_genai and not args.get("is_kokoro_model", False):
        tts_hook = TTSHook()
        tts_hook.new_encoder(model)
        tts_hook.new_decoder(model)
        tts_hook.new_postnet(model)
        tts_hook.new_vocoder(model)

    if use_genai:
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
