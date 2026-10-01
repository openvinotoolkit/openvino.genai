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
from pathlib import Path
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
from llm_bench_utils.hook_forward import TTSHook
import openvino as ov
from llm_bench_utils.tts_utils import (
    extract_audio_array,
    get_tts_sample_rate,
    get_qwen3_tts_variant,
    kokoro_preprocess_once,
    kokoro_generate_from_preprocessed,
    normalize_kokoro_lang_code,
    resolve_kokoro_speaker_embedding,
    resolve_omni_generation_settings,
    load_qwen3_reference_audio,
)
import llm_bench_utils.metrics_print as metrics_print
from transformers import set_seed
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.prompt_utils import get_text_prompt

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}


def _merge_tts_prompt_overrides(input_text, args):
    if not isinstance(input_text, dict):
        return input_text, args

    if "prompt" not in input_text or input_text["prompt"] == "":
        raise RuntimeError("text_to_speech prompt entry must contain non-empty 'prompt'")

    merged_args = dict(args)
    for param in ["speech_language", "speech_voice", "speech_instruct", "speech_ref_audio", "speech_ref_text"]:
        if param in input_text:
            merged_args[param] = input_text[param]

    return input_text["prompt"], merged_args


def _get_qwen3_tts_generation_defaults():
    return {
        "do_sample": False,
        "subtalker_dosample": False,
        "non_streaming_mode": True,
        "repetition_penalty": 1.2,
    }


def _get_qwen3_tts_generation_settings(args):
    variant = args.get("qwen3_tts_variant")
    if variant is None:
        model_source = args.get("model_path")
        if model_source is not None:
            variant = get_qwen3_tts_variant(model_source)

    return {
        "variant": variant,
        "language": (args.get("speech_language") or "").strip() or "Auto",
        "voice": (args.get("speech_voice") or "").strip(),
        "instruct": (args.get("speech_instruct") or "").strip(),
        "ref_audio": (args.get("speech_ref_audio") or "").strip(),
        "ref_text": (args.get("speech_ref_text") or "").strip(),
    }


def _validate_qwen3_tts_generation_settings(settings):
    variant = settings["variant"]
    if variant not in {"base", "custom_voice", "voice_design"}:
        raise RuntimeError(f"Unsupported Qwen3-TTS variant: {variant}")

    if settings["ref_audio"] and not Path(settings["ref_audio"]).exists():
        raise RuntimeError(f"Incorrect speech reference audio path: {settings['ref_audio']}")

    if variant == "base" and not settings["ref_audio"]:
        raise RuntimeError("Qwen3-TTS Base requires --speech_ref_audio")
    if variant == "custom_voice" and not settings["voice"]:
        raise RuntimeError("Qwen3-TTS CustomVoice requires --speech_voice")
    if variant == "voice_design" and settings["voice"]:
        log.warning("Qwen3-TTS VoiceDesign ignores --speech_voice")


def _prepare_qwen3_tts_variant_context(settings):
    context = {
        "variant": settings["variant"],
        "language": settings["language"],
        "speaker": settings["voice"],
        "instruct": settings["instruct"],
        "ref_text": settings["ref_text"] if settings["ref_text"] else None,
    }

    if settings["variant"] == "base":
        ref_audio_waveform, ref_sample_rate = load_qwen3_reference_audio(settings["ref_audio"])
        context["ref_audio_waveform"] = ref_audio_waveform
        context["ref_sample_rate"] = ref_sample_rate
        context["x_vector_only_mode"] = context["ref_text"] is None

    return context


def run_qwen3_text_to_speech_generation_optimum(
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
    if args["output_dir"] is not None and num == 0:
        llm_bench_utils.output_file.output_input_text(input_text, args, model_precision, prompt_index, 0, proc_id)

    settings = _get_qwen3_tts_generation_settings(args)
    _validate_qwen3_tts_generation_settings(settings)
    variant_context = _prepare_qwen3_tts_variant_context(settings)
    generation_kwargs = _get_qwen3_tts_generation_defaults()
    if args.get("infer_count") is not None:
        generation_kwargs["max_new_tokens"] = int(args["infer_count"])

    preprocess_kwargs = {
        "text": [input_text],
        "language": [variant_context["language"]],
    }
    if variant_context["variant"] == "base":
        preprocess_kwargs["ref_audio"] = (variant_context["ref_audio_waveform"], variant_context["ref_sample_rate"])
        if variant_context["ref_text"]:
            preprocess_kwargs["ref_text"] = variant_context["ref_text"]
        else:
            preprocess_kwargs["x_vector_only_mode"] = True
    elif variant_context["variant"] == "custom_voice":
        preprocess_kwargs["speaker"] = variant_context["speaker"]
        if variant_context["instruct"]:
            preprocess_kwargs["instruct"] = variant_context["instruct"]
    elif variant_context["variant"] == "voice_design":
        if variant_context["instruct"]:
            preprocess_kwargs["instruct"] = variant_context["instruct"]

    tok_encode_start = time.perf_counter()
    inputs = model.preprocess_input(**preprocess_kwargs)
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000

    input_token_size = len(input_text.split())

    # x_vector_only_mode is consumed by preprocess_input for Optimum.
    generation_kwargs.pop("x_vector_only_mode", None)

    mem_consumption.start(num)
    start = time.perf_counter()
    speeches = model.generate(**inputs, **generation_kwargs)
    end = time.perf_counter()
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    if isinstance(speeches, list):
        speech = speeches[0]
    else:
        speech = speeches
    waveform = speech.numpy() if hasattr(speech, "numpy") else np.asarray(speech)
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    out_size = int(waveform.size)

    sample_rate = int(getattr(model, "sampling_rate", get_tts_sample_rate(args)))
    audio_file_path = llm_bench_utils.output_file.output_gen_audio(
        waveform, args, prompt_index, num, 0, proc_id, ".wav", samplerate=sample_rate
    )
    data, _ = sf.read(audio_file_path)
    result_md5_list = [hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest()]

    md5_list[num][prompt_index] = result_md5_list

    tokenization_kwargs = {"tokenization_time": [tok_encode_time]}
    _record_tts_iter(
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
        sample_rate=sample_rate,
    )


def run_qwen3_text_to_speech_generation_genai(
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
    if args["output_dir"] is not None and num == 0:
        llm_bench_utils.output_file.output_input_text(input_text, args, model_precision, prompt_index, 0, proc_id)

    settings = _get_qwen3_tts_generation_settings(args)
    _validate_qwen3_tts_generation_settings(settings)
    variant_context = _prepare_qwen3_tts_variant_context(settings)
    generation_properties = _get_qwen3_tts_generation_defaults()
    if args.get("infer_count") is not None:
        generation_properties["max_new_tokens"] = int(args["infer_count"])
    generation_properties["language"] = variant_context["language"]

    if variant_context["variant"] == "base":
        generation_properties["ref_audio"] = ov.Tensor(variant_context["ref_audio_waveform"])
        if variant_context["ref_text"]:
            generation_properties["ref_text"] = variant_context["ref_text"]
    elif variant_context["variant"] == "custom_voice":
        generation_properties["speaker"] = variant_context["speaker"]
        if variant_context["instruct"]:
            generation_properties["instruct"] = variant_context["instruct"]
    elif variant_context["variant"] == "voice_design":
        if variant_context["instruct"]:
            generation_properties["instruct"] = variant_context["instruct"]

    mem_consumption.start(num)
    start = time.perf_counter()
    try:
        result = model.generate(input_text, None, **generation_properties)
    except TypeError:
        result = model.generate(input_text, **generation_properties)
    end = time.perf_counter()
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    if not getattr(result, "speeches", None):
        raise RuntimeError("Qwen3-TTS generation produced no speech waveforms")

    waveform = extract_audio_array(result.speeches[0].data)
    sample_rate = int(getattr(result, "output_sample_rate", get_tts_sample_rate(args)))
    out_size = int(waveform.size)

    audio_file_path = llm_bench_utils.output_file.output_gen_audio(
        waveform, args, prompt_index, num, 0, proc_id, ".wav", samplerate=sample_rate
    )
    data, _ = sf.read(audio_file_path)
    result_md5_list = [hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest()]

    tokenization_kwargs = {}
    perf_metrics = getattr(result, "perf_metrics", None)
    if perf_metrics is not None:
        tokenization_duration = perf_metrics.get_tokenization_duration().mean
        if tokenization_duration > 0:
            tokenization_kwargs = {"tokenization_time": [tokenization_duration]}

    _record_tts_iter(
        num,
        args,
        iter_data_list,
        md5_list,
        prompt_index,
        result_md5_list,
        in_size=len(input_text.split()),
        out_size=out_size,
        generation_time=generation_time,
        tokenization_kwargs=tokenization_kwargs,
        memory_metrics=memory_metrics,
        sample_rate=sample_rate,
    )


def run_qwen3_text_to_speech_generation_pt(
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
    set_seed(args["seed"])
    if args["output_dir"] is not None and num == 0:
        llm_bench_utils.output_file.output_input_text(input_text, args, model_precision, prompt_index, 0, proc_id)

    settings = _get_qwen3_tts_generation_settings(args)
    _validate_qwen3_tts_generation_settings(settings)
    variant_context = _prepare_qwen3_tts_variant_context(settings)
    generation_kwargs = _get_qwen3_tts_generation_defaults()
    if args.get("infer_count") is not None:
        generation_kwargs["max_new_tokens"] = int(args["infer_count"])

    mem_consumption.start(num)
    start = time.perf_counter()
    if variant_context["variant"] == "base":
        if variant_context["x_vector_only_mode"]:
            generation_kwargs["x_vector_only_mode"] = True
        wavs, sample_rate = model.generate_voice_clone(
            text=input_text,
            language=variant_context["language"],
            instruct=variant_context["instruct"],
            ref_audio=(variant_context["ref_audio_waveform"], variant_context["ref_sample_rate"]),
            ref_text=variant_context["ref_text"],
            **generation_kwargs,
        )
    elif variant_context["variant"] == "custom_voice":
        wavs, sample_rate = model.generate_custom_voice(
            text=input_text,
            speaker=variant_context["speaker"],
            language=variant_context["language"],
            instruct=variant_context["instruct"],
            **generation_kwargs,
        )
    else:
        wavs, sample_rate = model.generate_voice_design(
            text=input_text,
            language=variant_context["language"],
            instruct=variant_context["instruct"],
            **generation_kwargs,
        )
    end = time.perf_counter()
    generation_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    waveform = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    out_size = int(waveform.size)

    audio_file_path = llm_bench_utils.output_file.output_gen_audio(
        waveform, args, prompt_index, num, 0, proc_id, ".wav", samplerate=sample_rate
    )
    data, _ = sf.read(audio_file_path)
    result_md5_list = [hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest()]

    _record_tts_iter(
        num,
        args,
        iter_data_list,
        md5_list,
        prompt_index,
        result_md5_list,
        in_size=len(input_text.split()),
        out_size=out_size,
        generation_time=generation_time,
        tokenization_kwargs={},
        memory_metrics=memory_metrics,
        sample_rate=sample_rate,
    )


def _build_tts_audio_metrics(out_size, sample_rate, generation_time):
    output_duration_s = (out_size / sample_rate) if sample_rate > 0 else 0
    rtf = (generation_time / output_duration_s) if output_duration_s > 0 else -1
    return {
        "samples": int(out_size),
        "duration_s": output_duration_s,
        "sample_rate": int(sample_rate),
        "rtf": rtf,
    }


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
    tts_audio_metrics = _build_tts_audio_metrics(out_size, sample_rate, generation_time)
    iter_data["tts_output_duration_s"] = tts_audio_metrics["duration_s"]
    iter_data["tts_sample_rate"] = tts_audio_metrics["sample_rate"]
    iter_data["tts_rtf"] = tts_audio_metrics["rtf"]
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        warm_up=(num == 0),
        **tokenization_kwargs,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index,
        tts=tts_hook,
        tts_audio=tts_audio_metrics,
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
        additional_args["language"] = normalize_kokoro_lang_code(args.get("speech_language", ""))

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
    tts_audio_metrics = _build_tts_audio_metrics(out_size, sample_rate, generation_time)
    iter_data["tts_output_duration_s"] = tts_audio_metrics["duration_s"]
    iter_data["tts_sample_rate"] = tts_audio_metrics["sample_rate"]
    iter_data["tts_rtf"] = tts_audio_metrics["rtf"]
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        warm_up=(num == 0),
        tokenization_time=tokenization_time,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index,
        tts_audio=tts_audio_metrics,
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


def _record_tts_iter(
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
    sample_rate,
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
    tts_audio_metrics = _build_tts_audio_metrics(out_size, sample_rate, generation_time)
    iter_data["tts_output_duration_s"] = tts_audio_metrics["duration_s"]
    iter_data["tts_sample_rate"] = tts_audio_metrics["sample_rate"]
    iter_data["tts_rtf"] = tts_audio_metrics["rtf"]
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        warm_up=(num == 0),
        **tokenization_kwargs,
        batch_size=args["batch_size"],
        prompt_idx=prompt_index,
        tts_audio=tts_audio_metrics,
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
    audio_array, md5 = _save_omni_speech(waveform, args, prompt_index, num, 0, proc_id, sample_rate)
    result_md5_list.append(md5)

    out_size = int(audio_array.size)
    tokenization_kwargs = {"tokenization_time": [tok_encode_time]}
    _record_tts_iter(
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
        sample_rate=sample_rate,
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

    _record_tts_iter(
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
        sample_rate=sample_rate,
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
    if (
        args.get("is_kokoro_model", False)
        or args.get("is_omni_model", False)
        or args.get("is_qwen3_tts_model", False)
    ) and args.get("batch_size", 1) != 1:
        log.warning("Only batch size 1 available for benchmarking with kokoro / qwen3-omni / qwen3-tts models")
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
    if (
        framework == "ov"
        and not use_genai
        and not args.get("is_kokoro_model", False)
        and not is_omni_model
        and not args.get("is_qwen3_tts_model", False)
    ):
        tts_hook = TTSHook()
        tts_hook.new_encoder(model)
        tts_hook.new_decoder(model)
        tts_hook.new_postnet(model)
        tts_hook.new_vocoder(model)

    if args.get("is_qwen3_tts_model", False):
        if framework == "pt":
            gen_fn = run_qwen3_text_to_speech_generation_pt
        else:
            gen_fn = run_qwen3_text_to_speech_generation_genai if use_genai else run_qwen3_text_to_speech_generation_optimum
    elif is_omni_model:
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
                    metrics_print.print_unicode(
                        f'[warm-up][P{p_idx}] Input text: {input_text}',
                        f'[warm-up][P{p_idx}] Unable print input text',
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                cur_prompt, cur_args = _merge_tts_prompt_overrides(input_text, args)
                gen_fn(
                    cur_prompt,
                    num,
                    model,
                    processor,
                    vocoder,
                    cur_args,
                    iter_data_list,
                    md5_list,
                    p_idx,
                    tts_hook,
                    model_precision,
                    proc_id,
                    mem_consumption,
                )
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                if num == 0:
                    metrics_print.print_unicode(
                        f'[warm-up][P{p_idx}] Input text: {input_text}',
                        f'[warm-up][P{p_idx}] Unable print input text',
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                cur_prompt, cur_args = _merge_tts_prompt_overrides(input_text, args)
                gen_fn(
                    cur_prompt,
                    num,
                    model,
                    processor,
                    vocoder,
                    cur_args,
                    iter_data_list,
                    md5_list,
                    prompt_idx_list[idx],
                    tts_hook,
                    model_precision,
                    proc_id,
                    mem_consumption,
                )
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    metrics_print.print_average_tts(iter_data_list, prompt_idx_list)

    return iter_data_list, pretrain_time, iter_timestamp
