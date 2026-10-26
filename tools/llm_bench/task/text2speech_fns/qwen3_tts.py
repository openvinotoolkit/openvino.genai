# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
import hashlib
import logging as log

import numpy as np
import soundfile as sf
import openvino as ov
from transformers import set_seed

import llm_bench_utils.output_file
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.gen_output_data as gen_output_data
from llm_bench_utils.tts_utils import (
    extract_audio_array,
    get_tts_sample_rate,
    get_qwen3_tts_variant,
    load_qwen3_reference_audio,
    validate_reference_audio_source,
    DEFAULT_QWEN3_CUSTOM_VOICE_SPEAKER,
    DEFAULT_QWEN3_VOICE_DESIGN_INSTRUCT,
    DEFAULT_QWEN3_BASE_REF_AUDIO_URL,
    DEFAULT_QWEN3_BASE_REF_TEXT,
)


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

    settings = {
        "variant": variant,
        "language": (args.get("speech_language") or "").strip() or "Auto",
        "voice": (args.get("speech_voice") or "").strip(),
        "instruct": (args.get("speech_instruct") or "").strip(),
        "ref_audio": (args.get("speech_ref_audio") or "").strip(),
        "ref_text": (args.get("speech_ref_text") or "").strip(),
    }

    if settings["variant"] == "base" and not settings["ref_audio"]:
        settings["ref_audio"] = DEFAULT_QWEN3_BASE_REF_AUDIO_URL
        log.info("Qwen3-TTS Base: speech_ref_audio not provided, using default reference audio.")
        if not settings["ref_text"]:
            settings["ref_text"] = DEFAULT_QWEN3_BASE_REF_TEXT
            log.info("Qwen3-TTS Base: speech_ref_text not provided, using default reference text.")

    if settings["variant"] == "custom_voice" and not settings["voice"]:
        settings["voice"] = DEFAULT_QWEN3_CUSTOM_VOICE_SPEAKER
        log.info(
            "Qwen3-TTS CustomVoice: speech_voice not provided, using default speaker '%s'.",
            DEFAULT_QWEN3_CUSTOM_VOICE_SPEAKER,
        )

    if settings["variant"] == "voice_design" and not settings["instruct"]:
        settings["instruct"] = DEFAULT_QWEN3_VOICE_DESIGN_INSTRUCT
        log.info("Qwen3-TTS VoiceDesign: speech_instruct not provided, using default instruction.")
    return settings


def _validate_qwen3_tts_generation_settings(settings):
    variant = settings["variant"]
    if variant not in {"base", "custom_voice", "voice_design"}:
        raise RuntimeError(f"Unsupported Qwen3-TTS variant: {variant}")

    ref_audio = settings["ref_audio"]
    if ref_audio:
        validate_reference_audio_source(ref_audio)

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


def _build_tts_audio_metrics(out_size, sample_rate, generation_time):
    output_duration_s = (out_size / sample_rate) if sample_rate > 0 else 0
    rtf = (generation_time / output_duration_s) if output_duration_s > 0 else -1
    return {
        "samples": int(out_size),
        "duration_s": output_duration_s,
        "sample_rate": int(sample_rate),
        "rtf": rtf,
    }


def _record_qwen3_tts_iter(
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

    mem_consumption.start(num)
    start = time.perf_counter()

    inputs = model.preprocess_input(**preprocess_kwargs)

    # x_vector_only_mode is consumed by preprocess_input for Optimum.
    generation_kwargs.pop("x_vector_only_mode", None)

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

    _record_qwen3_tts_iter(
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

    _record_qwen3_tts_iter(
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

    _record_qwen3_tts_iter(
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
