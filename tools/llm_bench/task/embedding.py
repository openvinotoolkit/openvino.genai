# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import datetime
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.output_csv
import llm_bench_utils.output_json
import llm_bench_utils.output_file
from llm_bench_utils.prompt_utils import get_text_embed_prompt, extract_prompt_data
import llm_bench_utils.gen_output_data as gen_output_data

FW_UTILS = {"pt": llm_bench_utils.pt_utils, "ov": llm_bench_utils.ov_utils}

MULTIMODAL_DEFAULT_EMBEDDING_PROMPT = "Represent the user's input."


def _is_multimodal(args):
    return args.get("model_type") == "qwen3-vl"


def _resolve_embedding_prompt(args, is_multimodal):
    prompt = args.get("emb_prompt")
    if prompt is None and is_multimodal:
        return MULTIMODAL_DEFAULT_EMBEDDING_PROMPT
    return prompt


def _summarize_entry(entry):
    parts = [entry.get("prompt", "")]
    if entry.get("media") is not None:
        parts.append(f"[image={entry['media']}]")
    if entry.get("video") is not None:
        parts.append(f"[video={entry['video']}]")
    return " ".join(str(p) for p in parts if p)


def run_text_embeddings_optimum(
    entry, num, model, tokenizer, args, iter_data_list, prompt_index, bench_hook, proc_id, mem_consumption
):
    is_multimodal = _is_multimodal(args)
    embedding_prompt = _resolve_embedding_prompt(args, is_multimodal)
    batch_size = args["batch_size"]

    tok_encode_start = time.perf_counter()
    if is_multimodal:
        prompts, images, videos = extract_prompt_data([entry], args.get("video_frames"), False)
        prompt_text = prompts[0] if prompts else ""
        content = [{"type": "text", "text": prompt_text}] if prompt_text else []
        for image in images:
            content.append({"type": "image", "image": image})
        for video in videos:
            content.append({"type": "video", "video": video})
        conversation = [
            {"role": "system", "content": embedding_prompt},
            {"role": "user", "content": content},
        ]
        messages = [conversation] * batch_size
        batch_images = images * batch_size
        batch_videos = videos * batch_size

        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "padding_side": args.get("emb_padding_side") or "right",
        }
        if args.get("emb_pad_to_max_length") is True:
            tokenizer_kwargs["padding"] = "max_length"
        if args.get("emb_max_length") is not None:
            tokenizer_kwargs["max_length"] = args["emb_max_length"]
        input_kwargs = {}
        if batch_images:
            input_kwargs["images"] = batch_images
        if batch_videos:
            input_kwargs["videos"] = batch_videos

        templated_text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False, add_generation_prompt=False
        )
        input_data = tokenizer(text=templated_text_input, return_tensors="pt", **input_kwargs, **tokenizer_kwargs)
    else:
        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "padding_side": args.get("emb_padding_side") or "right",
        }
        if args.get("emb_pad_to_max_length") is True:
            tokenizer_kwargs.update({"padding": "max_length"})
        if args.get("emb_max_length") is not None:
            tokenizer_kwargs.update({"max_length": args["emb_max_length"]})
        prompt_text = entry.get("prompt") if isinstance(entry, dict) else entry
        input_text_list = [prompt_text] * batch_size
        input_data = tokenizer(input_text_list, return_tensors="pt", **tokenizer_kwargs)
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    input_tokens = input_data["input_ids"] if "input_ids" in input_data else input_data
    input_token_size = input_tokens[0].numel()
    if batch_size > 1:
        out_str = "[warm-up]" if num == 0 else "[{}]".format(num)
        out_str += " Batch_size={}, ".format(batch_size)
        out_str += "all input token size after padding: {} * {}, ".format(input_token_size, batch_size)
        if args["infer_count"] is not None:
            out_str += "all max_output_token_size: {} * {}".format(args["infer_count"], batch_size)
        log.info(out_str)

    mem_consumption.start(num)
    start = time.perf_counter()
    model(**input_data)
    end = time.perf_counter()
    embed_time = end - start
    embed_time_full = end - tok_encode_start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    tm_list = []
    tm_infer_list = []
    if bench_hook is not None:
        tm_list = bench_hook.get_time_list()
        log.debug("latency of all texts:")
        [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_list)]
        tm_infer_list = bench_hook.get_time_infer_list()
        log.debug("latency of all infers:")
        [log.debug("[{}]{:.4f}".format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
    iter_data = gen_output_data.embed_iterate_data(
        iter_idx=num,
        in_size=input_token_size * batch_size,
        infer_count=len(tm_infer_list),
        total_time=embed_time_full,
        latency=embed_time,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time,),
        **memory_metrics,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        tokenization_time=(tok_encode_time,),
        batch_size=batch_size,
        prompt_idx=prompt_index,
        latency_unit="prompt",
        text_emb=True,
    )
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()


def run_text_embeddings_genai(
    entry, num, model, tokenizer, args, iter_data_list, prompt_index, bench_hook, proc_id, mem_consumption
):
    is_multimodal = _is_multimodal(args)
    embedding_prompt = _resolve_embedding_prompt(args, is_multimodal)
    batch_size = args["batch_size"]

    prompts, images, videos = extract_prompt_data([entry], args.get("video_frames"), True)
    prompts = prompts * batch_size

    if is_multimodal:
        # EmbeddingPipeline applies the chat template, visual placeholders, and media
        # preprocessing internally, so an external tokenize step would not describe the
        # measured inference. Report 0 for tokenization time and prompt tokens here;
        # embed_time below covers the pipeline-internal path end-to-end.
        tok_encode_time = 0.0
        input_token_size = 0
    else:
        tokenizer_kwargs = {"padding": True, "truncation": True}
        if args.get("emb_pad_to_max_length") is True:
            tokenizer_kwargs["padding"] = "max_length"
        if args.get("emb_max_length") is not None:
            tokenizer_kwargs["max_length"] = args["emb_max_length"]
        tok_encode_start = time.perf_counter()
        input_data = tokenizer(prompts, return_tensors="pt", **tokenizer_kwargs)
        tok_encode_end = time.perf_counter()
        tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
        input_tokens = input_data["input_ids"] if "input_ids" in input_data else input_data
        input_token_size = input_tokens[0].numel()
    if batch_size > 1:
        out_str = "[warm-up]" if num == 0 else "[{}]".format(num)
        out_str += " Batch_size={}, ".format(batch_size)
        out_str += "all input token size after padding: {} * {}, ".format(input_token_size, batch_size)
        if args["infer_count"] is not None:
            out_str += "all max_output_token_size: {} * {}".format(args["infer_count"], batch_size)
        log.info(out_str)

    mem_consumption.start(num)
    start = time.perf_counter()
    if hasattr(model, "embed_documents"):
        # Older openvino_genai without EmbeddingPipeline: text-only TextEmbeddingPipeline.
        model.embed_documents(prompts)
    else:
        media_kwargs = {}
        if embedding_prompt is not None:
            media_kwargs["embedding_prompt"] = embedding_prompt
        if images:
            media_kwargs["images"] = images
        if videos:
            media_kwargs["videos"] = videos
        model.embed(prompts, **media_kwargs)
    end = time.perf_counter()
    embed_time = end - start
    memory_metrics = mem_consumption.iter_stop_and_collect_data(num)

    tm_list = []
    tm_infer_list = []
    tm_list.append(embed_time)
    tm_infer_list.append(embed_time)
    iter_data = gen_output_data.embed_iterate_data(
        iter_idx=num,
        in_size=input_token_size * batch_size,
        infer_count=len(tm_list),
        total_time=embed_time,
        latency=embed_time,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time,),
        **memory_metrics,
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        tokenization_time=(tok_encode_time,),
        batch_size=batch_size,
        prompt_idx=prompt_index,
        latency_unit="prompt",
        text_emb=True,
    )


def run_text_embddings_benchmark(model_path, framework, device, args, num_iters, mem_consumption):
    input_entries = get_text_embed_prompt(args)

    if args["prompt_index"] is None:
        prompt_idx_list = [prompt_idx for prompt_idx, _ in enumerate(input_entries)]
        entries_list = input_entries
    else:
        prompt_idx_list = []
        entries_list = []
        for i in args["prompt_index"]:
            if 0 <= i < len(input_entries):
                entries_list.append(input_entries[i])
                prompt_idx_list.append(i)

    if len(entries_list) == 0:
        raise RuntimeError("==Failure prompts is empty ==")

    if not _is_multimodal(args):
        for entry in entries_list:
            if entry.get("media") is not None or entry.get("video") is not None:
                raise RuntimeError(
                    "`--media`/`--images`/`--video` (and JSONL media/video fields) are only supported for "
                    f"multimodal embedding models (Qwen3-VL-Embedding). Model type '{args.get('model_type')}' "
                    "does not accept media inputs."
                )

    mem_consumption.update_marker("model")
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_embeddings_model(
        model_path, device, mem_consumption, **args
    )
    iter_data_list = []

    if not use_genai:
        text_emb_fn = run_text_embeddings_optimum
    else:
        text_emb_fn = run_text_embeddings_genai

    proc_id = os.getpid()
    mem_consumption.activate_cooldown("after model compilation")
    text_descriptions = [_summarize_entry(e) for e in entries_list]
    iter_timestamp = model_utils.init_timestamp(num_iters, text_descriptions, prompt_idx_list)
    if args["subsequent"] is False:
        for num in range(num_iters + 1):
            for idx, input_entry in enumerate(entries_list):
                p_idx = prompt_idx_list[idx]
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                if num == 0:
                    metrics_print.print_unicode(
                        f"[warm-up][P{p_idx}] Input text: {text_descriptions[idx]}",
                        f"[warm-up][P{p_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][p_idx]["start"] = datetime.datetime.now().isoformat()
                text_emb_fn(
                    input_entry,
                    num,
                    model,
                    tokenizer,
                    args,
                    iter_data_list,
                    p_idx,
                    bench_hook,
                    proc_id,
                    mem_consumption,
                )
                iter_timestamp[num][p_idx]["end"] = datetime.datetime.now().isoformat()
                prefix = "[warm-up]" if num == 0 else "[{}]".format(num)
                log.info(
                    f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}"
                )
    else:
        for idx, input_entry in enumerate(entries_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                mem_consumption.update_marker(f"step-{num}-{p_idx}")
                if num == 0:
                    metrics_print.print_unicode(
                        f"[warm-up][P{p_idx}] Input text: {text_descriptions[idx]}",
                        f"[warm-up][P{p_idx}] Unable print input text",
                        max_output=metrics_print.MAX_INPUT_TXT_IN_LOG,
                    )
                iter_timestamp[num][p_idx]["start"] = datetime.datetime.now().isoformat()
                text_emb_fn(
                    input_entry,
                    num,
                    model,
                    tokenizer,
                    args,
                    iter_data_list,
                    prompt_idx_list[idx],
                    bench_hook,
                    proc_id,
                    mem_consumption,
                )
                iter_timestamp[num][p_idx]["end"] = datetime.datetime.now().isoformat()
                prefix = "[warm-up]" if num == 0 else "[{}]".format(num)
                log.info(
                    f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}"
                )

    metrics_print.print_average(iter_data_list, prompt_idx_list, args["batch_size"], False, True, latency_unit="prompt")
    return iter_data_list, pretrain_time, iter_timestamp
