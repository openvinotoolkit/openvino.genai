# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def text_output_repr(text):
    """Word-count summary of a generated *text* output, e.g. ``"text:42w"``.

    Mirrors the input-side text repr produced by ``BenchPrompt.__repr__``
    (``prompt_repr``) so ``output_repr`` and ``prompt_repr`` read the same way:
    text as a whitespace word count (``w`` suffix), media as dimensions (the
    media-generating tasks pass their own ``output_repr`` such as
    ``"image:512x512"``). Returns ``""`` for empty output.
    """
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text)
    n = len(str(text).split()) if text else 0
    return f"text:{n}w" if n else ""


def gen_iterate_data(
    iter_idx="",
    in_size="",
    infer_count="",
    out_size="",
    gen_time="",
    latency="",
    res_md5="",
    max_rss_mem="",
    max_rss_mem_increase="",
    max_rss_mem_share="",
    max_sys_mem="",
    max_sys_mem_increase="",
    max_sys_mem_share="",
    prompt_idx="",
    tokenization_time=[],
    mm_embeddings_preparation_time="",
    output_repr="",
    chat_idx="",
):
    iter_data = {}
    iter_data["iteration"] = iter_idx
    iter_data["input_size"] = in_size
    iter_data["infer_count"] = infer_count
    iter_data["output_size"] = out_size
    # output_repr: compact summary of the generated output, symmetric with
    # prompt_repr. Text tasks pass an explicit "text:<N>w" word count (via
    # text_output_repr); media tasks pass their own dimensions string
    # (e.g. "image:512x512"). Empty when neither applies.
    iter_data["output_repr"] = output_repr or ""
    iter_data["generation_time"] = gen_time
    iter_data["latency"] = latency
    iter_data["result_md5"] = res_md5
    iter_data["first_token_latency"] = -1
    iter_data["other_tokens_avg_latency"] = -1
    iter_data["first_token_infer_latency"] = -1
    iter_data["other_tokens_infer_avg_latency"] = -1
    iter_data["max_rss_mem_consumption"] = max_rss_mem
    iter_data["max_rss_mem_increase"] = max_rss_mem_increase
    iter_data["max_rss_mem_share"] = max_rss_mem_share
    iter_data["max_sys_mem_consumption"] = max_sys_mem
    iter_data["max_sys_mem_increase"] = max_sys_mem_increase
    iter_data["max_sys_mem_share"] = max_sys_mem_share
    iter_data["prompt_idx"] = prompt_idx
    iter_data["tokenization_time"] = tokenization_time[0] if len(tokenization_time) > 0 else ""
    iter_data["detokenization_time"] = tokenization_time[1] if len(tokenization_time) > 1 else ""
    iter_data["mm_embeddings_preparation_time"] = mm_embeddings_preparation_time
    iter_data["chat_idx"] = chat_idx
    return iter_data


def embed_iterate_data(
    iter_idx="",
    in_size="",
    infer_count="",
    total_time="",
    latency="",
    available_mem="",
    max_rss_mem="",
    max_rss_mem_increase="",
    max_rss_mem_share="",
    max_sys_mem="",
    max_sys_mem_increase="",
    max_sys_mem_share="",
    prompt_idx="",
    tokenization_time=[],
):
    iter_data = {}
    iter_data["iteration"] = iter_idx
    iter_data["input_size"] = in_size
    iter_data["infer_count"] = infer_count
    iter_data["generation_time"] = total_time
    iter_data["latency"] = latency
    iter_data["first_token_latency"] = -1
    iter_data["other_tokens_avg_latency"] = -1
    iter_data["first_token_infer_latency"] = -1
    iter_data["other_tokens_infer_avg_latency"] = -1
    iter_data["available_mem"] = available_mem
    iter_data["max_rss_mem_consumption"] = max_rss_mem
    iter_data["max_rss_mem_increase"] = max_rss_mem_increase
    iter_data["max_rss_mem_share"] = max_rss_mem_share
    iter_data["max_sys_mem_consumption"] = max_sys_mem
    iter_data["max_sys_mem_increase"] = max_sys_mem_increase
    iter_data["max_sys_mem_share"] = max_sys_mem_share
    iter_data["prompt_idx"] = prompt_idx
    iter_data["tokenization_time"] = tokenization_time[0] if len(tokenization_time) > 0 else ""
    iter_data["detokenization_time"] = ""
    iter_data["result_md5"] = ""
    iter_data["output_size"] = ""
    iter_data["output_repr"] = ""  # embeddings have no generated output
    return iter_data
