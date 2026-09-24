# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import logging as log

# CJK / kana characters: one word each (these languages don't separate words with spaces).
_CJK = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
# A word: letters/digits (Unicode-aware, no underscore, no CJK), optionally joined
# by an apostrophe or hyphen ("don't", "state-of-the-art" -> 1 word).
_WORD_RE = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+(?:['’\-][^\W_{_CJK}]+)*")


def count_words(text):
    """Number of words in *text*, as shown by the ``text:<N>w`` reprs.

    Punctuation is not a word, whether attached ("Hello,") or standalone
    (" - "), and every CJK / kana character counts as one word.
    """
    return len(_WORD_RE.findall(text)) if text else 0


def count_text_tokens(tokenizer, text):
    """Number of tokens the raw *text* encodes to, or ``None`` if unknown.

    Counts the text alone: no special tokens, no chat template, no media
    placeholders — so it is not comparable with ``input_size``, which is what
    the model actually processed. Accepts a HF tokenizer, a HF processor
    (unwrapped via its ``.tokenizer``) or an ``openvino_genai.Tokenizer``.
    Call it outside the timed region; any tokenizer failure yields ``None``
    so the repr falls back to the word count alone.
    """
    if tokenizer is None or not text:
        return None
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception as exc:
        log.debug(f"Cannot count input text tokens: {exc}")
        return None
    # openvino_genai returns TokenizedInputs, HF a plain list of ids.
    ids = getattr(ids, "input_ids", ids)
    return ids.shape[-1] if hasattr(ids, "shape") else len(ids)


def text_output_repr(text, tokenizer=None):
    """Size summary of a generated *text* output, e.g. ``"text:42w/57t"``.

    Mirrors the input-side text repr produced by ``BenchPrompt._repr``
    (``prompt_repr``) so ``output_repr`` and ``prompt_repr`` read the same way:
    text as a word count (``w`` suffix, see :func:`count_words`) plus, when a
    *tokenizer* is given, its token count (``/<M>t`` suffix, see
    :func:`count_text_tokens`); media as dimensions (the media-generating tasks
    pass their own ``output_repr`` such as ``"image:512x512"``). The token
    count is dropped if it cannot be determined. Returns ``""`` for empty
    output.

    Callers pass the single generated string they also report elsewhere (the
    first batch element), matching ``prompt_repr``, which likewise describes
    one item rather than the whole batch — so the token count is not
    comparable with ``output_size``, which is batch-total. Call it outside the
    timed region, as tokenizing the output takes time.
    """
    n = count_words(text)
    if not n:
        return ""
    text_tokens = count_text_tokens(tokenizer, text)
    return f"text:{n}w/{text_tokens}t" if text_tokens is not None else f"text:{n}w"


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
    chat_idx="",
    output_repr="",
    in_text_tokens="",
):
    iter_data = {}
    iter_data["iteration"] = iter_idx
    iter_data["input_size"] = in_size
    # input_text_tokens: token count of the raw prompt text alone (see
    # count_text_tokens). Not a report column: BenchPrompt renders it into
    # prompt_repr as "text:<N>w/<M>t".
    iter_data["input_text_tokens"] = in_text_tokens if in_text_tokens is not None else ""
    iter_data["infer_count"] = infer_count
    iter_data["output_size"] = out_size
    # output_repr: compact summary of the generated output, symmetric with
    # prompt_repr. Text tasks pass "text:<N>w/<M>t" words and tokens (via
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
