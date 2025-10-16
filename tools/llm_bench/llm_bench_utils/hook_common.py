# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa
import logging as log
import transformers
from packaging import version

TRANS_MIN_VERSION = '4.40.0'


def get_bench_hook(num_beams, ov_model, rag=False):
    min_version = version.parse(TRANS_MIN_VERSION)
    trans_version = version.parse(transformers.__version__)
    search_type = 'beam search' if num_beams > 1 else 'greedy search'
    if rag:
        import llm_bench_utils.hook_forward
        bench_hook = llm_bench_utils.hook_forward.RAGForwardHook()
        bench_hook.new_forward(ov_model)
        return bench_hook

    if trans_version >= min_version:
        import llm_bench_utils.hook_greedy_search
        import llm_bench_utils.hook_beam_search
        if num_beams > 1:
            bench_hook = llm_bench_utils.hook_beam_search.BeamSearchHook()
        else:
            bench_hook = llm_bench_utils.hook_greedy_search.GreedySearchHook()
        bench_hook.new_forward(ov_model)
        if hasattr(ov_model, "get_multimodal_embeddings"):
            bench_hook.new_get_multimodal_embeddings(ov_model)
    else:
        log.warning(f'The minimum version of transformers to get 1st and 2nd tokens latency of {search_type} is: {min_version}')
        bench_hook = None
    return bench_hook