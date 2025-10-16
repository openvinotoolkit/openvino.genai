# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa
import time
import types
from packaging import version
import transformers


tm_list = []
tm_infer_list = []
tm_mm_embeddings = []

if version.parse(transformers.__version__) >= version.parse("4.55.0"):
    import llm_bench_utils.llm_hook_beam_search.hook_beam_search_v55 as hook_beam_search_v55
    new_beam_search = hook_beam_search_v55.new_beam_search_v55
elif version.parse(transformers.__version__) >= version.parse("4.52.0"):
    import llm_bench_utils.llm_hook_beam_search.hook_beam_search_v52 as hook_beam_search_v52
    new_beam_search = hook_beam_search_v52.new_beam_search_v52
elif version.parse(transformers.__version__) >= version.parse("4.51.0"):
    import llm_bench_utils.llm_hook_beam_search.hook_beam_search_v51 as hook_beam_search_v51
    new_beam_search = hook_beam_search_v51.new_beam_search_v51
else:
    import llm_bench_utils.llm_hook_beam_search.hook_beam_search_v40 as hook_beam_search_v40
    new_beam_search = hook_beam_search_v40.new_beam_search_v40

def new_get_multimodal_embeddings(
        self, input_ids, pixel_values=None, attention_mask=None, position_ids=None, **kwargs
    ):

    start = time.perf_counter()
    result = self._orig_get_multimodal_embeddings(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
    end = time.perf_counter()
    tm_mm_embeddings.append(end - start)
    return result

class BeamSearchHook:
    def __init__(self):
        """Clear the time list."""
        global tm_list
        tm_list.clear()
        global tm_infer_list
        tm_infer_list.clear()

    def clear_time_list(self):
        """Clear the time list."""
        global tm_list
        tm_list.clear()

    def get_time_list(self):
        """Return the time list."""
        return tm_list

    def clear_time_infer_list(self):
        """Clear the infer time list."""
        global tm_infer_list
        tm_infer_list.clear()

    def get_time_infer_list(self):
        """Return the infer time list."""
        global tm_infer_list
        return tm_infer_list

    def get_mm_embeddings_time_list(self):
        global tm_mm_embeddings
        return tm_mm_embeddings

    def clear_mm_embeddins_time_list(self):
        """Clear the infer time list."""
        global tm_mm_embeddings
        tm_mm_embeddings.clear()

    def new_forward(self, model):
        """Define a new beam search function."""
        model._beam_search = new_beam_search.__get__(model, model.__class__)

    def new_get_multimodal_embeddings(self, model):
        model._orig_get_multimodal_embeddings = model.get_multimodal_embeddings
        model.get_multimodal_embeddings = types.MethodType(new_get_multimodal_embeddings, model)