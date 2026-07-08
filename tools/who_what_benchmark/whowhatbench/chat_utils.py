# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

import numpy as np

from typing import TYPE_CHECKING
from .inputs_preprocessors.vlm_inputs_preprocessor import VLMInputsPreprocessor

if TYPE_CHECKING:
    from transformers import Cache


def find_common_prefix_length(
    new_tokens: list, tokenized_history: list, input_processor: VLMInputsPreprocessor = None
) -> int:
    kv_cache_len = min(len(new_tokens), len(tokenized_history))
    prefix_len = kv_cache_len
    for idx in range(kv_cache_len):
        if new_tokens[idx] != tokenized_history[idx]:
            prefix_len = idx
            if input_processor is not None and (
                input_processor.is_image_token(tokenized_history, prefix_len)
                or input_processor.is_image_token(new_tokens, prefix_len)
            ):
                prefix_len = 0
            break

    return prefix_len


def get_kv_cache_seq_len(model: Any, past_key_values: Union[tuple, "Cache", None], tokenized_chat_hist: list) -> int:
    past_key_values_len = 0
    if past_key_values is None:
        return past_key_values_len

    if "transformers" in str(type(model)):
        from transformers import Cache

        if isinstance(past_key_values, Cache):
            if hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() is not None:
                past_key_values_len = past_key_values.get_seq_length()
        elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            past_key_values_len = past_key_values[0][0].shape[-2]
    else:
        # kv_cache doesn't include last generated token, len(kv_cache) == len(output tokens) - 1
        past_key_values_len = len(tokenized_chat_hist) - 1

    return past_key_values_len


def trim_kv_cache(
    model, past_key_values: Union[tuple, "Cache", None], prefix_len: int, kv_axes_pos: int = 2
) -> Union[tuple, "Cache", None]:
    if past_key_values is None:
        return None

    if prefix_len == 0:
        if "transformers" in str(type(model)):
            return None
        else:
            model.request.reset_state()
            return [None]

    if "transformers" in str(type(model)):
        from transformers import Cache

        if isinstance(past_key_values, Cache):
            if hasattr(past_key_values, "crop"):
                past_key_values.crop(max_length=prefix_len)
        elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
            trimmed = tuple((k[..., :prefix_len, :], v[..., :prefix_len, :]) for k, v in past_key_values)
            past_key_values = trimmed
    else:
        model._past_length = prefix_len
        import openvino as ov

        states = model.request.query_state()
        for state in states:
            old_tensor = state.state
            # [BATCH_SIZE, num_kv_heads, seq_len, head_size]
            data = np.array(old_tensor.data)
            slices = [slice(None)] * data.ndim
            slices[kv_axes_pos] = slice(None, prefix_len)
            trimmed_tensor = data[tuple(slices)]

            new_tensor = ov.Tensor(trimmed_tensor)
            state.state = new_tensor

    return past_key_values


def get_kv_axes_pos(model: Any) -> int:
    # sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
    # therefore usually seq_length_axis = 2
    kv_pos = 2

    # "ReadValue" node is KV cache representation in stateful model
    kv_node_type_name = "ReadValue"

    for op in model.get_ops():
        # check input size, as in LoRA adapters case it could be 0
        if op.get_type_name() != kv_node_type_name or op.get_input_size() < 1:
            continue

        # Shape example: [-1,4,0,64]
        shape = op.get_input_partial_shape(0)
        if shape.rank.is_dynamic or shape.rank.get_length() != 4:
            # kv cache should have 4 dimensions
            continue

        for i in range(shape.rank.get_length()):
            # Find axis = 0. This would be sequence length axis.
            if shape[i] == 0:
                kv_pos = i
                break

    return kv_pos
