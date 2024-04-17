# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa
import torch
from typing import Union, List, Dict
from transformers.utils import ModelOutput

TRANS_MIN_VERSION = '4.36.0'
TRANS_SENCOND_VERSION = '4.39.0'


# Copied from https://github.com/huggingface/transformers/blob/v4.39-release/src/transformers/generation/utils.py#L4783
def _split(data, full_batch_size: int, split_size: int = None):
    """
    Takes care of three cases:
    1. data is a tensor: e.g. last_hidden_state, pooler_output etc. split them on the batch_size dim
    2. data is a tuple: e.g. hidden_states, attentions etc. Keep the tuple as it is and split each tensor in it and
       return a list of tuples
    3. data is a tuple of tuples, e.g. past_key_values. Keep the tuple as it is and split each tuple in it and
       return a list of tuples of tuples
    (see documentation of ModelOutput)
    """
    if data is None:
        return [None] * (full_batch_size // split_size)
    if isinstance(data, torch.Tensor):
        return [data[i : i + split_size] for i in range(0, full_batch_size, split_size)]
    elif isinstance(data, tuple):
        # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
        if isinstance(data[0], tuple):
            return [
                tuple(tuple(tensor[i : i + split_size] for tensor in inner_tuple) for inner_tuple in data)
                for i in range(0, full_batch_size, split_size)
            ]

        else:
            return [
                tuple(sub_tensor[i : i + split_size] for sub_tensor in data)
                for i in range(0, full_batch_size, split_size)
            ]
    else:
        raise ValueError(f"Unexpected attribute type: {type(data)}")


# Copied from https://github.com/huggingface/transformers/blob/v4.39-release/src/transformers/generation/utils.py#L4814
def _split_model_inputs(
    model_input: Union[ModelOutput, Dict], split_size: int, full_batch_size: int
) -> List[Union[ModelOutput, Dict]]:
    """
    Split a ModelOutput object (or its subclasses) or Dict into a list of same-class objects based on a specified split
    size. The input object is dict when it was prepared for forward pass and ModelOutput when it was returned from
    previous forward pass.
    """
    # Edge case: if model_input is None, return a list of Nones
    # this happens with Whisper where encoder_outputs is None
    if model_input is None:
        return [model_input] * (full_batch_size // split_size)
    # Infer the class from the object
    model_output_cls = type(model_input)
    if (full_batch_size % split_size) != 0:
        raise ValueError("`full_batch_size` must be divisible by `split_size`")

    if split_size > full_batch_size:
        raise ValueError("`split_size` must be smaller or equal to `full_batch_size`")

    # Helper function to split tensors or tuples of tensors

    # Find all the dataclass fields (e.g., last_hidden_state, pooler_output etc.) and split them
    keys = (
        model_input.__dataclass_fields__.keys() if hasattr(model_input, "__dataclass_fields__") else model_input.keys()
    )
    # We only keep keys that are in the model_input
    keys = [k for k in keys if k in model_input]
    # Here we can have four types of values: tensors, tuples of tensors and booleans, and encoder_outputs which is a
    # ModelOutput object.
    # bool should not be split but replicated for each split
    bool_keys = [k for k in keys if isinstance(model_input[k], bool) or k == "cache_position"]
    keys_to_ignore = ["cache_position", "encoder_outputs"]
    non_bool_keys = [k for k in keys if not isinstance(model_input[k], bool) and k not in keys_to_ignore]

    # we split the tensors and tuples of tensors
    data_split_list = [
        {k: _split(model_input[k], full_batch_size, split_size)[i] for k in non_bool_keys}
        for i in range(full_batch_size // split_size)
    ]
    # bool values are the same and replicated for each split
    bool_data = {k: model_input[k] for k in bool_keys}
    # encoder_outputs is a ModelOutput object and should be split by its own
    if "encoder_outputs" in model_input:
        encoder_outputs_split = _split_model_inputs(model_input["encoder_outputs"], split_size, full_batch_size)
        data_split_list = [
            {**data_split, "encoder_outputs": encoder_outputs_split[i]} for i, data_split in enumerate(data_split_list)
        ]

    # Convert each dictionary in the list to an object of the inferred class
    split_model_inputs: List[Union[ModelOutput, Dict]] = [
        model_output_cls(**data_split, **bool_data) for data_split in data_split_list
    ]

    return split_model_inputs


# Copied from https://github.com/huggingface/transformers/blob/v4.39-release/src/transformers/generation/utils.py#L4871
def stack_model_outputs(model_outputs: List[ModelOutput]) -> ModelOutput:
    """
    Stack a list of ModelOutput objects (or its subclasses) along the batch_size dimension. The function infers the
    specific ModelOutput subclass from the list provided.
    """
    if not model_outputs:
        raise ValueError("Input list is empty.")

    # Infer the class from the first object in the list
    model_output_cls = type(model_outputs[0])

    # Ensure all objects are of the same type
    if not all(isinstance(obj, model_output_cls) for obj in model_outputs):
        raise ValueError("All elements in the list should be of the same type.")

    # Helper function to concat tensors or tuples of tensors
    def _concat(data):
        """
        Reverse of `_split` function above.
        """
        if any(data is None for data in data):
            return None
        if isinstance(data[0], torch.Tensor):
            return torch.cat(data, dim=0)
        elif isinstance(data[0], tuple):
            # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
            if isinstance(data[0][0], tuple):
                return tuple(
                    tuple(torch.cat([attr[i][j] for attr in data], dim=0) for j in range(len(data[0][0])))
                    for i in range(len(data[0]))
                )
            else:
                return tuple(torch.cat([attr[i] for attr in data], dim=0) for i in range(len(data[0])))
        elif isinstance(data[0], (int, float)):
            # If the elements are integers or floats, return a tensor
            return torch.tensor(data)
        else:
            raise ValueError(f"Unexpected attribute type: {type(data[0])}")

    # Use a dictionary comprehension to gather attributes from all objects and concatenate them
    concatenated_data = {
        k: _concat([getattr(model_output, k) for model_output in model_outputs])
        for k in model_output_cls.__dataclass_fields__.keys()
    }

    # Return a new object of the inferred class with the concatenated attributes
    return model_output_cls(**concatenated_data)