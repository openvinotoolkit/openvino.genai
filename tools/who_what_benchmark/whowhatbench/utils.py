from typing import Any, Union, Tuple, List, Optional
import torch
import numpy as np

from diffusers.utils import torch_utils


def new_randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    latents = torch.zeros(shape)  # Create an empty array
    # Fill the array using nested loops
    for i in range(shape[0]):         # Loop over the first dimension
        for j in range(shape[1]):     # Loop over the second dimension
            for k in range(shape[2]): # Loop over the third dimension
                for l in range(shape[3]): # Loop over the fourth dimension
                    latents[i, j, k, l] = torch.randn(1, generator=generator, dtype=torch.float32).item()

    return latents


def patch_diffusers():
    torch_utils.randn_tensor = new_randn_tensor