from typing import Union, Tuple, List, Optional
import torch

from diffusers.utils import torch_utils


def new_randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"],
                              "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    latents = torch.zeros(shape).view(-1)
    for i in range(latents.shape[0]):
        latents[i] = torch.randn(
            1, generator=generator, dtype=torch.float32).item()

    return latents.view(shape)


def patch_diffusers():
    torch_utils.randn_tensor = new_randn_tensor
