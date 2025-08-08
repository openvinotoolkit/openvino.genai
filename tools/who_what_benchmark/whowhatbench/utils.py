from typing import Union, Optional
from packaging.version import Version
import torch
import transformers
from contextlib import contextmanager


def new_randn_tensor(
    shape: Union[tuple, list],
    generator: Optional[Union[list["torch.Generator"],
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
    from diffusers.utils import torch_utils
    torch_utils.randn_tensor = new_randn_tensor


@contextmanager
def mock_AwqQuantizer_validate_environment(to_patch):
    original_fun = transformers.quantizers.quantizer_awq.AwqQuantizer.validate_environment
    if to_patch:
        transformers.quantizers.quantizer_awq.AwqQuantizer.validate_environment = lambda self, device_map, **kwargs: None
    try:
        yield
    finally:
        if to_patch:
            transformers.quantizers.quantizer_awq.AwqQuantizer.validate_environment = original_fun


@contextmanager
def mock_torch_cuda_is_available(to_patch):
    try:
        # import bnb before patching for avoid attempt to load cuda extension during first import
        import bitsandbytes as bnb  # noqa: F401
    except ImportError:
        pass
    original_is_available = torch.cuda.is_available
    if to_patch:
        torch.cuda.is_available = lambda: True
    try:
        yield
    finally:
        if to_patch:
            torch.cuda.is_available = original_is_available


@contextmanager
def patch_awq_for_inference(to_patch):
    orig_gemm_forward = None
    if to_patch:
        # patch GEMM module to allow inference without CUDA GPU
        from awq.modules.linear.gemm import WQLinearMMFunction
        from awq.utils.packing_utils import dequantize_gemm

        def new_forward(
            ctx,
            x,
            qweight,
            qzeros,
            scales,
            w_bit=4,
            group_size=128,
            bias=None,
            out_features=0,
        ):
            ctx.out_features = out_features

            out_shape = x.shape[:-1] + (out_features,)
            x = x.to(torch.float16)
            out = dequantize_gemm(qweight.to(torch.int32), qzeros.to(torch.int32), scales, w_bit, group_size)
            out = torch.matmul(x, out.to(x.dtype))

            out = out + bias if bias is not None else out
            out = out.reshape(out_shape)

            if len(out.shape) == 2:
                out = out.unsqueeze(0)
            return out

        orig_gemm_forward = WQLinearMMFunction.forward
        WQLinearMMFunction.forward = new_forward
    try:
        yield
    finally:
        if orig_gemm_forward is not None:
            WQLinearMMFunction.forward = orig_gemm_forward


def get_ignore_parameters_flag():
    from transformers import __version__

    transformers_version = Version(__version__)

    if transformers_version >= Version("4.51.0"):
        return {"use_model_defaults": False}
    return {}
