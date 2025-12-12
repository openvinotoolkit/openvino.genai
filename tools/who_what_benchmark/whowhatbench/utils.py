from typing import Union, Optional
from packaging.version import Version
import json
import torch
import logging
import itertools
import transformers

import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from contextlib import contextmanager
from datasets.utils.file_utils import xopen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def get_json_config(config):
    if config is None or (isinstance(config, str) and config.strip() == ""):
        raise ValueError("Config must be a non-empty string or path to a JSON file.")
    json_config = {}
    if Path(config).is_file():
        with open(config, 'r') as f:
            try:
                json_config = json.load(f)
            except json.JSONDecodeError:
                raise RuntimeError(f'Failed to parse JSON from file: {config}')
    else:
        try:
            json_config = json.loads(config)
        except json.JSONDecodeError:
            raise RuntimeError(f'Failed to parse JSON config: {config}')

    return json_config


# for patching function datasets.packaged_modules.parquet.parquet.Parquet._generate_tables
# according to code: https://github.com/huggingface/datasets/issues/7357#issuecomment-3354047772
def parquet_generate_tables(self, files, *args, **kwargs):
    if self.config.features is not None and self.config.columns is not None:
        if sorted(field.name for field in self.info.features.arrow_schema) != sorted(self.config.columns):
            raise ValueError(
                f"Tried to load parquet data with columns '{self.config.columns}' with mismatching features '{self.info.features}'"
            )
    for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
        try:
            with xopen(file, "rb") as f:
                parquet_file = pq.ParquetFile(f)
                if parquet_file.metadata.num_row_groups > 0:
                    batch_size = self.config.batch_size or parquet_file.metadata.row_group(0).num_rows
                try:
                    for batch_idx, record_batch in enumerate(
                        parquet_file.iter_batches(batch_size=batch_size, columns=self.config.columns)
                    ):
                        pa_table = pa.Table.from_batches([record_batch])
                        # Uncomment for debugging (will print the Arrow table size and elements)
                        # logger.warning(f"pa_table: {pa_table} num rows: {pa_table.num_rows}")
                        # logger.warning('\n'.join(str(pa_table.slice(i, 1).to_pydict()) for i in range(pa_table.num_rows)))
                        yield f"{file_idx}_{batch_idx}", self._cast_table(pa_table)
                except ValueError as e:
                    logger.error(f"Failed to read file '{file}' with error {type(e)}: {e}")
                    raise
        except (pa.ArrowInvalid, ValueError) as e:
            if self.config.on_bad_files == "error":
                logger.warning(f"Failed to read file '{file}' with error {type(e).__name__}: {e}")
                raise
            elif self.config.on_bad_files == "warn":
                logger.warning(f"Skipping bad file '{file}'. {type(e).__name__}: {e}`")
            else:
                logger.warning(f"Skipping bad file '{file}'. {type(e).__name__}: {e}`")
