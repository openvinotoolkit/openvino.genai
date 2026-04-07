# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Optional
from packaging.version import Version

import os
import json
import torch
import random
import logging
import tarfile
import datasets
import itertools
import transformers

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pathlib import Path
from transformers import set_seed
from contextlib import contextmanager
from datasets.utils.file_utils import xopen
from transformers.image_utils import load_image

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

    if transformers_version >= Version("4.51.0") and transformers_version < Version("5.0.0"):
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


def normalize_lora_adapters_and_alphas(adapters, alphas):
    if adapters is None:
        return None, None

    if isinstance(adapters, (str, Path, os.PathLike)):
        adapters = [adapters]
    elif not isinstance(adapters, (list, tuple)):
        raise ValueError("`adapters` must be a non-empty list/tuple, or a single adapter path")

    if len(adapters) == 0:
        raise ValueError("`adapters` must be a non-empty list/tuple")

    if alphas is None:
        raise ValueError("`alphas` must be provided and match the number of adapters")

    if isinstance(alphas, (int, float)):
        alphas = [alphas]
    elif not isinstance(alphas, (list, tuple)):
        raise ValueError("`alphas` must be a list/tuple with one value per adapter, or a single float")

    if len(alphas) != len(adapters):
        raise ValueError("`alphas` must be the same length as `adapters`")

    return list(adapters), list(alphas)


def apply_peft_adapters(model, adapters, alphas, merged_adapter_name="merged_lora"):
    adapters, alphas = normalize_lora_adapters_and_alphas(adapters, alphas)

    from peft import PeftModel

    adapter_names = ["adapter_0"]
    model = PeftModel.from_pretrained(model, adapters[0], adapter_name=adapter_names[0])

    for idx, adapter in enumerate(adapters[1:], start=1):
        adapter_name = f"adapter_{idx}"
        model.load_adapter(adapter, adapter_name=adapter_name)
        adapter_names.append(adapter_name)

    model.add_weighted_adapter(adapter_names, alphas, merged_adapter_name)
    model.set_adapter(merged_adapter_name)

    return model


# preapre default dataset for visualtext(VLM) evalutor
def preprocess_fn(example):
    return {
        "prompts": example["instruction"],
        "images": load_image(example["image_url"]),
        "videos": None,
    }


def prepare_default_data_image(num_samples=None):
    DATASET_NAME = "ucla-contextual/contextual_test"
    NUM_SAMPLES = 24 if num_samples is None else num_samples
    set_seed(42)
    default_dataset = datasets.load_dataset(
        DATASET_NAME, split="test", streaming=True
    ).shuffle(42).take(NUM_SAMPLES)
    return default_dataset.map(
        lambda x: preprocess_fn(x), remove_columns=default_dataset.column_names
    )


def prepare_default_data_video(num_samples=None, num_frames=10):
    from huggingface_hub import hf_hub_download
    from transformers.video_utils import load_video

    DATASET_NAME = "lmms-lab/LLaVA-Video-178K"
    SUBSET = "30_60_s_academic_v0_1"
    NUM_SAMPLES = 24 if num_samples is None else num_samples

    questions_per_video_set = datasets.load_dataset(DATASET_NAME, SUBSET,
                                                    split="open_ended",
                                                    data_files={"open_ended": f"{SUBSET}/30_60_s_academic_oe_v0_1_qa_processed.json"})
    questions_per_video = {val['video']: val for val in questions_per_video_set}

    # 30_60_s_academic_v0_1_videos_10.tar.gz - just the most lightweight chunk among subset
    # https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main/30_60_s_academic_v0_1
    # the archive contains 56 videos
    videos_arc_path = hf_hub_download(repo_id="lmms-lab/LLaVA-Video-178K",
                                      filename=f"{SUBSET}/{SUBSET}_videos_10.tar.gz",
                                      repo_type="dataset")

    video_samples = []
    extract_dir = "./videos"
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(videos_arc_path, "r:gz") as tar:
        all_videos = tar.getnames()

        if len(all_videos) < NUM_SAMPLES:
            logger.warning(f"The required number of samples {NUM_SAMPLES} exceeds the available amount of data {len(all_videos)}."
                           f"num-samples will be updated to max available: {len(all_videos)}.")
            NUM_SAMPLES = len(all_videos)

        video_samples = random.Random(42).sample(all_videos, NUM_SAMPLES)  # nosec
        for sample in video_samples:
            tar.extract(sample, path=extract_dir)

    # if num_frames < total_num_frames, sample each total_num_frames/num_frames frames or sample all frames
    def default_sample_indices_fn(metadata, **kwargs):
        total_num_frames = metadata.total_num_frames
        if num_frames < total_num_frames:
            return np.arange(0, total_num_frames, total_num_frames / num_frames, dtype=int)
        return np.arange(0, total_num_frames, dtype=int)

    data = []
    for video_rel_path in video_samples:
        video_tensor = load_video(os.path.join(extract_dir, video_rel_path), backend="opencv", sample_indices_fn=default_sample_indices_fn)
        prompt = questions_per_video[video_rel_path]['conversations'][0]['value'].replace("<image>\n", "")
        data.append({'prompts': prompt, "images": None, 'videos': video_tensor[0]})

    return data


# for patching function datasets.packaged_modules.parquet.parquet.Parquet._generate_tables
# according to code: https://github.com/huggingface/datasets/issues/7357#issuecomment-3354047772
def parquet_generate_tables(self, files, *args, **kwargs):
    if self.config.features is not None and self.config.columns is not None:
        if sorted(field.name for field in self.info.features.arrow_schema) != sorted(self.config.columns):
            raise ValueError(
                f"Tried to load parquet data with columns '{self.config.columns}' with mismatching features '{self.info.features}'"
            )

    files_list = files if files and isinstance(files[0], str) else itertools.chain.from_iterable(files)
    for file_idx, file in enumerate(files_list):
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


def disable_diffusers_model_progress_bar(model):
    if hasattr(model, "set_progress_bar_config"):
        model.set_progress_bar_config(disable=True)

def is_json_dataset(dataset: str) -> bool:
    """Check if the dataset path has a .json or .jsonl extension."""
    if dataset is None:
        return False
    return dataset.lower().endswith((".json", ".jsonl"))


def resolve_json_dataset_path(dataset_path: str) -> str:
    """Resolve JSON dataset path, checking local filesystem first, then whowhatbench.prompts resources."""
    if os.path.exists(dataset_path):
        logger.info(f"JSON dataset found at local path: {dataset_path}")
        return dataset_path

    from importlib.resources import files
    resource_path = files('whowhatbench.prompts').joinpath(dataset_path)
    if resource_path.is_file():
        logger.info(f"JSON dataset found in whowhatbench.prompts resources: {resource_path}")
        return str(resource_path)

    logger.error(f"JSON dataset file '{dataset_path}' not found in local filesystem or whowhatbench.prompts resources")
    raise FileNotFoundError(
        f"JSON dataset file '{dataset_path}' was not found as a local file or in whowhatbench.prompts resources"
    )


def read_json_dataset(dataset_path: str):
    """Read JSON dataset from file. Supports both JSON array and JSONL (line-delimited JSON) formats."""
    logger.info(f"Reading JSON dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as input_file:
        content = input_file.read()

    # Try parsing as a standard JSON array first
    try:
        items = json.loads(content)
        if not isinstance(items, list):
            raise ValueError("Top-level JSON value is not an array")
        logger.info(f"Loaded {len(items)} records from JSON dataset")
        return items
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to JSONL (line-delimited JSON)
    items = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} records from JSON dataset")
    return items


