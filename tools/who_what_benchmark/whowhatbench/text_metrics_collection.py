# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import logging

import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from collections import OrderedDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationResults:
    def __init__(self, answer_text, prompt_input_ids=None, logits=None, generated_token_ids=None):
        self.answer_text = answer_text
        self.prompt_input_ids = prompt_input_ids
        self.logits = logits
        self.generated_token_ids = generated_token_ids


class Metrics(Enum):
    SIMILARITY = "similarity"
    DIVERGENCY = "divergency"
    KL_DIVERGENCY = "kl_divergency"
    TOKEN_SIMILARITY = "token_similarity"


class MetricStorageKind(Enum):
    CSV = "csv"
    NPY = "npy"
    TEXT_FILE = "text_file"


class ArtifactsSchema:
    def __init__(self, metrics_list, long_prompts=True):
        self.artifacts_schema = OrderedDict([("prompts", MetricStorageKind.CSV)])
        self.output_dir_required = False

        if long_prompts:
            self.artifacts_schema["prompts"] = MetricStorageKind.TEXT_FILE
            self.output_dir_required = True

        if Metrics.SIMILARITY.value in metrics_list or Metrics.DIVERGENCY.value in metrics_list:
            self.artifacts_schema["answers"] = MetricStorageKind.CSV

        if Metrics.TOKEN_SIMILARITY.value in metrics_list:
            self.artifacts_schema["prompt_input_ids"] = MetricStorageKind.NPY
            self.artifacts_schema["generated_token_ids"] = MetricStorageKind.NPY
            self.output_dir_required = True

        if Metrics.KL_DIVERGENCY.value in metrics_list:
            self.artifacts_schema["prompt_input_ids"] = MetricStorageKind.NPY
            self.artifacts_schema["generated_token_ids"] = MetricStorageKind.NPY
            self.artifacts_schema["logits"] = MetricStorageKind.NPY
            self.output_dir_required = True


class ArtifactsManager:
    DEFAULT_ARTIFACT_ROOT = "wwb_output"

    def __init__(self, artifacts_schema: ArtifactsSchema, artifact_root: Path | None):
        self.artifacts_schema = artifacts_schema.artifacts_schema
        self.artifact_root = None
        if artifacts_schema.output_dir_required:
            self.artifact_root = artifact_root or self.DEFAULT_ARTIFACT_ROOT
            self.artifact_root = Path(self.artifact_root)
            if self.artifact_root.exists():
                logger.warning(f"Artifact root {self.artifact_root} already exists. The data will be overwritten.")
            else:
                self.artifact_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"All artifacts will be stored to {self.artifact_root}.")

        self.artifacts = {name: [] for name in self.artifacts_schema}

    def collect(self, sample_idx: int, prompt: str, output: GenerationResults):
        for name, storage_kind in self.artifacts_schema.items():
            if name == "prompts":
                value = prompt
            elif name == "answers":
                value = output.answer_text
            else:
                value = getattr(output, name, None)

            if value is not None:
                self.artifacts[name].append(self.store(value, sample_idx, storage_kind, name))

    def collect_metadata_for_generation(self, paths_to_data=None):
        if not paths_to_data:
            return
        data = []
        for gen_token_np_path in paths_to_data:
            data.append(np.load(gen_token_np_path))
        return data

    def store(self, value, sample_idx, storage_kind, name):
        if MetricStorageKind.CSV == storage_kind:
            return value
        if MetricStorageKind.NPY == storage_kind:
            sample_dir = self.artifact_root / f"sample_{sample_idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = sample_dir / f"{name}.npy"
            tensor = value.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                # NumPy has no native bfloat16 support, so it must be upcast before conversion
                tensor = tensor.float()
            np.save(artifact_path, tensor.numpy())
            return str(artifact_path)
        if MetricStorageKind.TEXT_FILE == storage_kind:
            sample_dir = self.artifact_root / f"sample_{sample_idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = sample_dir / f"{name}_{sample_idx}.txt"
            artifact_path.write_text(value, encoding="utf-8")
            return str(artifact_path)
        raise ValueError(f"Unsupported storage kind: {storage_kind}")

    def build_csv(self, language: str, long_prompt: bool, metrcis_list: list = None):
        res_data = {"prompts": list(self.artifacts["prompts"])}
        for key in [
            "logits",
            "prompt_input_ids",
            "generated_token_ids",
            "answers",
        ]:
            if self.artifacts.get(key):
                res_data_key = key if self.artifacts_schema[key] == MetricStorageKind.CSV else f"{key}_path"
                res_data[res_data_key] = self.artifacts[key]

        df = pd.DataFrame(res_data)
        df["language"] = language
        df["prompt_length_type"] = "long" if long_prompt else "short"
        df["metrics"] = ";".join(metrcis_list) if metrcis_list else ""
        return df
