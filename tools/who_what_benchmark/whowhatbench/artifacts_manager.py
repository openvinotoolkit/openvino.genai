# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any

import torch
import logging

import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from collections import OrderedDict
from whowhatbench.text_generation_strategies import (
    GenAIGeneration,
    OptimumHFGeneration,
    LlamaCPPGeneration,
    KLDivergencyGeneration,
    LlamaCPPKLDivergencyGeneration,
    GenerationResults,
    Metrics,
)
from whowhatbench.whowhat_metrics import TextSimilarity, TextDivergency, KLDivergency


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricStorageKind(Enum):
    CSV = "csv"
    NPY = "npy"
    TEXT_FILE = "text_file"


class Artifact(Enum):
    PROMPTS = "prompts"
    ANSWER = "answers"
    PROMPT_INPUT_IDS = "prompt_input_ids"
    GENERATED_TOKEN_IDS = "generated_token_ids"
    LOGITS = "logits"


@dataclass(frozen=True)
class MetricContext:
    tokenizer: Any = None
    similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2"
    kld_ctx: int | None = None
    kld_chunk: int | None = None


class MetricSpec(ABC):
    name: str = None

    def __init__(self, long_prompts=False, **kwargs):
        self.required_artifacts: dict[Artifact, MetricStorageKind] = OrderedDict(
            [(Artifact.PROMPTS, MetricStorageKind.CSV)]
        )
        self.output_dir_required = False

        if long_prompts:
            self.required_artifacts[Artifact.PROMPTS] = MetricStorageKind.TEXT_FILE
            self.output_dir_required = True

    @abstractmethod
    def create_score_evaluator(self, context):
        """Returns an object with evaluate(gt, prediction) -> (metrics, metrics_per_prompt)."""

    @staticmethod
    def get_generator_cls(is_genai, is_llamacpp):
        if is_genai:
            return GenAIGeneration
        if is_llamacpp:
            return LlamaCPPGeneration
        return OptimumHFGeneration


class SimilaritySpec(MetricSpec):
    name = Metrics.SIMILARITY.value

    def __init__(self, long_prompts=False, **kwargs):
        super().__init__(long_prompts, **kwargs)
        self.required_artifacts.update({Artifact.ANSWER: MetricStorageKind.CSV})

    def create_score_evaluator(self, context):
        return TextSimilarity(context.similarity_model_id)


class DivergencySpec(MetricSpec):
    name = Metrics.DIVERGENCY.value

    def __init__(self, long_prompts=False, **kwargs):
        super().__init__(long_prompts, **kwargs)
        self.required_artifacts.update({Artifact.ANSWER: MetricStorageKind.CSV})

    def create_score_evaluator(self, context):
        if context.tokenizer is None:
            raise ValueError("Tokenizer must be provided in the context for 'divergency' metric.")
        return TextDivergency(context.tokenizer)


class TokenSimilaritySpec(MetricSpec):
    name = Metrics.TOKEN_SIMILARITY.value

    def __init__(self, long_prompts=False, **kwargs):
        super().__init__(long_prompts, **kwargs)
        self.required_artifacts.update(
            {Artifact.GENERATED_TOKEN_IDS: MetricStorageKind.NPY, Artifact.PROMPT_INPUT_IDS: MetricStorageKind.NPY}
        )
        self.output_dir_required = True

    def create_score_evaluator(self, context):
        return None


class KLDivergencySpec(MetricSpec):
    name = Metrics.KL_DIVERGENCY.value

    def __init__(self, long_prompts=False, **kwargs):
        super().__init__(long_prompts, **kwargs)
        self.required_artifacts.update(
            {Artifact.LOGITS: MetricStorageKind.NPY, Artifact.PROMPT_INPUT_IDS: MetricStorageKind.NPY}
        )
        self.output_dir_required = True

    def create_score_evaluator(self, context):
        return KLDivergency()

    @staticmethod
    def get_generator_cls(is_genai, is_llamacpp, **kwargs):
        if is_genai:
            raise ValueError("KL divergency is not supported for GenAI models.")
        if is_llamacpp:
            return LlamaCPPKLDivergencyGeneration
        return KLDivergencyGeneration


class MetricsStrategy:
    metric_types = {
        Metrics.SIMILARITY.value: SimilaritySpec,
        Metrics.DIVERGENCY.value: DivergencySpec,
        Metrics.TOKEN_SIMILARITY.value: TokenSimilaritySpec,
        Metrics.KL_DIVERGENCY.value: KLDivergencySpec,
    }

    def __init__(self, metrics_list, long_prompt=False):
        self.output_dir_required = False
        self.artifacts_schema = OrderedDict()

        self.metrics = []
        for metric in metrics_list:
            new_metric = MetricsStrategy.metric_types[metric](long_prompts=long_prompt)
            self.metrics.append(new_metric)
            if self.metrics[-1].output_dir_required:
                self.output_dir_required = True

            for artifact_name, storage_kind in new_metric.required_artifacts.items():
                self.artifacts_schema[artifact_name.value] = storage_kind


class ArtifactsManager:
    DEFAULT_ARTIFACT_ROOT = "wwb_output"

    def __init__(self, metrics: MetricsStrategy, artifact_root: Path | None):
        self.artifacts_schema = metrics.artifacts_schema
        self.artifact_root = None
        self.metrics_list = {metric.name for metric in metrics.metrics}
        if metrics.output_dir_required:
            self.artifact_root = artifact_root or self.DEFAULT_ARTIFACT_ROOT
            self.artifact_root = Path(self.artifact_root)
            if self.artifact_root.exists():
                logger.warning(f"Artifact root {self.artifact_root} already exists. The data will be overwritten.")
            else:
                self.artifact_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"All artifacts will be stored in {self.artifact_root}.")

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

    def build_csv(self, language: str, long_prompt: bool):
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
        df["metrics"] = ";".join(self.metrics_list) if self.metrics_list else ""
        return df
