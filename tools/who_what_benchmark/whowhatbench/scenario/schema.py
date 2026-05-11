# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

__all__ = [
    "ValidationError",
    "MODEL_TYPE",
    "BackendEnum",
    "DatasetTypeEnum",
    "DraftModelConfig",
    "ModelConfig",
    "DatasetConfig",
    "GenerationParams",
    "EmbeddingParams",
    "SpeechParams",
    "VlmParams",
    "ChatParams",
    "TaskConfig",
    "DefaultsConfig",
    "ReportConfig",
    "Scenario",
]

MODEL_TYPE = Literal[
    "text",
    "text-chat",
    "text-to-image",
    "text-to-video",
    "speech-generation",
    "visual-text",
    "visual-text-chat",
    "visual-video-text",
    "image-to-image",
    "image-inpainting",
    "text-embedding",
    "text-reranking",
]


class BackendEnum(str, Enum):
    hf = "hf"
    genai = "genai"
    llamacpp = "llamacpp"
    onnx = "onnx"


class DatasetTypeEnum(str, Enum):
    builtin = "builtin"
    huggingface = "huggingface"
    csv = "csv"
    inline = "inline"


class DraftModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    device: str = "CPU"
    cb_config: Optional[dict[str, Any]] = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    backend: BackendEnum
    device: str = "CPU"
    tokenizer: Optional[str] = None
    ov_config: Optional[dict[str, Any]] = None
    cb_config: Optional[dict[str, Any]] = None
    draft: Optional[DraftModelConfig] = None
    adapters: list[str] = []
    alphas: list[float] = []
    empty_adapters: bool = False
    gguf_file: Optional[str] = None
    taylorseer_config: Optional[dict[str, Any]] = None


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: DatasetTypeEnum
    path: Optional[str] = None
    name: Optional[str] = None
    split: str = "validation"
    field: str = "text"
    language: str = "en"
    prompts: Optional[list[str]] = None
    passages: Optional[list[str]] = None
    chats: Optional[list[Any]] = None


class GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_new_tokens: int = 128
    num_inference_steps: Optional[int] = None
    image_size: Optional[int] = None
    video_frames_num: Optional[int] = None
    long_prompt: bool = False
    num_assistant_tokens: Optional[int] = None
    assistant_confidence_threshold: Optional[float] = None
    seed: int = 42


class EmbeddingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pooling_type: Optional[Literal["cls", "mean", "last_token"]] = None
    normalize: bool = False
    padding_side: Optional[Literal["left", "right"]] = None
    batch_size: Optional[int] = None


class SpeechParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker_embeddings: Optional[str] = None
    whisper_model: str = "base.en"
    vocoder_path: Optional[str] = None


class VlmParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_frames_num: Optional[int] = None
    pruning_ratio: Optional[int] = None
    relevance_weight: Optional[float] = None
    omit_chat_template: bool = False


class ChatParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    omit_chat_template: bool = False
    num_assistant_tokens: Optional[int] = None
    assistant_confidence_threshold: Optional[float] = None
    empty_adapters: bool = False


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: MODEL_TYPE
    base: str
    targets: list[str]
    dataset: str
    num_samples: Optional[int] = None
    seed: Optional[int] = None
    data_encoder: Optional[str] = None
    gt_data: Optional[str] = None
    generation: GenerationParams = GenerationParams()
    embeddings: EmbeddingParams = EmbeddingParams()
    speech: SpeechParams = SpeechParams()
    vlm: VlmParams = VlmParams()
    chat: ChatParams = ChatParams()

    @field_validator("targets")
    @classmethod
    def targets_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("targets must be a non-empty list")
        return v


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: str = "CPU"
    num_samples: Optional[int] = None
    seed: int = 42
    data_encoder: str = "sentence-transformers/all-mpnet-base-v2"
    output_dir: str = "./_wwb_runs/${scenario.name}/${timestamp}"


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    formats: list[Literal["markdown", "json"]] = ["markdown", "json"]
    group_by: Literal["task", "target"] = "task"
    worst_examples_top_k: int = 5


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int
    name: str
    description: str = ""
    defaults: DefaultsConfig = DefaultsConfig()
    models: dict[str, ModelConfig]
    datasets: dict[str, DatasetConfig]
    tasks: list[TaskConfig]
    report: ReportConfig = ReportConfig()

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported schema_version {v!r}. Only 1 is supported.")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9_-]+$", v):
            raise ValueError(
                f"Scenario name {v!r} is invalid. Must match ^[a-z0-9_-]+$ "
                "(lowercase letters, digits, hyphens, underscores only)."
            )
        return v

    @field_validator("tasks")
    @classmethod
    def tasks_non_empty(cls, v: list[TaskConfig]) -> list[TaskConfig]:
        if not v:
            raise ValueError("tasks must be a non-empty list")
        return v

    @model_validator(mode="after")
    def validate_cross_references(self) -> "Scenario":
        for task in self.tasks:
            if task.base not in self.models:
                raise ValueError(
                    f"Task {task.id!r}: base model {task.base!r} is not declared in models. "
                    f"Available models: {list(self.models.keys())}"
                )
            for target in task.targets:
                if target not in self.models:
                    raise ValueError(
                        f"Task {task.id!r}: target {target!r} is not declared in models. "
                        f"Available models: {list(self.models.keys())}"
                    )
            if task.dataset not in self.datasets:
                raise ValueError(
                    f"Task {task.id!r}: dataset {task.dataset!r} is not declared in datasets. "
                    f"Available datasets: {list(self.datasets.keys())}"
                )
        return self
