# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from whowhatbench.scenario.schema import (
    BackendEnum,
    DatasetConfig,
    DatasetTypeEnum,
    ModelConfig,
    Scenario,
    TaskConfig,
)

def build_args_namespace(
    scenario: Scenario,
    task: TaskConfig,
    target_id: str,
    output_dir: Path,
    gt_data_path: Optional[str],
) -> argparse.Namespace:
    """Map a scenario task + target into the legacy argparse.Namespace contract.

    The existing wwb call sites (load_model, create_evaluator, etc.) all expect
    an argparse.Namespace. Rather than fork those code paths, this builder
    materialises every flag the legacy ``wwb`` CLI would set, so the scenario
    runner can call into the same evaluator pipeline unchanged.
    """
    base_model_cfg: ModelConfig = scenario.models[task.base]
    target_model_cfg: ModelConfig = scenario.models[target_id]
    dataset_cfg: DatasetConfig = scenario.datasets[task.dataset]

    # Backend flags — mutually exclusive booleans
    hf = target_model_cfg.backend == BackendEnum.hf
    genai = target_model_cfg.backend == BackendEnum.genai
    llamacpp = target_model_cfg.backend == BackendEnum.llamacpp
    from_onnx = target_model_cfg.backend == BackendEnum.onnx

    # JSON-serialise dict config fields (wwb code expects JSON strings)
    def _to_json(d: Optional[dict[str, Any]]) -> Optional[str]:
        return json.dumps(d) if d is not None else None

    # Dataset routing
    dataset_str: Optional[str]
    split_str: Optional[str]
    dataset_field_str: str
    if dataset_cfg.type == DatasetTypeEnum.huggingface:
        if dataset_cfg.path is None:
            raise ValueError(f"Dataset {task.dataset!r} of type 'huggingface' must declare a 'path' field.")
        dataset_str = f"{dataset_cfg.path},{dataset_cfg.name}" if dataset_cfg.name else dataset_cfg.path
        split_str = dataset_cfg.split
        dataset_field_str = dataset_cfg.field
    else:
        # builtin, inline, csv — all handled by runner as test_data; args.dataset=None
        dataset_str = None
        split_str = None
        dataset_field_str = "text"

    # Draft model
    draft = target_model_cfg.draft
    draft_model = draft.path if draft else None
    draft_device = draft.device if draft else None
    draft_cb_config = _to_json(draft.cb_config) if draft else None

    # LoRA adapters — use target model's adapters
    adapters = target_model_cfg.adapters or None
    alphas = target_model_cfg.alphas or None

    # Language: from dataset (builtin) or default "en"
    language = dataset_cfg.language if dataset_cfg.type == DatasetTypeEnum.builtin else "en"

    # data_encoder: task-level or scenario default
    data_encoder = task.data_encoder or scenario.defaults.data_encoder

    return argparse.Namespace(
        # Models
        base_model=base_model_cfg.path,
        target_model=target_model_cfg.path,
        tokenizer=target_model_cfg.tokenizer,
        # Task
        model_type=task.type,
        # Data
        gt_data=gt_data_path,
        target_data=None,  # runner handles predictions, never via args
        dataset=dataset_str,
        dataset_field=dataset_field_str,
        split=split_str,
        # Output
        output=str(output_dir),
        # Inference settings
        device=target_model_cfg.device,
        num_samples=task.num_samples,
        seed=task.generation.seed,
        verbose=False,  # controlled by runner, never by scenario
        language=language,
        data_encoder=data_encoder,
        # Backend flags (mutually exclusive)
        hf=hf,
        genai=genai,
        llamacpp=llamacpp,
        from_onnx=from_onnx,
        # OV / CB configs (JSON strings)
        ov_config=_to_json(target_model_cfg.ov_config),
        cb_config=_to_json(target_model_cfg.cb_config),
        taylorseer_config=_to_json(target_model_cfg.taylorseer_config),
        rag_config=None,  # not yet surfaced in scenario schema v1
        # Generation params
        max_new_tokens=task.generation.max_new_tokens,
        num_inference_steps=task.generation.num_inference_steps,
        image_size=task.generation.image_size,
        video_frames_num=(
            task.generation.video_frames_num
            if task.generation.video_frames_num is not None
            else task.vlm.video_frames_num
        ),
        long_prompt=task.generation.long_prompt,
        num_assistant_tokens=(
            task.chat.num_assistant_tokens
            if task.chat.num_assistant_tokens is not None
            else task.generation.num_assistant_tokens
        ),
        assistant_confidence_threshold=(
            task.chat.assistant_confidence_threshold
            if task.chat.assistant_confidence_threshold is not None
            else task.generation.assistant_confidence_threshold
        ),
        # LoRA
        adapters=adapters,
        alphas=alphas,
        empty_adapters=task.chat.empty_adapters or target_model_cfg.empty_adapters,
        # VLM params
        pruning_ratio=task.vlm.pruning_ratio,
        relevance_weight=task.vlm.relevance_weight,
        omit_chat_template=task.vlm.omit_chat_template or task.chat.omit_chat_template,
        # Embedding params
        embeds_pooling_type=task.embeddings.pooling_type,
        embeds_normalize=task.embeddings.normalize,
        embeds_padding_side=task.embeddings.padding_side,
        embeds_batch_size=task.embeddings.batch_size,
        # Speech params
        speaker_embeddings=task.speech.speaker_embeddings,
        tts_eval_whisper_model=task.speech.whisper_model,
        vocoder_path=task.speech.vocoder_path,
        # Draft model (speculative decoding)
        draft_model=draft_model,
        draft_device=draft_device,
        draft_cb_config=draft_cb_config,
        # GGUF
        gguf_file=target_model_cfg.gguf_file,
    )
