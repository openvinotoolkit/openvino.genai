# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path

import pytest
import gc

from openvino_genai import (
    ContinuousBatchingPipeline,
    GenerationConfig,
    SparseAttentionMode,
    SchedulerConfig,
)

from utils.constants import get_default_llm_properties
from utils.hugging_face import download_and_convert_model


def load_prompts_dataset(file_name: str) -> dict[str, list[str]]:
    tests_root = Path(__file__).parent
    file_path = tests_root / "data" / file_name
    with open(file_path, "r", encoding="utf-8") as f:
        return {"prompts": [line for line in f]}


@dataclass(frozen=True)
class XAttentionSimilarityTestData:
    test_id: str
    prompt_file: str
    max_new_tokens: int
    similarity_threshold: float


@pytest.mark.parametrize(
    "test_struct",
    [
        XAttentionSimilarityTestData(
            test_id="short_prompts",
            prompt_file="short_prompts.txt",
            max_new_tokens=64,
            similarity_threshold=0.90,
        ),
        XAttentionSimilarityTestData(
            test_id="long_prompts",
            prompt_file="long_prompts.txt",
            max_new_tokens=64,
            similarity_threshold=0.80,
        ),
    ],
    ids=lambda x: x.test_id,
)
def test_xattention_enabled_vs_disabled_similarity(test_struct, monkeypatch, capfd):
    import whowhatbench

    seqs_per_request = 1
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    model_schema = download_and_convert_model(model_id)
    tokenizer = model_schema.hf_tokenizer
    models_path = model_schema.models_path

    monkeypatch.setenv("OPENVINO_LOG_LEVEL", "5")

    scheduler_cfg_no_xattn = SchedulerConfig()
    scheduler_cfg_no_xattn.use_sparse_attention = False

    scheduler_cfg_xattn = SchedulerConfig()
    scheduler_cfg_xattn.use_sparse_attention = True
    scheduler_cfg_xattn.sparse_attention_config.num_last_dense_tokens_in_prefill = 10
    scheduler_cfg_xattn.sparse_attention_config.mode = SparseAttentionMode.XATTENTION
    scheduler_cfg_xattn.sparse_attention_config.xattention_threshold = 0.9
    scheduler_cfg_xattn.sparse_attention_config.xattention_block_size = 128
    scheduler_cfg_xattn.sparse_attention_config.xattention_stride = 16

    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = test_struct.max_new_tokens
    generation_config.apply_chat_template = False

    model_no_xattn = ContinuousBatchingPipeline(
        models_path,
        scheduler_cfg_no_xattn,
        "CPU",
        {},
        get_default_llm_properties(),
    )

    model_xattn = ContinuousBatchingPipeline(
        models_path,
        scheduler_cfg_xattn,
        "CPU",
        {},
        get_default_llm_properties(),
    )
    xattn_logs = capfd.readouterr()
    xattn_log_text = xattn_logs.out + xattn_logs.err

    assert "use_sparse_attention: true" in xattn_log_text
    assert "sparseAttentionMode: XATTENTION" in xattn_log_text
    assert "xattention_threshold: 0.9" in xattn_log_text
    assert "xattention_block_size: 128" in xattn_log_text
    assert "xattention_stride: 16" in xattn_log_text

    data_dict = load_prompts_dataset(test_struct.prompt_file)
    evaluator = whowhatbench.Evaluator(
        base_model=model_no_xattn,
        tokenizer=tokenizer,
        test_data=data_dict,
        generation_config=generation_config,
        generation_config_base=generation_config,
        max_new_tokens=test_struct.max_new_tokens,
        seqs_per_request=seqs_per_request,
    )

    _, all_metrics = evaluator.score(model_xattn)
    similarity_metric = float(all_metrics["similarity"][0])
    print(f"XAttention similarity: {similarity_metric}")

    del evaluator
    del model_xattn
    del model_no_xattn
    del data_dict

    gc.collect()

    assert similarity_metric > test_struct.similarity_threshold
