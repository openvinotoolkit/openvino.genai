# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import logging
import pytest

import pandas as pd

from pathlib import Path
from whowhatbench import GenerationResults, TextEvaluator, TextGenerationContext, TextGenerationFn

WWB_ROOT = Path(__file__).resolve().parents[1]
if str(WWB_ROOT) not in sys.path:
    sys.path.insert(0, str(WWB_ROOT))

from whowhatbench.text_generation_strategies import (  # noqa: E402
    CompositeGenerationStrategy,
    GenAIGeneration,
    GenerationStrategy,
    LlamaCPPKLDivergencyGeneration,
    LlamaCPPGeneration,
    KLDivergencyGeneration,
    OptimumHFGeneration,
)

from whowhatbench.artifacts_manager import Metrics  # noqa: E402

from conftest import run_wwb  # noqa: E402
from test_cli_text import base_model_path, target_model_path  # noqa: E402


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SIMILARITY = Metrics.SIMILARITY.value
KL_DIVERGENCY = Metrics.KL_DIVERGENCY.value
TOKEN_SIMILARITY = Metrics.TOKEN_SIMILARITY.value


@pytest.mark.parametrize(
    ("metrics_list", "is_genai", "is_llamacpp", "expected_cls"),
    [
        ([SIMILARITY], False, False, OptimumHFGeneration),
        ([SIMILARITY], True, False, GenAIGeneration),
        ([SIMILARITY], False, True, LlamaCPPGeneration),
        ([KL_DIVERGENCY], False, False, KLDivergencyGeneration),
        ([KL_DIVERGENCY], False, True, LlamaCPPKLDivergencyGeneration),
    ],
)
def test_factory_creates_single_strategy(metrics_list, is_genai, is_llamacpp, expected_cls):
    strategy = GenerationStrategy.create(metrics_list, is_genai, is_llamacpp)

    assert type(strategy) is expected_cls


@pytest.mark.parametrize("use_chat_template", [False, True])
def test_llamacpp_generation_preserves_special_tokens(use_chat_template):
    from unittest.mock import Mock

    model = Mock()
    prompt = "Hello <|im_start|> café"
    prompt_tokens = [1, 42, 73]
    model.tokenize.return_value = prompt_tokens
    model.return_value = {"choices": [{"text": "completion"}]}
    model.create_chat_completion.return_value = {"choices": [{"message": {"content": "chat"}}]}

    strategy = GenerationStrategy.create([SIMILARITY], False, True)
    result = strategy.run_generation(model, None, prompt, 8, True, use_chat_template=use_chat_template)

    if use_chat_template:
        assert result.answer_text == "chat"
        model.create_chat_completion.assert_called_once_with(
            messages=[{"role": "user", "content": prompt}], max_tokens=8, temperature=0.0
        )
        model.tokenize.assert_not_called()
        model.assert_not_called()
    else:
        assert result.answer_text == "completion"
        model.tokenize.assert_called_once_with(prompt.encode("utf-8"), add_bos=True, special=True)
        model.assert_called_once_with(prompt_tokens, max_tokens=8, echo=False, temperature=0.0)
        model.create_chat_completion.assert_not_called()


@pytest.mark.parametrize(
    ("is_llamacpp", "expected_classes"),
    [
        (False, [OptimumHFGeneration, KLDivergencyGeneration]),
        (True, [LlamaCPPGeneration, LlamaCPPKLDivergencyGeneration]),
    ],
)
def test_factory_creates_composite_for_similarity_and_kl(is_llamacpp, expected_classes):
    strategy = GenerationStrategy.create([SIMILARITY, KL_DIVERGENCY], False, is_llamacpp, kld_ctx=8, kld_chunk=3)

    assert isinstance(strategy, CompositeGenerationStrategy)
    assert [type(inner) for inner in strategy.strategies] == expected_classes
    assert strategy.produced_fields == frozenset({"answer_text", "prompt_input_ids", "logits"})

    reference_strategy = strategy.strategies[1]
    assert reference_strategy.kld_ctx == 8
    assert reference_strategy.kld_chunk == 3


def _run_text_wwb(tmp_path, metrics_list, base_model=base_model_path, target_model=target_model_path, extra_args=()):
    gt_data_path = tmp_path / "gt.csv"
    output_dir = tmp_path / "target"
    output = run_wwb(
        [
            "--base-model",
            base_model,
            "--target-model",
            target_model,
            "--gt-data",
            gt_data_path,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
            "--output",
            output_dir,
            "--metrics",
            *metrics_list,
            *extra_args,
        ]
    )
    assert "Metrics for model" in output
    gt_data = pd.read_csv(gt_data_path, keep_default_na=False)
    metrics = pd.read_csv(output_dir / "metrics.csv")
    return gt_data, metrics


def _assert_npy_artifacts_exist(data, column):
    assert column in data.columns
    assert all(Path(path).exists() for path in data[column].values)


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    # Reference artifacts are stored relative to the working directory of the wwb process
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_cli_genai_self_sufficient_strategy(chdir_tmp):
    gt_data, metrics = _run_text_wwb(chdir_tmp, [SIMILARITY], extra_args=["--genai"])

    assert len(gt_data["answers"].values) == 1
    assert "similarity" in metrics.columns
    assert metrics["similarity"].values[0] > 0.99
    assert "kl_divergency" not in metrics.columns


def test_cli_reference_strategy_kl_divergency(tmp_path):
    gt_data, metrics = _run_text_wwb(tmp_path, [KL_DIVERGENCY])

    gt_data_path = tmp_path / "gt.csv"
    output_dir = tmp_path / "target"

    run_wwb(
        [
            "--base-model",
            base_model_path,
            "--gt-data",
            gt_data_path,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
            "--output",
            output_dir,
            "--metrics",
            KL_DIVERGENCY,
            TOKEN_SIMILARITY,
            SIMILARITY,
        ]
    )

    run_wwb(
        [
            "--target-model",
            target_model_path,
            "--gt-data",
            gt_data_path,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
            "--output",
            output_dir,
            "--metrics",
            KL_DIVERGENCY,
            TOKEN_SIMILARITY,
            SIMILARITY,
        ]
    )
    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert SIMILARITY in metrics.columns
    assert KL_DIVERGENCY in metrics.columns

    assert metrics[SIMILARITY].values[0] > 0.99
    assert metrics[KL_DIVERGENCY].values[0] < 3.0

    run_wwb(
        [
            "--target-model",
            target_model_path,
            "--gt-data",
            gt_data_path,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
            "--output",
            output_dir,
            "--metrics",
            KL_DIVERGENCY,
            TOKEN_SIMILARITY,
            SIMILARITY,
        ]
    )

    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert SIMILARITY in metrics.columns
    assert KL_DIVERGENCY in metrics.columns
    assert metrics[SIMILARITY].values[0] > 0.99
    assert metrics[KL_DIVERGENCY].values[0] < 3.0


def test_cli_composite_strategy_similarity_and_kl_divergency(chdir_tmp):
    gt_data, metrics = _run_text_wwb(chdir_tmp, [SIMILARITY, KL_DIVERGENCY, TOKEN_SIMILARITY])

    assert all(answer for answer in gt_data["answers"].values)
    _assert_npy_artifacts_exist(gt_data, "logits_path")
    _assert_npy_artifacts_exist(gt_data, "prompt_input_ids_path")
    _assert_npy_artifacts_exist(gt_data, "generated_token_ids_path")
    assert "similarity" in metrics.columns
    assert "kl_divergency" in metrics.columns
    assert "kl_divergency_p99" in metrics.columns
    assert "kl_divergency_p95" in metrics.columns
    assert "kl_divergency_p10" in metrics.columns
    assert "kl_divergency_min" in metrics.columns
    assert "kl_divergency_max" in metrics.columns


def ov_gen_fn(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    crop_question: bool,
    use_chat_template: bool = False,
    empty_adapters: bool = False,
    num_assistant_tokens: int = 0,
    assistant_confidence_threshold: float = 0.0,
    generation_config_extra: dict | None = None,
    context: TextGenerationContext | None = None,
) -> GenerationResults:
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    output = output[:, inputs["input_ids"].shape[-1] :]

    return GenerationResults(
        answer_text=tokenizer.batch_decode(output, skip_special_tokens=True)[0],
    )


def genai_gen_fn(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    crop_question: bool,
    use_chat_template: bool = False,
    empty_adapters: bool = False,
    num_assistant_tokens: int = 0,
    assistant_confidence_threshold: float = 0.0,
    generation_config_extra: dict | None = None,
    context: TextGenerationContext | None = None,
) -> GenerationResults:
    output = model.generate(
        prompt,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        apply_chat_template=False,
    )

    return GenerationResults(answer_text=output)


def test_custom_gen_answer_fn_generates_reference_and_target_answers():
    from transformers import AutoTokenizer
    from optimum.intel.openvino import OVModelForCausalLM
    from openvino_genai import LLMPipeline

    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    ov_model = OVModelForCausalLM.from_pretrained(target_model_path)

    genai_model = LLMPipeline(models_path=target_model_path, device="CPU")

    ov_custom_fn: TextGenerationFn = ov_gen_fn
    genai_custom_fn: TextGenerationFn = genai_gen_fn

    evaluator = TextEvaluator(
        base_model=ov_model,
        tokenizer=tokenizer,
        test_data=["What is OpenVINO?"],
        long_prompt=False,
        max_new_tokens=16,
        gen_answer_fn=ov_custom_fn,
    )
    per_prompt, aggregate = evaluator.score(genai_model, gen_answer_fn=genai_custom_fn)

    assert aggregate["similarity"].iloc[0] >= 0.9
