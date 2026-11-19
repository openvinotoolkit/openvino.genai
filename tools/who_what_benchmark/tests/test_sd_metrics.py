# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from whowhatbench.artifacts_manager import GenerationResults
from whowhatbench.text_evaluator import TextEvaluator
from whowhatbench.text_generation_strategies import GenAIGeneration
from whowhatbench.utils import SDMetricsCollector

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "llm_bench"))
from llm_bench_utils.gen_output_data import gen_iterate_data  # noqa: E402
from llm_bench_utils.metrics_collection_utils import get_sd_metrics, get_sd_per_model_metrics  # noqa: E402
from llm_bench_utils.metrics_print import print_metrics  # noqa: E402


def submodel(processed=40, durations=(), batches=(), ttft=-1, ttst=-1, tpot=-1):
    return SimpleNamespace(
        raw_metrics=SimpleNamespace(
            m_durations=list(durations), m_batch_sizes=list(batches), token_infer_durations=[2000, 1000]
        ),
        get_num_generated_tokens=lambda: processed,
        get_ttft=lambda: SimpleNamespace(mean=ttft),
        get_ttst=lambda: SimpleNamespace(mean=ttst),
        get_tpot=lambda: SimpleNamespace(mean=tpot),
    )


def metrics(processed=40, candidates=32, accepted=24):
    # Mirrors the native zero-candidate getter behavior, including its accepted-count fallback.
    return SimpleNamespace(
        draft_model_metrics=submodel(processed),
        main_model_metrics=submodel(),
        get_num_accepted_tokens=lambda: accepted,
        get_num_draft_tokens=lambda: candidates,
        get_num_rejected_tokens=lambda: candidates - accepted if candidates else 0,
        get_draft_acceptance_rate=lambda: accepted / candidates if candidates else float("nan"),
    )


def collect_pair(extended):
    return get_sd_metrics(extended), SDMetricsCollector._get_token_numbers(
        SimpleNamespace(extended_perf_metrics=extended)
    )


@pytest.mark.parametrize(
    "processed,candidates,accepted,acceptance,miss",
    [
        (40, 32, 24, 75, 25),
        (6, 0, 0, 0, 0),
        (6, 0, 3, None, None),
        (0, 0, 0, 0, 0),
        (40, 32, 0, 0, 100),
        (40, 32, 32, 100, 0),
    ],
)
def test_rates_use_candidates_in_both_tools(processed, candidates, accepted, acceptance, miss):
    for numbers in collect_pair(metrics(processed, candidates, accepted)):
        assert numbers["draft_processed_tokens"] == processed
        assert numbers["draft_candidate_tokens"] == candidates
        assert numbers["num_accepted"] == accepted
        if acceptance is None:
            assert numbers["acceptance_rate"] is None
            assert numbers["miss_rate"] is None
        else:
            assert numbers["acceptance_rate"] == pytest.approx(acceptance)
            assert numbers["miss_rate"] == pytest.approx(miss)


@pytest.mark.parametrize("getter", [None, float("nan"), float("inf")])
def test_acceptance_fallback_uses_candidates(getter):
    extended = metrics()
    if getter is None:
        del extended.get_draft_acceptance_rate
    else:
        extended.get_draft_acceptance_rate = lambda: getter
    for numbers in collect_pair(extended):
        assert numbers["acceptance_rate"] == 75
        assert numbers["miss_rate"] == 25


@pytest.mark.parametrize("failure", ["missing_candidates", "missing_rejected", "invalid_counts"])
def test_unavailable_candidate_data_does_not_fall_back_to_processed(failure):
    extended = metrics()
    if failure == "missing_candidates":
        del extended.get_num_draft_tokens
    elif failure == "missing_rejected":
        del extended.get_num_rejected_tokens
    else:
        extended.get_num_rejected_tokens = lambda: 33
    for numbers in collect_pair(extended):
        assert numbers["draft_processed_tokens"] == 40
        assert numbers["num_accepted"] == 24
        assert numbers.get("acceptance_rate") is None
        assert numbers.get("miss_rate") is None


@pytest.mark.parametrize("tool", ["llm_bench", "wwb"])
def test_native_counter_consistency_errors_are_not_hidden(tool):
    extended = metrics()
    extended.get_num_rejected_tokens = Mock(side_effect=RuntimeError("inconsistent counters"))
    with pytest.raises(RuntimeError, match="inconsistent counters"):
        if tool == "llm_bench":
            get_sd_metrics(extended)
        else:
            SDMetricsCollector._get_token_numbers(SimpleNamespace(extended_perf_metrics=extended))


@pytest.mark.parametrize(
    "ttft,ttst,tpot,durations,batches",
    [
        (100, 20, 5, [100000, 20000, 10000], [1, 1, 2]),
        (95, 20, 5, [100000, 20000, 10000, 10000, 90000, 20000, 10000], [1, 1, 2, 2, 1, 1, 2]),
        (100, -1, -1, [100000], [1]),
        (100, 20, -1, [100000, 20000], [1, 1]),
        (-1, -1, -1, [], []),
    ],
)
def test_per_model_timings_use_native_means_in_milliseconds(ttft, ttst, tpot, durations, batches):
    result = get_sd_per_model_metrics(submodel(durations=durations, batches=batches, ttft=ttft, ttst=ttst, tpot=tpot))
    assert result["ttft"] == ttft
    assert result["ttst"] == ttst
    assert result["tpot"] == tpot
    assert result["throughput"] == (1000 / tpot if tpot > 0 else -1)
    assert result["generate_duration"] == sum(durations) / 1000
    assert result["first_infer_latency"] == 2
    assert result["other_infers_avg_latency"] == 1


def test_raw_inference_duration_fallback_uses_microseconds():
    model = submodel(durations=[100000, 20000, 10000], batches=[1, 1, 2], tpot=5)
    model.raw_metrics.token_infer_durations = []
    result = get_sd_per_model_metrics(model)
    assert result["first_infer_latency"] == 100
    assert result["other_infers_avg_latency"] == 15


def test_mixed_csv_keeps_zero_rows_and_missing_values():
    collector = SDMetricsCollector()
    for extended in (metrics(), metrics(0, 0, 0), metrics(6, 0, 3), None):
        collector.collect(SimpleNamespace(extended_perf_metrics=extended))
    df = pd.DataFrame({"prompts": ["a", "b", "c", "d"]})
    collector.add_columns(df)
    result = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    assert result["acceptance_rate"].iloc[:2].tolist() == [75, 0]
    assert result["miss_rate"].iloc[:2].tolist() == [25, 0]
    assert result[["acceptance_rate", "miss_rate"]].iloc[2].isna().all()
    assert result["num_accepted"].iloc[:3].tolist() == [24, 0, 3]
    assert result["draft_processed_tokens"].iloc[1] == 0
    assert result[list(collector.COLUMNS)].iloc[3].isna().all()


def test_cb_batch_totals_are_recorded_once_and_marked_with_their_scope():
    collector = SDMetricsCollector()
    batch_metrics = metrics()
    batch_metrics.get_num_draft_tokens = Mock(return_value=32)
    collector.collect_batch([SimpleNamespace(extended_perf_metrics=batch_metrics)] * 2)
    collector.collect_batch([SimpleNamespace(extended_perf_metrics=metrics(10, 8, 4))])
    df = pd.DataFrame({"answers": ["a", "b", "c"]})
    collector.add_columns(df)
    result = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    batch_metrics.get_num_draft_tokens.assert_called_once()
    assert result["answers"].tolist() == ["a", "b", "c"]
    assert result["draft_candidate_tokens"].sum() == 40
    assert result["num_accepted"].sum() == 28
    assert result[list(collector.COLUMNS)].iloc[1].isna().all()
    assert result["sd_metrics_scope"].iloc[[0, 2]].tolist() == ["batch", "batch"]
    assert result["sd_batch_size"].iloc[[0, 2]].tolist() == [2, 1]


def test_csv_preserves_unavailable_rate_columns_when_no_candidates_are_recorded():
    collector = SDMetricsCollector()
    collector.collect(SimpleNamespace(extended_perf_metrics=metrics(6, 0, 3)))
    df = pd.DataFrame({"answers": ["answer"]})
    collector.add_columns(df)
    result = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    assert set(collector.COLUMNS).issubset(result.columns)
    assert result[["acceptance_rate", "miss_rate"]].isna().all().all()
    assert result["num_accepted"].iloc[0] == 3


def test_non_sd_batch_has_no_sd_columns():
    collector = SDMetricsCollector()
    collector.collect_batch([])
    collector.collect_batch([SimpleNamespace(extended_perf_metrics=None)] * 2)
    df = pd.DataFrame({"answers": ["a", "b"]})
    collector.add_columns(df)
    assert list(df.columns) == ["answers"]


def test_non_sd_has_no_csv_columns_or_sd_log(caplog):
    collector = SDMetricsCollector()
    collector.collect("plain answer")
    df = pd.DataFrame({"answers": ["plain answer"]})
    collector.add_columns(df)
    assert list(df.columns) == ["answers"]
    assert get_sd_metrics(SimpleNamespace()) is None
    with caplog.at_level(logging.INFO):
        print_metrics(1, gen_iterate_data(), sd_metric=None)
    assert "Speculative decoding" not in caplog.text


@pytest.mark.parametrize("missing_candidates", [False, True])
def test_logger_formats_unavailable_rates(caplog, missing_candidates):
    extended = metrics(6, 0, 3)
    if missing_candidates:
        del extended.get_num_draft_tokens
    with caplog.at_level(logging.INFO):
        print_metrics(1, gen_iterate_data(), sd_metric=get_sd_metrics(extended))
    assert "acceptance rate: NA, miss rate: NA" in caplog.text


@pytest.mark.parametrize("sd_enabled", [False, True])
def test_genai_strategy_preserves_result_metrics_and_answer_text(sd_enabled):
    extended = metrics() if sd_enabled else None
    model = Mock()
    model.generate.return_value = SimpleNamespace(texts=["answer"], extended_perf_metrics=extended)
    result = GenAIGeneration(["similarity"]).run_generation(model, None, "prompt", 20, True)
    assert model.generate.call_args.args[0] == ["prompt"]
    assert result.answer_text == "answer"
    assert result.extended_perf_metrics is extended


@pytest.mark.parametrize("batched", [False, True])
def test_evaluator_keeps_per_prompt_sd_csv_with_upstream_artifacts(monkeypatch, batched):
    monkeypatch.setattr("whowhatbench.text_evaluator.TextSimilarity", lambda _: None)
    extended = [metrics(40, 32, 24), metrics(6, 0, 3), None]
    prompts = ["a", "b", "c"]
    if batched:
        model = Mock()
        model.generate.side_effect = [
            [SimpleNamespace(m_generation_ids=["answer"], extended_perf_metrics=extended[0]) for _ in range(2)],
            [SimpleNamespace(m_generation_ids=["answer"], extended_perf_metrics=None)],
        ]
        kwargs = {"generation_config": SimpleNamespace(), "seqs_per_request": 2}
    else:
        model = object()
        gen_fn = Mock(side_effect=[GenerationResults("answer", extended_perf_metrics=m) for m in extended])
        kwargs = {"gen_answer_fn": gen_fn}
    evaluator = TextEvaluator(base_model=model, test_data=prompts, long_prompt=False, metrics="similarity", **kwargs)
    result = pd.read_csv(io.StringIO(evaluator.gt_data.to_csv(index=False)))
    assert result["prompts"].tolist() == prompts
    assert result["answers"].tolist() == ["answer"] * 3
    assert result["metrics"].tolist() == ["similarity"] * 3
    assert result["acceptance_rate"].iloc[0] == 75
    assert result["miss_rate"].iloc[0] == 25
    assert result[["acceptance_rate", "miss_rate"]].iloc[1:].isna().all().all()
    if batched:
        assert result["sd_metrics_scope"].iloc[0] == "batch"
        assert result["sd_batch_size"].iloc[0] == 2
        assert result[list(SDMetricsCollector.COLUMNS)].iloc[1].isna().all()
    else:
        assert result["num_accepted"].iloc[1] == 3
        assert set(SDMetricsCollector.BATCH_COLUMNS).isdisjoint(result.columns)


@pytest.mark.parametrize("sd_enabled", [False, True])
def test_encoded_strategy_preserves_sd_metrics_and_token_artifacts(sd_enabled):
    extended = metrics() if sd_enabled else None
    model = Mock()
    tokenizer = model.get_tokenizer.return_value
    tokenizer.encode.return_value = SimpleNamespace(input_ids=SimpleNamespace(data=np.array([[1, 2]])))
    tokenizer.decode.return_value = "answer"
    model.generate.return_value = SimpleNamespace(tokens=[[3, 4]], extended_perf_metrics=extended)
    result = GenAIGeneration(["token_similarity"]).run_generation(model, None, "prompt", 20, True)
    assert result.answer_text == "answer"
    assert result.extended_perf_metrics is extended
    assert result.prompt_input_ids.tolist() == [[1, 2]]
    assert result.generated_token_ids.tolist() == [3, 4]
