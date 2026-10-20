# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from math import isfinite

# the raw metrics durations are reported in microseconds, while the metrics below are reported in milliseconds
US_IN_MS = 1000


def get_average(values):
    return sum(values) / len(values) if len(values) > 0 else -1


def get_sd_per_model_metrics(model_metrics):
    raw_metrics = model_metrics.raw_metrics

    durations = [duration / US_IN_MS for duration in raw_metrics.m_durations]
    batch_sizes = raw_metrics.m_batch_sizes

    infer_durations = [duration / US_IN_MS for duration in raw_metrics.token_infer_durations] or durations
    tpot = model_metrics.get_tpot().mean
    return {
        "generate_duration": sum(durations),
        "ttft": model_metrics.get_ttft().mean,
        "ttst": model_metrics.get_ttst().mean,
        "tpot": tpot,
        "throughput": US_IN_MS / tpot if tpot > 0 else -1,
        "num_generated_tokens": sum(batch_sizes) if len(batch_sizes) > 0 else len(durations),
        "infer_count": len(durations),
        "first_infer_latency": infer_durations[0] if len(infer_durations) > 0 else -1,
        "other_infers_avg_latency": get_average(infer_durations[1:]),
    }


def get_sd_value(extended_perf_metrics, getter):
    if not hasattr(extended_perf_metrics, getter):
        return None
    value = getattr(extended_perf_metrics, getter)()
    return value if isfinite(value) else None


def get_sd_candidate_tokens(extended_perf_metrics):
    if not all(
        hasattr(extended_perf_metrics, getter) for getter in ("get_num_draft_tokens", "get_num_rejected_tokens")
    ):
        return None
    candidates = extended_perf_metrics.get_num_draft_tokens()
    try:
        rejected = extended_perf_metrics.get_num_rejected_tokens()
    except RuntimeError:
        # Older producers can report accepted tokens exceeding recorded candidates.
        return None
    if candidates < 0 or rejected < 0 or rejected > candidates:
        return None
    candidate_tokens = {"draft_candidate_tokens": candidates, "rejected_tokens": rejected}
    # No recorded candidates means both rates are reported as zero in both tools.
    acceptance_rate = 0.0
    if candidates > 0:
        acceptance_rate = get_sd_value(extended_perf_metrics, "get_draft_acceptance_rate")
        if acceptance_rate is None:
            acceptance_rate = (candidates - rejected) / candidates
    candidate_tokens["acceptance_rate"] = acceptance_rate * 100
    candidate_tokens["miss_rate"] = rejected / candidates * 100 if candidates > 0 else 0.0
    return candidate_tokens


def get_sd_metrics(extended_perf_metrics):
    # SDPerModelsPerfMetrics is set as extended_perf_metrics only for speculative decoding pipelines
    if not hasattr(extended_perf_metrics, "get_num_accepted_tokens"):
        return None
    # get_num_draft_processed_tokens() is the draft model specific alias of get_num_generated_tokens()
    processed = extended_perf_metrics.draft_model_metrics.get_num_generated_tokens()
    num_accepted = extended_perf_metrics.get_num_accepted_tokens()
    sd_metric = {
        "draft_processed_tokens": processed,
        "num_accepted": num_accepted,
        "acceptance_rate": None,
        "miss_rate": None,
        "draft_candidate_tokens": None,
        "rejected_tokens": None,
        "draft_to_main_inference_duration_ratio": get_sd_value(
            extended_perf_metrics, "get_draft_to_main_inference_duration_ratio"
        ),
        "main_model": get_sd_per_model_metrics(extended_perf_metrics.main_model_metrics),
        "draft_model": get_sd_per_model_metrics(extended_perf_metrics.draft_model_metrics),
    }

    candidate_tokens = get_sd_candidate_tokens(extended_perf_metrics)
    if candidate_tokens is not None:
        sd_metric.update(candidate_tokens)
    return sd_metric
