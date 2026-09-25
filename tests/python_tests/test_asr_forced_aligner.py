# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import functools
import gc
import json
import math

import numpy as np
import openvino_genai as ov_genai
import pytest

from utils.asr_utils.qwen3_asr import (
    forced_aligner_audio,
    forced_aligner_model_path,
    make_broken_aligner,
    qwen3_asr_model_path,
    shared_forced_aligner,
)


def assert_monotonic_word_list(words):
    assert isinstance(words, list)
    assert len(words) > 0
    previous_end = -1.0
    for word in words:
        assert math.isfinite(word.start_ts) and math.isfinite(word.end_ts)
        assert word.start_ts <= word.end_ts
        assert word.start_ts >= previous_end - 1e-3
        previous_end = word.end_ts


def test_asr_forced_aligner_align():
    words = shared_forced_aligner().align(forced_aligner_audio(), "how are you doing today", language="english")
    assert_monotonic_word_list(words)


def test_asr_forced_aligner_state_reset():
    # Use a fresh aligner so the test does not depend on shared-instance history.
    aligner = ov_genai.ASRForcedAligner(forced_aligner_model_path(), "CPU")
    audio_a = forced_aligner_audio()
    audio_b = (np.random.default_rng(1).standard_normal(16000) * 0.01).astype(np.float32).tolist()

    first_a = aligner.align(audio_a, "how are you doing today", language="english")
    aligner.align(audio_b, "hello world today", language="english")
    second_a = aligner.align(audio_a, "how are you doing today", language="english")

    assert [(w.text, w.start_ts, w.end_ts) for w in first_a] == [(w.text, w.start_ts, w.end_ts) for w in second_a]


def test_asr_forced_aligner_chinese_alignment():
    # Chinese exercises the CJK path: each Han character becomes its own alignment unit.
    words = shared_forced_aligner().align(forced_aligner_audio(), "你好世界", language="chinese")
    assert_monotonic_word_list(words)
    assert [word.text for word in words] == ["你", "好", "世", "界"]


@pytest.mark.parametrize(
    "language_kwargs,message",
    [
        ({}, "requires a language"),
        ({"language": "japanese"}, "not yet implemented by OpenVINO GenAI"),
    ],
)
def test_asr_forced_aligner_language_errors(language_kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        shared_forced_aligner().align(forced_aligner_audio(), "how are you", **language_kwargs)


def test_asr_forced_aligner_bad_directory_rejected():
    # A normal Qwen3-ASR export lacks the forced-aligner config fields.
    with pytest.raises(RuntimeError, match="not a Qwen3 forced aligner"):
        ov_genai.ASRForcedAligner(qwen3_asr_model_path(), "CPU")


# classify_num is model-configured and must match the decoder logits width.
@pytest.mark.parametrize(
    "mutation,message",
    [
        ("non_positive", "classify_num must be a positive integer"),
        ("width_mismatch", "decoder logits last dimension must equal classify_num"),
    ],
)
def test_asr_forced_aligner_invalid_classify_num_rejected(mutation, message, tmp_path):
    def mutate(cfg):
        if mutation == "non_positive":
            cfg["thinker_config"]["classify_num"] = 0
        else:
            cfg["thinker_config"]["classify_num"] = int(cfg["thinker_config"]["classify_num"]) + 1

    broken = make_broken_aligner(forced_aligner_model_path(), tmp_path / "broken-aligner", mutate)
    with pytest.raises(RuntimeError, match=message):
        ov_genai.ASRForcedAligner(str(broken), "CPU")


# ASRPipeline forced-aligner integration.


@functools.lru_cache()
def _qwen3_asr_pipeline_with_aligner():
    return ov_genai.ASRPipeline(qwen3_asr_model_path(), "CPU", forced_aligner=shared_forced_aligner())


def test_qwen3_asr_word_timestamps_requires_forced_aligner():
    audio = forced_aligner_audio()
    pipe = ov_genai.ASRPipeline(qwen3_asr_model_path(), "CPU")

    pipe.generate(audio, max_new_tokens=4)

    with pytest.raises(RuntimeError, match="forced aligner"):
        pipe.generate(audio, word_timestamps=True, max_new_tokens=4)


def test_forced_aligner_rejected_for_non_qwen(tmp_path):
    # The dispatcher rejects the property before constructing a non-Qwen backend.
    model_dir = tmp_path / "whisper"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    with pytest.raises(RuntimeError, match="only supported for Qwen3-ASR"):
        ov_genai.ASRPipeline(model_dir, "CPU", forced_aligner=shared_forced_aligner())


def test_qwen3_asr_forced_aligner_toggle():
    pipe = _qwen3_asr_pipeline_with_aligner()

    disabled = pipe.generate(forced_aligner_audio(), word_timestamps=False, max_new_tokens=6)
    assert disabled.words is None

    enabled = pipe.generate(forced_aligner_audio(), word_timestamps=True, language="English", max_new_tokens=6)
    assert enabled.perf_metrics.get_word_level_timestamps_processing_duration().mean > 0


def test_qwen3_asr_forced_aligner_instance_outlives_local():
    # A fresh aligner is required: the pipeline must remain the sole owner after the local is dropped.
    aligner = ov_genai.ASRForcedAligner(forced_aligner_model_path(), "CPU")
    pipe = ov_genai.ASRPipeline(qwen3_asr_model_path(), "CPU", forced_aligner=aligner)

    del aligner
    gc.collect()

    result = pipe.generate(forced_aligner_audio(), word_timestamps=True, language="English", max_new_tokens=6)
    assert result.words is not None
    assert_monotonic_word_list(result.words[0])


def test_qwen3_asr_forced_aligner_offset_zero_matches_standalone():
    # A single chunk has zero offset, so integrated and standalone alignment must match exactly.
    audio = forced_aligner_audio()

    pipe = _qwen3_asr_pipeline_with_aligner()
    result = pipe.generate(audio, word_timestamps=True, language="English", max_new_tokens=6)
    integrated = result.words[0]

    standalone = shared_forced_aligner().align(audio, result.texts[0], language="english")

    assert len(integrated) > 0
    assert [(w.text, w.start_ts, w.end_ts) for w in integrated] == [(w.text, w.start_ts, w.end_ts) for w in standalone]


def test_qwen3_asr_forced_aligner_multichunk_runs():
    # The tiny random classifier cannot validate timestamp quality; verify that the multi-chunk path
    # executes and records multiple encoder invocations.
    pipe = _qwen3_asr_pipeline_with_aligner()
    duration_sec = 182
    audio = (np.random.default_rng(3).standard_normal(16000 * duration_sec) * 0.01).astype(np.float32).tolist()
    result = pipe.generate(audio, word_timestamps=True, language="English", max_new_tokens=4)

    assert len(result.perf_metrics.asr_raw_metrics.encode_inference_durations) >= 2
    assert result.words is not None
    assert len(result.words) == 1
    for sample_words in result.words:
        for word in sample_words:
            assert math.isfinite(word.start_ts) and math.isfinite(word.end_ts)
            assert word.start_ts >= 0.0
            assert word.start_ts <= word.end_ts
