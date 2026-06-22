# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404
from argparse import Namespace
from pathlib import Path

import pytest

from conftest import run_wwb
from ov_utils import download_hf_files_to_cache
from ov_utils import get_ov_cache_dir
from whowhatbench import model_loaders
from whowhatbench import wwb


def _assert_wwb_cli_error(args: list[str], expected_message: str) -> str:
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_wwb(args)

    output = error.value.output
    assert expected_message in output
    return output


def test_text_llamacpp_chat_requires_llamacpp(tmp_path):
    gt_data = tmp_path / "gt.csv"
    gt_data.write_text("stub\n", encoding="utf-8")

    _assert_wwb_cli_error(
        [
            "--gt-data",
            str(gt_data),
            "--model-type",
            "text",
            "--llamacpp-chat",
        ],
        "--llamacpp-chat requires --llamacpp",
    )


def test_text_llamacpp_n_ctx_requires_llamacpp():
    with pytest.raises(ValueError, match="--llamacpp-n-ctx requires --llamacpp"):
        model_loaders.load_model(
            "text",
            "dummy_model",
            "CPU",
            None,
            False,
            False,
            False,
            llamacpp_n_ctx=4096,
        )


def test_text_non_llamacpp_run_not_blocked_by_n_ctx_default(tmp_path):
    output = _assert_wwb_cli_error(
        [
            "--gt-data",
            str(tmp_path / "missing.csv"),
            "--model-type",
            "text",
        ],
        "Ground-truth data file",
    )

    assert "--llamacpp-n-ctx requires --llamacpp" not in output


def test_text_base_model_load_passes_llamacpp_flag(monkeypatch):
    calls = []

    args = Namespace(
        base_model="dummy_base_model",
        target_model=None,
        tokenizer=None,
        omit_chat_template=False,
        gt_data=None,
        target_data=None,
        model_type="text",
        data_encoder="sentence-transformers/all-mpnet-base-v2",
        dataset=None,
        dataset_field="text",
        split=None,
        output=None,
        num_samples=None,
        verbose=False,
        device="CPU",
        ov_config=None,
        language="en",
        hf=False,
        genai=False,
        cb_config=None,
        llamacpp=True,
        llamacpp_chat=False,
        llamacpp_n_ctx=None,
        image_size=None,
        num_inference_steps=None,
        seed=42,
        taylorseer_config=None,
        from_onnx=False,
        adapters=None,
        alphas=None,
        long_prompt=False,
        short_prompt=True,
        empty_adapters=False,
        embeds_pooling_type=None,
        embeds_normalize=False,
        embeds_padding_side=None,
        embeds_batch_size=None,
        rag_config=None,
        gguf_file=None,
        draft_model=None,
        draft_device=None,
        draft_cb_config=None,
        num_assistant_tokens=None,
        assistant_confidence_threshold=None,
        video_frames_num=None,
        speaker_embeddings=None,
        speech_language="",
        speech_voice="",
        tts_eval_whisper_model="base.en",
        vocoder_path=None,
        pruning_ratio=None,
        relevance_weight=None,
        max_new_tokens=128,
    )

    def fake_load_model(*load_args, **load_kwargs):
        calls.append((load_args, load_kwargs))
        return object()

    class DummyEvaluator:
        def dump_gt(self, _):
            return None

    monkeypatch.setattr(wwb, "parse_args", lambda: args)
    monkeypatch.setattr(wwb, "load_model", fake_load_model)
    monkeypatch.setattr(wwb, "create_evaluator", lambda *_: DummyEvaluator())

    wwb.main()

    assert len(calls) == 1
    load_args, load_kwargs = calls[0]
    assert load_args[6] is True
    assert load_kwargs["llamacpp_n_ctx"] is None


def test_text_base_model_load_forwards_llamacpp_kwargs_for_loader_sanitization(monkeypatch):
    calls = []

    args = Namespace(
        base_model="dummy_base_model",
        target_model=None,
        tokenizer=None,
        omit_chat_template=False,
        gt_data=None,
        target_data=None,
        model_type="text",
        data_encoder="sentence-transformers/all-mpnet-base-v2",
        dataset=None,
        dataset_field="text",
        split=None,
        output=None,
        num_samples=None,
        verbose=False,
        device="CPU",
        ov_config=None,
        language="en",
        hf=True,
        genai=False,
        cb_config=None,
        llamacpp=True,
        llamacpp_chat=False,
        llamacpp_n_ctx=4096,
        image_size=None,
        num_inference_steps=None,
        seed=42,
        taylorseer_config=None,
        from_onnx=False,
        adapters=None,
        alphas=None,
        long_prompt=False,
        short_prompt=True,
        empty_adapters=False,
        embeds_pooling_type=None,
        embeds_normalize=False,
        embeds_padding_side=None,
        embeds_batch_size=None,
        rag_config=None,
        gguf_file=None,
        draft_model=None,
        draft_device=None,
        draft_cb_config=None,
        num_assistant_tokens=None,
        assistant_confidence_threshold=None,
        video_frames_num=None,
        speaker_embeddings=None,
        speech_language="",
        speech_voice="",
        tts_eval_whisper_model="base.en",
        vocoder_path=None,
        pruning_ratio=None,
        relevance_weight=None,
        max_new_tokens=128,
    )

    def fake_load_model(*load_args, **load_kwargs):
        calls.append((load_args, load_kwargs))
        return object()

    class DummyEvaluator:
        def dump_gt(self, _):
            return None

    monkeypatch.setattr(wwb, "parse_args", lambda: args)
    monkeypatch.setattr(wwb, "load_model", fake_load_model)
    monkeypatch.setattr(wwb, "create_evaluator", lambda *_: DummyEvaluator())

    wwb.main()

    assert len(calls) == 1
    _, load_kwargs = calls[0]
    assert load_kwargs["llamacpp_n_ctx"] == 4096


def test_loader_strips_llamacpp_n_ctx_for_hf_text_backend(monkeypatch):
    captured = {}

    def fake_load_text_hf_pipeline(model_id, device, **kwargs):
        captured["model_id"] = model_id
        captured["device"] = device
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(model_loaders, "load_text_hf_pipeline", fake_load_text_hf_pipeline)

    model_loaders.load_model(
        "text",
        "dummy_model",
        "CPU",
        None,
        True,
        False,
        True,
        llamacpp_n_ctx=4096,
    )

    assert captured["kwargs"].get("llamacpp_n_ctx") is None


def test_loader_sets_llamacpp_n_ctx_default_for_llamacpp_backend(monkeypatch):
    captured = {}

    def fake_load_text_llamacpp_pipeline(model_dir, **kwargs):
        captured["model_dir"] = model_dir
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(model_loaders, "load_text_llamacpp_pipeline", fake_load_text_llamacpp_pipeline)

    model_loaders.load_model(
        "text",
        "dummy_model",
        "CPU",
        None,
        False,
        False,
        True,
        llamacpp_n_ctx=None,
    )

    assert captured["kwargs"]["llamacpp_n_ctx"] == 8192


def _get_tiny_llamacpp_model() -> Path:
    repo_id = "afrideva/Tinystories-gpt-0.1-3m-GGUF"
    gguf_file = "tinystories-gpt-0.1-3m.Q2_K.gguf"
    cache_dir = get_ov_cache_dir() / "test_data" / "wwb_tinystories_gpt_0_1_3m_gguf"
    model_dir = download_hf_files_to_cache(repo_id, cache_dir, [gguf_file])
    return model_dir / gguf_file


def test_text_llamacpp_real_gguf_logs_backend_and_n_ctx(tmp_path):
    pytest.importorskip("llama_cpp")

    gguf_path = _get_tiny_llamacpp_model()
    gt_data = tmp_path / "llamacpp_gt.csv"
    n_ctx = 2048

    output = run_wwb(
        [
            "--base-model",
            str(gguf_path),
            "--llamacpp",
            "--llamacpp-n-ctx",
            str(n_ctx),
            "--model-type",
            "text",
            "--gt-data",
            str(gt_data),
            "--num-samples",
            "1",
            "--max_new_tokens",
            "8",
        ]
    )

    assert "Using llama.cpp API" in output
    assert f"Using llama.cpp API (n_ctx={n_ctx})" in output
    assert gt_data.exists()
