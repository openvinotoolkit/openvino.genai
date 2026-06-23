# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404
from argparse import Namespace
from pathlib import Path

import pytest

from conftest import run_wwb
from huggingface_hub import hf_hub_download
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


def test_main_rejects_mutually_exclusive_backend_flags(monkeypatch):
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

    monkeypatch.setattr(wwb, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="Options --hf, --genai and --llamacpp are mutually exclusive"):
        wwb.main()


def test_main_csv_only_rejects_llamacpp_n_ctx_without_llamacpp(monkeypatch, tmp_path):
    gt_data = tmp_path / "gt.csv"
    target_data = tmp_path / "target.csv"
    gt_data.write_text("prompt,answer\nq,a\n", encoding="utf-8")
    target_data.write_text("prompt,answer\nq,a\n", encoding="utf-8")

    args = Namespace(
        base_model=None,
        target_model=None,
        tokenizer=None,
        omit_chat_template=False,
        gt_data=str(gt_data),
        target_data=str(target_data),
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
        llamacpp=False,
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

    monkeypatch.setattr(wwb, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="--llamacpp-n-ctx requires --llamacpp"):
        wwb.main()


def test_loader_raises_for_llamacpp_n_ctx_when_hf_backend_selected():
    with pytest.raises(
        ValueError,
        match="--llamacpp-n-ctx is supported only when llama.cpp is the selected text backend",
    ):
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


def test_loader_sets_llamacpp_n_ctx_default_when_key_omitted(monkeypatch):
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
    )

    assert captured["kwargs"]["llamacpp_n_ctx"] == 8192


def test_check_args_allows_llamacpp_for_text_chat(monkeypatch):
    args = Namespace(
        base_model="dummy_base_model",
        target_model=None,
        tokenizer=None,
        omit_chat_template=False,
        gt_data=None,
        target_data=None,
        model_type="text-chat",
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

    monkeypatch.setattr(wwb, "parse_args", lambda: args)
    monkeypatch.setattr(wwb, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(
        wwb, "create_evaluator", lambda *_: type("DummyEvaluator", (), {"dump_gt": lambda self, _: None})()
    )
    wwb.main()


def _get_tiny_llamacpp_model() -> Path:
    return Path(
        hf_hub_download(
            repo_id="afrideva/Tinystories-gpt-0.1-3m-GGUF",
            filename="tinystories-gpt-0.1-3m.Q2_K.gguf",
        )
    )


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
