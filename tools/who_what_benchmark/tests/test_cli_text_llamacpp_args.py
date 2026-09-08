# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess  # nosec B404
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from conftest import run_wwb
from huggingface_hub import hf_hub_download
from whowhatbench import model_loaders
from whowhatbench.wwb import llamacpp_gen_text


pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="llama.cpp tests run on Linux only")


def test_llamacpp_raw_completion_preserves_gguf_prompt_tokens():
    model = Mock()
    # Hunyuan has a separator token, but its tokenizer does not add it to raw prompts.
    prompt_tokens = [18166, 316, 7991, 107918, 30]
    model.tokenize.return_value = prompt_tokens
    model.return_value = {"choices": [{"text": "answer"}]}

    assert llamacpp_gen_text(model, None, "Who is Mark Twain?", 32, True) == "answer"

    model.tokenize.assert_called_once_with(
        b"Who is Mark Twain?", add_bos=True, special=True
    )
    model.assert_called_once_with(
        prompt_tokens, max_tokens=32, echo=False, temperature=0.0
    )
    model.create_chat_completion.assert_not_called()


def test_llamacpp_chat_completion_uses_chat_template():
    model = Mock()
    model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "answer"}}]
    }

    assert (
        llamacpp_gen_text(
            model, None, "Who is Mark Twain?", 32, True, use_chat_template=True
        )
        == "answer"
    )

    model.create_chat_completion.assert_called_once_with(
        messages=[{"role": "user", "content": "Who is Mark Twain?"}],
        max_tokens=32,
        temperature=0.0,
    )
    model.tokenize.assert_not_called()
    model.assert_not_called()


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


def test_cli_rejects_mutually_exclusive_backend_flags():
    output = _assert_wwb_cli_error(
        [
            "--base-model",
            "dummy_base_model",
            "--model-type",
            "text",
            "--llamacpp",
            "--hf",
        ],
        "Options --hf, --genai and --llamacpp are mutually exclusive",
    )

    assert "Options --hf, --genai and --llamacpp are mutually exclusive" in output


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


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"llamacpp_n_ctx": None},
        {},
    ],
    ids=["explicit_none", "key_omitted"],
)
def test_loader_sets_llamacpp_n_ctx_default_for_llamacpp_backend(monkeypatch, extra_kwargs):
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
        **extra_kwargs,
    )

    assert captured["kwargs"]["llamacpp_n_ctx"] == 8192


def test_cli_rejects_llamacpp_for_text_chat(tmp_path):
    gt_data = tmp_path / "gt.csv"
    gt_data.write_text("stub\n", encoding="utf-8")

    _assert_wwb_cli_error(
        [
            "--gt-data",
            str(gt_data),
            "--model-type",
            "text-chat",
            "--llamacpp",
        ],
        "--llamacpp is supported only with --model-type text",
    )


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
