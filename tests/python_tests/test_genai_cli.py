# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests of the genai-cli export tool (openvino_genai._cli)."""

import json
import os
import shutil
import subprocess  # nosec B404
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from openvino_genai._cli import main as cli_main
from openvino_genai._cli import optimum_fallback, uv_env
from openvino_genai._cli.exporters.common import copy_model_files, weight_compression_config_from_args


def _checkpoint(tmp_path: Path, model_type: str) -> Path:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    return model_dir


def _export_args(*argv: str):
    return cli_main._build_parser().parse_args(["export", *argv])


def test_fallback_to_optimum_intel_for_non_native_models(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(optimum_fallback, "export_with_optimum_intel", lambda *args: calls.append(args) or 0)
    model_dir = _checkpoint(tmp_path, "qwen2")
    argv = ["export", "-m", str(model_dir), "--weight-format", "int4", str(tmp_path / "out")]

    assert cli_main.main(argv) == 0
    model_type, task, export_argv, full_argv = calls[0]
    assert (model_type, task) == ("qwen2", "auto")
    # Everything after `export` is forwarded to `optimum-cli export openvino` unchanged.
    assert export_argv == argv[1:]
    assert full_argv == argv


def test_native_export_reexecutes_with_pinned_transformers(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(uv_env, "is_satisfied", lambda requirement: False)
    monkeypatch.setattr(uv_env, "reexec_with", lambda requirements, argv: calls.append((requirements, argv)) or 0)
    model_dir = _checkpoint(tmp_path, "molmo2")
    argv = ["export", "-m", str(model_dir), "--trust-remote-code", str(tmp_path / "out")]

    assert cli_main.main(argv) == 0
    assert calls == [(["transformers==4.57.6"], argv)]


@pytest.mark.parametrize(
    "options, message",
    [
        ([], "--trust-remote-code"),
        (["--trust-remote-code", "--disable-stateful"], "--disable-stateful"),
        (["--trust-remote-code", "--model-kwargs", '{"a": 1}'], "--model-kwargs"),
        (["--trust-remote-code", "--pad-token-id", "0"], "--pad-token-id"),
        (["--trust-remote-code", "--quant-mode", "int8"], "--quant-mode"),
        (["--trust-remote-code", "--weight-format", "int4", "--dataset", "contextual"], "--dataset"),
        (["--trust-remote-code", "--ratio", "0.8"], "--weight-format"),
        (["--trust-remote-code", "--task", "text-generation"], "text-generation"),
    ],
)
def test_native_exporter_rejects_unsupported_options(tmp_path, monkeypatch, capsys, options, message):
    monkeypatch.setattr(uv_env, "reexec_with", lambda *args: pytest.fail("must fail before re-executing"))
    model_dir = _checkpoint(tmp_path, "molmo2")
    with pytest.raises(SystemExit) as error:
        cli_main.main(["export", "-m", str(model_dir), *options, str(tmp_path / "out")])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_reexec_command_overlays_requirements(monkeypatch):
    captured = {}

    def fake_call(cmd, env):
        captured.update(cmd=cmd, env=env)
        return 0

    monkeypatch.setattr(uv_env, "find_uv", lambda: "uv")
    monkeypatch.setattr(uv_env.subprocess, "call", fake_call)
    monkeypatch.delenv("GENAI_CLI_UV_OVERLAY", raising=False)
    monkeypatch.delenv("GENAI_CLI_BASE_PREFIX", raising=False)

    assert uv_env.reexec_with(["transformers==4.57.6"], ["export", "-m", "model", "out"]) == 0
    cmd, env = captured["cmd"], captured["env"]
    assert cmd[:2] == ["uv", "run"] and "--no-project" in cmd
    assert cmd[cmd.index("--with") + 1] == "transformers==4.57.6"
    assert cmd[-7:] == ["python", "-m", "openvino_genai._cli", "export", "-m", "model", "out"]
    assert json.loads(env["GENAI_CLI_UV_OVERLAY"]) == ["transformers==4.57.6"]


def test_reexec_replaces_overlaid_requirement_of_the_same_distribution(monkeypatch):
    captured = {}
    monkeypatch.setattr(uv_env, "find_uv", lambda: "uv")
    monkeypatch.setattr(uv_env.subprocess, "call", lambda cmd, env: captured.update(cmd=cmd) or 0)
    monkeypatch.setenv("GENAI_CLI_UV_OVERLAY", json.dumps(["transformers==5.5.4", "optimum==2.3.0"]))

    uv_env.reexec_with(["transformers==4.57.6"], ["export"])
    overlay = [captured["cmd"][i + 1] for i, arg in enumerate(captured["cmd"]) if arg == "--with"]
    assert sorted(overlay) == ["optimum==2.3.0", "transformers==4.57.6"]

    monkeypatch.setenv("GENAI_CLI_UV_OVERLAY", json.dumps(overlay))
    with pytest.raises(RuntimeError, match="still not satisfied"):
        uv_env.reexec_with(["transformers==4.57.6"], ["export"])


def test_copy_model_files_skips_weights_and_in_place_export(tmp_path):
    model_dir = _checkpoint(tmp_path, "molmo2")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "tokenizer.json").write_text("{}")

    output = tmp_path / "out"
    output.mkdir()
    copy_model_files(model_dir, output)
    assert sorted(path.name for path in output.iterdir()) == ["config.json", "tokenizer.json"]

    # Exporting into the checkpoint directory must not delete the checkpoint files.
    copy_model_files(model_dir, model_dir)
    assert json.loads((model_dir / "config.json").read_text()) == {"model_type": "molmo2"}
    assert (model_dir / "tokenizer.json").exists()


def test_weight_compression_defaults():
    assert weight_compression_config_from_args(_export_args("-m", "m", "out")) is None
    assert weight_compression_config_from_args(_export_args("-m", "m", "--weight-format", "fp16", "out")) is None

    int8 = weight_compression_config_from_args(_export_args("-m", "m", "--weight-format", "int8", "out"))
    assert (int8.mode, int8.group_size) == ("int8_asym", -1)

    int4 = weight_compression_config_from_args(_export_args("-m", "m", "--weight-format", "int4", "out"))
    assert (int4.mode, int4.group_size, int4.ratio, int4.group_size_fallback) == ("int4_asym", 128, 1.0, "ignore")

    custom = weight_compression_config_from_args(
        _export_args("-m", "m", "--weight-format", "int4", "--sym", "--ratio", "0.8", "--group-size", "64", "out")
    )
    assert (custom.mode, custom.group_size, custom.ratio) == ("int4_sym", 64, 0.8)


_MAKE_TINY_MOLMO2 = textwrap.dedent(
    """
    import shutil, sys
    from pathlib import Path
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForImageTextToText

    src = Path(snapshot_download("allenai/MolmoWeb-4B", allow_patterns=["*.json", "*.py", "*.jinja", "*.txt"]))
    out = Path(sys.argv[1])
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
    t, v, a = cfg.text_config, cfg.vit_config, cfg.adapter_config
    t.hidden_size, t.num_attention_heads, t.num_key_value_heads, t.head_dim = 128, 4, 2, 32
    t.num_hidden_layers, t.intermediate_size = 2, 256
    v.hidden_size, v.intermediate_size, v.num_hidden_layers = 64, 128, 2
    v.num_attention_heads = v.num_key_value_heads = 2
    v.head_dim = 32
    a.hidden_size, a.num_attention_heads, a.num_key_value_heads, a.head_dim = 64, 2, 2, 32
    a.intermediate_size, a.text_hidden_size, a.vit_layers = 128, t.hidden_size, [-1, -2]
    AutoModelForImageTextToText.from_config(cfg, trust_remote_code=True).float().save_pretrained(out)
    for f in src.iterdir():
        if f.is_file() and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)
    """
)


@pytest.mark.nightly
@pytest.mark.vlm
def test_native_molmo2_export_and_inference(tmp_path):
    """Exports a tiny random Molmo2 with genai-cli and runs it in VLMPipeline (paged attention) on text, image and video."""
    import openvino as ov
    import openvino_genai

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required (pip install openvino-genai[export])")
    env = {**os.environ, "VIRTUAL_ENV": sys.prefix}
    tiny = tmp_path / "tiny-random-molmo2"
    script = tmp_path / "make_tiny_molmo2.py"
    script.write_text(_MAKE_TINY_MOLMO2)
    subprocess.run(
        [uv, "run", "--active", "--no-project", "--with", "transformers==4.57.6", "python", str(script), str(tiny)],
        check=True,
        env=env,
    )

    ov_dir = tmp_path / "tiny-random-molmo2-ov"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "openvino_genai._cli",
            "export",
            "-m",
            str(tiny),
            "--trust-remote-code",
            "--weight-format",
            "fp16",
            str(ov_dir),
        ],
        check=True,
        env=env,
    )
    for name in ("language", "text_embeddings", "vision_embeddings"):
        assert (ov_dir / f"openvino_{name}_model.xml").is_file()

    pipe = openvino_genai.VLMPipeline(
        ov_dir, "CPU", scheduler_config=openvino_genai.SchedulerConfig(), ATTENTION_BACKEND="PA"
    )
    config = openvino_genai.GenerationConfig(max_new_tokens=8, do_sample=False)
    rng = np.random.default_rng(0)
    image = ov.Tensor(rng.integers(0, 255, (1, 240, 320, 3), dtype=np.uint8))
    video = ov.Tensor(rng.integers(0, 255, (6, 240, 320, 3), dtype=np.uint8))
    for inputs in ({}, {"images": [image]}, {"videos": [video]}):
        result = pipe.generate("Describe the input.", generation_config=config, **inputs)
        assert len(result.texts) == 1 and result.texts[0]
