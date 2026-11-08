# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests of the genai-cli export tool (openvino_genai._cli)."""

import json
import subprocess  # nosec B404
import sys
from functools import partial
from importlib import metadata
from pathlib import Path

import pytest
from openvino_genai import GenerationConfig, LLMPipeline
from openvino_genai._cli import main as cli_main
from openvino_genai._cli import optimum_fallback, uv_env
from openvino_genai._cli.exporters.common import copy_model_files, weight_compression_config_from_args
from packaging.requirements import Requirement

from utils.export import genai_cli_command
from utils.network import retry_request


def _checkpoint(tmp_path: Path, model_type: str) -> Path:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    return model_dir


def _export_args(*argv: str):
    return cli_main._build_parser().parse_args(["export", *argv])


@pytest.mark.parametrize(
    "weight_format",
    [
        "int4",
        "int2",  # only optimum-intel implements it, it validates the format
    ],
)
def test_fallback_to_optimum_intel_for_non_native_models(tmp_path, monkeypatch, weight_format):
    calls = []
    monkeypatch.setattr(optimum_fallback, "export_with_optimum_intel", lambda *args: calls.append(args) or 0)
    model_dir = _checkpoint(tmp_path, "qwen2")
    argv = ["export", "-m", str(model_dir), "--weight-format", weight_format, str(tmp_path / "out")]

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
        # Formats only optimum-intel implements, and options the selected format would silently ignore.
        (["--trust-remote-code", "--weight-format", "int2"], "--weight-format int2"),
        (["--trust-remote-code", "--weight-format", "fp16", "--ratio", "0.8"], "--ratio"),
        (["--trust-remote-code", "--weight-format", "fp32", "--sym"], "--sym"),
        (["--trust-remote-code", "--weight-format", "int8", "--group-size", "64"], "--group-size"),
        (["--trust-remote-code", "--weight-format", "int8", "--backup-precision", "none"], "--backup-precision"),
        # Formats without a zero point have no symmetric variant.
        (["--trust-remote-code", "--weight-format", "nf4", "--sym"], "--sym"),
        (["--trust-remote-code", "--weight-format", "mxfp4", "--sym"], "--sym"),
        (["--trust-remote-code", "--weight-format", "cb4", "--sym"], "--sym"),
    ],
)
def test_native_exporter_rejects_unsupported_options(tmp_path, monkeypatch, capsys, options, message):
    monkeypatch.setattr(uv_env, "reexec_with", lambda *args: pytest.fail("must fail before re-executing"))
    model_dir = _checkpoint(tmp_path, "molmo2")
    with pytest.raises(SystemExit) as error:
        cli_main.main(["export", "-m", str(model_dir), *options, str(tmp_path / "out")])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "options",
    [
        ["--weight-format", "int8", "--sym", "--group-size", "-1", "--ratio", "1.0"],
        ["--weight-format", "int4", "--ratio", "0.8", "--group-size", "64", "--backup-precision", "none"],
        ["--weight-format", "fp16"],
    ],
)
def test_native_exporter_accepts_options_the_format_implements(tmp_path, monkeypatch, options):
    calls = []
    monkeypatch.setattr(uv_env, "is_satisfied", lambda requirement: False)
    monkeypatch.setattr(uv_env, "reexec_with", lambda requirements, argv: calls.append(argv) or 0)
    model_dir = _checkpoint(tmp_path, "molmo2")
    argv = ["export", "-m", str(model_dir), "--trust-remote-code", *options, "out"]
    assert cli_main.main(argv) == 0
    assert calls == [argv]


def _fake_requires(requirements: dict[str, list[str]]):
    return lambda distribution: requirements.get(distribution, [])


def test_fallback_keeps_optimum_intel_transformers_constraint_without_model_bounds(monkeypatch):
    monkeypatch.setattr(
        optimum_fallback.metadata,
        "requires",
        _fake_requires(
            {
                "optimum-intel": ["transformers<5.6,>=4.45", "torch; extra == 'tests'"],
                "optimum": ["transformers>=4.29"],
            }
        ),
    )
    monkeypatch.setattr(optimum_fallback, "_model_transformers_specifiers", lambda model_type, task: [])
    requirement = optimum_fallback._transformers_requirement(None, "auto")
    assert requirement is not None
    specifier = Requirement(requirement).specifier
    assert specifier.contains("5.5.4") and not specifier.contains("5.6.0") and not specifier.contains("4.44.0")

    monkeypatch.setattr(optimum_fallback, "_model_transformers_specifiers", lambda model_type, task: ["<=4.57.6"])
    specifier = Requirement(optimum_fallback._transformers_requirement("molmo2", "auto")).specifier
    assert specifier.contains("4.57.6") and not specifier.contains("5.0.0")


def test_fallback_resolves_a_new_overlay_when_switching_transformers(monkeypatch):
    """A transformers switch must not keep pins (e.g. huggingface-hub 1.x) of the previous overlay."""
    compiled, captured = [], {}
    monkeypatch.setattr(
        optimum_fallback, "installed_version", lambda name: {"optimum-intel": "2.2.0", "optimum": "2.1.0"}.get(name)
    )

    def fake_compile(requirements, exclude=()):
        compiled.append((requirements, set(exclude)))
        return {
            "optimum-intel": "optimum-intel==2.2.0",
            "optimum": "optimum==2.1.0",
            "transformers": "transformers==4.57.6",
            "huggingface-hub": "huggingface-hub==0.36.2",
        }

    monkeypatch.setattr(optimum_fallback, "compile_requirements", fake_compile)
    overlay = optimum_fallback._overlay_for_transformers("transformers>=4.57,<4.58")
    assert compiled[0][0] == ["optimum-intel==2.2.0", "optimum==2.1.0", "transformers>=4.57,<4.58"]
    assert "torch" in compiled[0][1]
    assert sorted(overlay) == ["huggingface-hub==0.36.2", "transformers==4.57.6"]

    monkeypatch.setattr(uv_env, "find_uv", lambda: "uv")
    monkeypatch.setattr(uv_env.subprocess, "call", lambda cmd, env: captured.update(cmd=cmd, env=env) or 0)
    monkeypatch.setenv("GENAI_CLI_UV_OVERLAY", json.dumps(["transformers==5.15.0", "huggingface-hub==1.20.0"]))
    uv_env.reexec_with(overlay, ["export"], replace_overlay=True)
    assert sorted(json.loads(captured["env"]["GENAI_CLI_UV_OVERLAY"])) == sorted(overlay)
    assert "huggingface-hub==1.20.0" not in captured["cmd"]


def test_test_exports_use_genai_cli_for_native_models(monkeypatch):
    from utils import export

    monkeypatch.setattr(
        export,
        "native_exporter_for_model",
        lambda model, library=None, cache_dir=None: (model, object() if model == "molmo2" else None),
    )
    export.native_genai_cli_model_type.cache_clear()
    try:
        assert export.export_command("molmo2", "out", ["--trust-remote-code"])[:4] == [
            sys.executable,
            "-m",
            "openvino_genai._cli",
            "export",
        ]
        assert export.export_command("qwen2", "out", ["--weight-format", "fp16"]) == [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            "qwen2",
            "--weight-format",
            "fp16",
            "out",
        ]
    finally:
        export.native_genai_cli_model_type.cache_clear()


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


def test_fallback_reads_model_transformers_bounds_from_optimum_intel():
    from optimum.exporters.openvino.model_configs import MiniCPMOpenVINOConfig

    expected = [f">={MiniCPMOpenVINOConfig.MIN_TRANSFORMERS_VERSION}"]
    if getattr(MiniCPMOpenVINOConfig, "MAX_TRANSFORMERS_VERSION", None) is not None:
        expected.append(f"<={MiniCPMOpenVINOConfig.MAX_TRANSFORMERS_VERSION}")
    assert optimum_fallback._model_transformers_specifiers("minicpm", "text-generation-with-past") == expected
    assert optimum_fallback._model_transformers_specifiers("not-a-model-type", "auto") == []


@pytest.mark.parametrize(
    "spec",
    [
        None,
        "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@20390257acb98993f99cdd3bf4df1c983eccadef",
    ],
)
def test_optimum_intel_is_provided_without_changing_the_environment(tmp_path, monkeypatch, spec):
    if spec is None:
        monkeypatch.delenv("GENAI_CLI_OPTIMUM_INTEL_SPEC", raising=False)
    else:
        monkeypatch.setenv("GENAI_CLI_OPTIMUM_INTEL_SPEC", spec)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pins = {
        "optimum-intel": spec or "optimum-intel==2.2.0",
        "optimum": "optimum==2.3.0",
        "transformers": "transformers==4.57.6",
        "packaging": "packaging==24.2",
    }
    compiled, installed, reexecuted = [], [], []
    monkeypatch.setattr(
        optimum_fallback, "compile_requirements", lambda reqs, exclude=(): compiled.append((reqs, set(exclude))) or pins
    )
    monkeypatch.setattr(
        optimum_fallback, "install_to_target", lambda packages, target: installed.append((packages, target))
    )
    monkeypatch.setattr(
        optimum_fallback,
        "reexec_with",
        lambda reqs, argv, pythonpath=None: reexecuted.append((reqs, argv, pythonpath)) or 0,
    )
    monkeypatch.setattr(
        optimum_fallback, "installed_version", lambda name: {"transformers": "5.5.4", "packaging": "24.2"}.get(name)
    )

    assert optimum_fallback._provide_optimum_intel(["export", "-m", "model", "out"]) == 0
    requirements, excluded = compiled[0]
    assert requirements == [spec or "optimum-intel"]
    assert {"torch", "openvino", "nncf"} <= excluded
    packages, target = installed[0]
    # optimum-intel and optimum go to a cached directory without their dependencies.
    assert packages == [pins["optimum-intel"], pins["optimum"]]
    assert target.parent == tmp_path / "openvino_genai" / "genai-cli"
    # Only the requirements the environment does not satisfy are overlaid.
    assert reexecuted == [(["transformers==4.57.6"], ["export", "-m", "model", "out"], target)]


def test_uv_resolves_requirements_without_excluded_distributions():
    pins = uv_env.compile_requirements(["transformers==4.57.6"], exclude=["numpy", "torch"])
    assert pins["transformers"] == "transformers==4.57.6"
    assert "huggingface-hub" in pins
    assert "numpy" not in pins


def test_uv_installs_to_target_once(tmp_path, monkeypatch):
    target = tmp_path / "target"
    uv_env.install_to_target(["packaging==24.2"], target)
    assert metadata.Distribution.at(next(target.glob("packaging-24.2.dist-info"))).version == "24.2"
    # A completed target is reused.
    monkeypatch.setattr(uv_env.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not install again"))
    uv_env.install_to_target(["packaging==24.2"], target)


def test_fallback_export_matches_optimum_cli(tmp_path):
    """A model without a native exporter goes through optimum-intel with the command line unchanged."""
    model_id = "katuni4ka/tiny-random-phi3"
    options = ["--task", "text-generation-with-past", "--weight-format", "fp16"]
    genai_cli_dir, optimum_cli_dir = tmp_path / "genai-cli", tmp_path / "optimum-cli"
    for command in (
        genai_cli_command(model_id, genai_cli_dir, options),
        ["optimum-cli", "export", "openvino", "--model", model_id, *options, str(optimum_cli_dir)],
    ):
        retry_request(partial(subprocess.run, command, check=True))

    config = GenerationConfig(max_new_tokens=8, do_sample=False)
    genai_cli_result = LLMPipeline(genai_cli_dir, "CPU").generate("Hello, my name is", config)
    optimum_cli_result = LLMPipeline(optimum_cli_dir, "CPU").generate("Hello, my name is", config)
    assert genai_cli_result == optimum_cli_result
    assert (genai_cli_dir / "openvino_tokenizer.xml").is_file()
