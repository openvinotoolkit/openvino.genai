# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests of the genai-cli export tool (openvino_genai._cli)."""

import json
import os
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


@pytest.mark.parametrize("uv_on_path", [False, True])
def test_missing_export_requirements_are_reported(tmp_path, monkeypatch, capsys, uv_on_path):
    """Without openvino-genai[export], genai-cli export stops with a hint instead of a ModuleNotFoundError."""
    find_spec = cli_main.importlib.util.find_spec
    monkeypatch.setattr(
        cli_main.importlib.util, "find_spec", lambda name: None if name in ("nncf", "uv") else find_spec(name)
    )
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/uv" if uv_on_path else None)
    monkeypatch.setattr(cli_main, "_export_openvino", lambda *args: pytest.fail("must not export"))

    with pytest.raises(SystemExit) as error:
        cli_main.main(["export", "-m", str(_checkpoint(tmp_path, "llama")), str(tmp_path / "out")])
    assert error.value.code == 2
    # A uv executable on PATH is used like the uv package.
    missing = "nncf" if uv_on_path else "nncf, uv"
    assert f"genai-cli export needs {missing}: pip install openvino-genai[export]" in capsys.readouterr().err


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
    model_config, task, export_argv, full_argv = calls[0]
    assert (model_config, task) == ({"model_type": "qwen2"}, "auto")
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
    monkeypatch.setattr(optimum_fallback, "_model_transformers_specifiers", lambda model_config, task: [])
    requirement = optimum_fallback._transformers_requirement(None, "auto")
    assert requirement is not None
    specifier = Requirement(requirement).specifier
    assert specifier.contains("5.5.4") and not specifier.contains("5.6.0") and not specifier.contains("4.44.0")

    monkeypatch.setattr(optimum_fallback, "_model_transformers_specifiers", lambda model_config, task: ["<=4.57.6"])
    specifier = Requirement(optimum_fallback._transformers_requirement({"model_type": "molmo2"}, "auto")).specifier
    assert specifier.contains("4.57.6") and not specifier.contains("5.0.0")


_OPTIMUM_INTEL_GIT = (
    "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@20390257acb98993f99cdd3bf4df1c983eccadef"
)


@pytest.mark.parametrize("optimum_intel", ["optimum-intel==2.2.0", _OPTIMUM_INTEL_GIT])
def test_fallback_resolves_a_new_overlay_when_switching_transformers(monkeypatch, optimum_intel):
    """A transformers switch must not keep pins (e.g. huggingface-hub 1.x) of the previous overlay, and optimum-intel
    installed from Git is resolved from the same commit."""
    compiled, captured = [], {}
    monkeypatch.setattr(
        optimum_fallback,
        "installed_requirement",
        lambda name: {"optimum-intel": optimum_intel, "optimum": "optimum==2.1.0"}.get(name),
    )

    def fake_compile(requirements, exclude=()):
        compiled.append((requirements, set(exclude)))
        return {
            "optimum-intel": optimum_intel,
            "optimum": "optimum==2.1.0",
            "transformers": "transformers==4.57.6",
            "huggingface-hub": "huggingface-hub==0.36.2",
        }

    monkeypatch.setattr(optimum_fallback, "compile_requirements", fake_compile)
    overlay = optimum_fallback._overlay_for_transformers("transformers>=4.57,<4.58")
    assert compiled[0][0] == [optimum_intel, "optimum==2.1.0", "transformers>=4.57,<4.58"]
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


def test_copy_model_files_keeps_new_irs_over_irs_of_a_previous_export(tmp_path):
    model_dir = _checkpoint(tmp_path, "molmo2")
    (model_dir / "openvino_language_model.xml").write_text("<previous export/>")
    (model_dir / "openvino_language_model.bin").write_bytes(b"previous weights")
    output = tmp_path / "out"
    output.mkdir()
    (output / "openvino_language_model.xml").write_text("<new export/>")

    copy_model_files(model_dir, output)
    assert (output / "openvino_language_model.xml").read_text() == "<new export/>"
    assert sorted(path.name for path in output.iterdir()) == ["config.json", "openvino_language_model.xml"]


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
    # The "ignore" fallback belongs to the int4 preset only, customized settings get the NNCF default ("error").
    assert custom.group_size_fallback is None


def test_customized_int4_keeps_the_nncf_group_size_fallback_default(monkeypatch):
    import nncf

    captured = {}
    monkeypatch.setattr(nncf, "compress_weights", lambda model, **kwargs: captured.update(kwargs) or model)
    args = _export_args("-m", "m", "--weight-format", "int4", "--group-size", "64", "out")
    weight_compression_config_from_args(args).compress("model")
    assert "advanced_parameters" not in captured


@pytest.mark.parametrize(
    "options, mode",
    [
        (["--weight-format", "int8"], "INT8_ASYM"),
        (["--weight-format", "int8", "--sym"], "INT8_SYM"),
        (["--weight-format", "int4"], "INT4_ASYM"),
        (["--weight-format", "int4", "--sym"], "INT4_SYM"),
        (["--weight-format", "mxfp4"], "MXFP4"),
        (["--weight-format", "nf4"], "NF4"),
        (["--weight-format", "cb4"], "CB4"),
    ],
)
def test_weight_format_selects_nncf_mode(monkeypatch, options, mode):
    import nncf

    captured = {}
    monkeypatch.setattr(nncf, "compress_weights", lambda model, **kwargs: captured.update(kwargs) or model)
    weight_compression_config_from_args(_export_args("-m", "m", *options, "out")).compress("model")
    assert captured["mode"] == getattr(nncf.CompressWeightsMode, mode)


def test_compression_options_reach_nncf(monkeypatch):
    import nncf
    from nncf.quantization.advanced_parameters import GroupSizeFallbackMode

    captured = {}
    monkeypatch.setattr(nncf, "compress_weights", lambda model, **kwargs: captured.update(kwargs) or model)
    args = _export_args(
        "-m",
        "m",
        "--weight-format",
        "int4",
        "--ratio",
        "0.8",
        "--group-size",
        "64",
        "--all-layers",
        "--awq",
        "--sensitivity-metric",
        "weight_quantization_error",
        "--backup-precision",
        "int8_sym",
        "--group-size-fallback",
        "adjust",
        "out",
    )
    assert weight_compression_config_from_args(args).compress("model") == "model"
    assert captured["ratio"] == 0.8
    assert captured["group_size"] == 64
    assert captured["all_layers"] is True
    assert captured["awq"] is True
    assert captured["sensitivity_metric"] == nncf.SensitivityMetric.WEIGHT_QUANTIZATION_ERROR
    assert captured["backup_mode"] == nncf.BackupMode.INT8_SYM
    assert captured["advanced_parameters"].group_size_fallback_mode == GroupSizeFallbackMode.ADJUST


def _class_transformers_specifiers(config_cls) -> list[str]:
    specifiers = [f">={config_cls.MIN_TRANSFORMERS_VERSION}"]
    if getattr(config_cls, "MAX_TRANSFORMERS_VERSION", None) is not None:
        specifiers.append(f"<={config_cls.MAX_TRANSFORMERS_VERSION}")
    return specifiers


def test_fallback_reads_model_transformers_bounds_from_optimum_intel():
    from optimum.exporters.openvino.model_configs import MiniCPMOpenVINOConfig

    specifiers = optimum_fallback._model_transformers_specifiers({"model_type": "minicpm"}, "text-generation-with-past")
    assert specifiers == _class_transformers_specifiers(MiniCPMOpenVINOConfig)
    assert optimum_fallback._model_transformers_specifiers({"model_type": "not-a-model-type"}, "auto") == []
    assert optimum_fallback._model_transformers_specifiers(None, "auto") == []


def test_fallback_reads_checkpoint_specific_transformers_bounds():
    """optimum-intel narrows the bounds per checkpoint: Eagle3 drafts of VLMs need transformers<=4.57.6."""
    from optimum.exporters.openvino.model_configs import LlamaOpenVINOConfig

    llama = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    eagle3_vlm = {"model_type": "llama", "architectures": ["LlamaForCausalLMEagle3"], "target_model_type": "qwen3_vl"}
    task = "text-generation-with-past"
    assert optimum_fallback._model_transformers_specifiers(llama, task) == _class_transformers_specifiers(
        LlamaOpenVINOConfig
    )
    assert "<=4.57.6" in optimum_fallback._model_transformers_specifiers(eagle3_vlm, task)


def test_installed_requirement_keeps_direct_url_provenance(tmp_path, monkeypatch):
    def distribution(name: str, direct_url: dict | None = None) -> metadata.Distribution:
        dist_info = tmp_path / f"{name}-2.2.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 2.2.0\n")
        if direct_url is not None:
            (dist_info / "direct_url.json").write_text(json.dumps(direct_url))
        return metadata.Distribution.at(dist_info)

    dists = {
        "from-index": distribution("from-index"),
        "from-git": distribution(
            "from-git",
            {"url": "https://github.com/org/repo.git", "vcs_info": {"vcs": "git", "commit_id": "abc123"}},
        ),
        "from-checkout": distribution("from-checkout", {"url": "file:///src/repo", "dir_info": {"editable": True}}),
    }

    def fake_distribution(name):
        if name not in dists:
            raise metadata.PackageNotFoundError(name)
        return dists[name]

    monkeypatch.setattr(uv_env.metadata, "distribution", fake_distribution)
    assert uv_env.installed_requirement("from-index") == "from-index==2.2.0"
    assert uv_env.installed_requirement("from-git") == "from-git @ git+https://github.com/org/repo.git@abc123"
    assert uv_env.installed_requirement("from-checkout") == "from-checkout @ file:///src/repo"
    assert uv_env.installed_requirement("not-installed") is None


def test_uv_compile_keeps_url_fragments(monkeypatch):
    output = (
        "# a comment line\n"
        "packaging==24.2  # trailing comment\n"
        "wheel-pkg @ https://example.com/wheel_pkg-1.0-py3-none-any.whl#sha256=0123abcd\n"
        "vcs-pkg @ git+https://github.com/org/repo.git@abc123#subdirectory=python\n"
    )
    monkeypatch.setattr(uv_env, "find_uv", lambda: "uv")
    monkeypatch.setattr(
        uv_env.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, "")
    )
    assert uv_env.compile_requirements(["anything"]) == {
        "packaging": "packaging==24.2",
        "wheel-pkg": "wheel-pkg @ https://example.com/wheel_pkg-1.0-py3-none-any.whl#sha256=0123abcd",
        "vcs-pkg": "vcs-pkg @ git+https://github.com/org/repo.git@abc123#subdirectory=python",
    }


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


_DEPENDENCY_GIT = "fork-dependency @ git+https://example.com/fork-dependency.git@0123456789abcdef"


@pytest.mark.parametrize("installed, overlaid", [(None, True), (_DEPENDENCY_GIT, False)])
def test_optimum_intel_overlay_keeps_direct_url_dependencies(tmp_path, monkeypatch, installed, overlaid):
    # A direct URL pin has no version to compare: it is overlaid unless installed from the same source.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pins = {"optimum-intel": "optimum-intel==2.2.0", "optimum": "optimum==2.3.0", "fork-dependency": _DEPENDENCY_GIT}
    reexecuted = []
    monkeypatch.setattr(optimum_fallback, "compile_requirements", lambda reqs, exclude=(): pins)
    monkeypatch.setattr(optimum_fallback, "install_to_target", lambda packages, target: None)
    monkeypatch.setattr(
        optimum_fallback, "reexec_with", lambda reqs, argv, pythonpath=None: reexecuted.append(reqs) or 0
    )
    monkeypatch.setattr(optimum_fallback, "installed_version", lambda name: None)
    monkeypatch.setattr(
        optimum_fallback, "installed_requirement", lambda name: installed if name == "fork-dependency" else None
    )
    optimum_fallback._provide_optimum_intel(["export"])
    assert reexecuted == [[_DEPENDENCY_GIT] if overlaid else []]


@pytest.mark.parametrize("distribution_satisfied", [False, True])
def test_fallback_imports_export_configs_only_with_supported_transformers(monkeypatch, distribution_satisfied):
    # optimum-intel's export configs are imported for the checkpoint-specific bounds: with a transformers version
    # optimum-intel does not support, the overlay for its own requirement is created first.
    reexecuted, inspected = [], []
    monkeypatch.setattr(optimum_fallback, "_is_optimum_intel_available", lambda: True)
    monkeypatch.setattr(optimum_fallback, "_distribution_transformers_requirement", lambda: "transformers<5")
    monkeypatch.setattr(
        optimum_fallback,
        "is_satisfied",
        lambda requirement: distribution_satisfied and requirement == "transformers<5",
    )
    monkeypatch.setattr(
        optimum_fallback,
        "_model_transformers_specifiers",
        lambda model_config, task: inspected.append(model_config) or ["<=4.57.6"],
    )
    monkeypatch.setattr(optimum_fallback, "_overlay_for_transformers", lambda requirement: [requirement])
    monkeypatch.setattr(
        optimum_fallback, "reexec_with", lambda reqs, argv, replace_overlay=False: reexecuted.append(reqs) or 0
    )
    assert optimum_fallback.export_with_optimum_intel({"model_type": "llama"}, "auto", [], ["export"]) == 0
    if distribution_satisfied:
        assert inspected == [{"model_type": "llama"}]
        assert reexecuted == [["transformers<5,<=4.57.6"]]
    else:
        assert inspected == []
        assert reexecuted == [["transformers<5"]]


def test_native_exporters_are_found_through_their_registration():
    from openvino_genai._cli.exporters import get_native_exporter
    from openvino_genai._cli.exporters.molmo2 import Molmo2Exporter

    assert get_native_exporter("molmo2", None) is Molmo2Exporter
    assert get_native_exporter("molmo2", "transformers") is Molmo2Exporter
    assert get_native_exporter("molmo2", "diffusers") is None
    assert get_native_exporter("llama", None) is None
    assert get_native_exporter(None, None) is None


def test_register_exporter_rejects_a_second_exporter_for_a_model_type(monkeypatch):
    from openvino_genai._cli.exporters import base

    monkeypatch.setattr(base, "_REGISTERED_EXPORTERS", {})

    @base.register_exporter
    class First(base.NativeExporter):
        MODEL_TYPE = "test-model"

    assert base.register_exporter(First) is First  # registering the same class again is a no-op
    assert base.registered_exporters() == {"test-model": First}

    class Second(base.NativeExporter):
        MODEL_TYPE = "test-model"

    with pytest.raises(ValueError, match="already has the native exporter"):
        base.register_exporter(Second)


def test_molmo2_hub_download_includes_pytorch_weights():
    from huggingface_hub.utils import filter_repo_objects

    from openvino_genai._cli.exporters import molmo2

    files = [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model.bin.index.json",
        "modeling_molmo2.py",
        "chat_template.jinja",
        "merges.txt",
        "model_and_optim/__0_0.distcp",  # training checkpoints of MolmoWeb repositories
        "model_and_optim/.metadata",
        "config.yaml",
    ]
    downloaded = set(filter_repo_objects(files, allow_patterns=molmo2._ALLOW_PATTERNS))
    assert downloaded == set(files) - {"model_and_optim/__0_0.distcp", "model_and_optim/.metadata", "config.yaml"}


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


# Installed as sitecustomize.py for the genai-cli processes of test_fallback_installs_optimum_intel_on_demand. The
# optimum and optimum-intel of the test environment are hidden, as in an environment installed with
# `pip install openvino-genai[export]` only: the `optimum` namespace package is only found in the genai-cli cache.
# The process that runs the export records where its modules come from.
_ENVIRONMENT_PROBE = """
import atexit
import importlib.machinery
import json
import os
import sys

_PROVIDED = os.environ["GENAI_CLI_TEST_PROVIDED"]


class _OnlyProvidedOptimum:
    def find_spec(self, name, path=None, target=None):
        if name != "optimum":
            return None  # the submodules are searched in optimum.__path__
        provided = [entry for entry in sys.path if entry.startswith(_PROVIDED)]
        spec = importlib.machinery.PathFinder.find_spec(name, provided)
        if spec is None:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        # A namespace package path is recomputed from sys.path, keep the provided portions only.
        spec.submodule_search_locations = list(spec.submodule_search_locations)
        return spec


sys.meta_path.insert(0, _OnlyProvidedOptimum())


def _record_origins():
    if "optimum.commands.optimum_cli" not in sys.modules:
        return  # not the process that ran the export
    origins = {}
    for name in ("torch", "openvino", "nncf", "optimum.intel"):
        module = sys.modules.get(name)
        origins[name] = getattr(module, "__file__", None) or next(iter(getattr(module, "__path__", [])), None)
    with open(os.environ["GENAI_CLI_TEST_ORIGINS"], "w", encoding="utf-8") as file:
        json.dump(origins, file)


atexit.register(_record_origins)
"""


def test_fallback_installs_optimum_intel_on_demand(tmp_path):
    """Without optimum-intel, genai-cli installs it (and optimum) without dependencies into its cache and exports with
    them, while torch, OpenVINO and NNCF still come from the environment genai-cli was started from."""
    import huggingface_hub
    import nncf
    import openvino
    import torch

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "sitecustomize.py").write_text(_ENVIRONMENT_PROBE, encoding="utf-8")
    origins_file = tmp_path / "origins.json"
    cache_home = tmp_path / "cache"  # a clean genai-cli cache
    provided = cache_home / "openvino_genai" / "genai-cli"
    uv_cache = subprocess.run([uv_env.find_uv(), "cache", "dir"], capture_output=True, text=True, check=True)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(probe_dir), os.environ.get("PYTHONPATH")])),
        "GENAI_CLI_TEST_ORIGINS": str(origins_file),
        "GENAI_CLI_TEST_PROVIDED": str(provided),
        "XDG_CACHE_HOME": str(cache_home),
        # XDG_CACHE_HOME moves the uv and Hugging Face caches as well, keep them.
        "UV_CACHE_DIR": uv_cache.stdout.strip(),
        "HF_HOME": str(huggingface_hub.constants.HF_HOME),
    }
    env.pop("GENAI_CLI_UV_OVERLAY", None)
    output = tmp_path / "ov"
    command = genai_cli_command(
        "katuni4ka/tiny-random-phi3", output, ["--task", "text-generation-with-past", "--weight-format", "fp16"]
    )
    retry_request(partial(subprocess.run, command, env=env, check=True))

    origins = json.loads(origins_file.read_text(encoding="utf-8"))
    assert Path(origins["optimum.intel"]).is_relative_to(provided)
    for name, module in (("torch", torch), ("openvino", openvino), ("nncf", nncf)):
        assert Path(origins[name]) == Path(module.__file__), name
    assert (output / "openvino_model.xml").is_file()
    assert (output / "openvino_tokenizer.xml").is_file()
    result = LLMPipeline(output, "CPU").generate("Hello, my name is", GenerationConfig(max_new_tokens=4))
    assert isinstance(result, str)
