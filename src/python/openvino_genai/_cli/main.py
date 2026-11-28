# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""genai-cli: exports Hugging Face models to OpenVINO IRs for OpenVINO GenAI.

Usage (the export options are the ones of ``optimum-cli export openvino``)::

    genai-cli export -m <model id or path> [options] <output dir>

genai-cli uses a built-in GenAI exporter when one supports the requested model. Otherwise, it passes the export
command to Optimum Intel, which handles its own supported models.
Some exporters require a specific Transformers version. When the active environment has a different version,
genai-cli restarts the export in a temporary environment managed by uv with the required version installed.
"""

import importlib.util
import json
import logging
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

from openvino_genai._cli.args import add_export_openvino_args

logger = logging.getLogger("openvino_genai._cli")

# The distributions of the openvino-genai[export] extra (pyproject.toml) and the modules they provide.
_EXPORT_REQUIREMENTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "pyyaml": "yaml",
    "nncf": "nncf",
    "packaging": "packaging",
    "uv": "uv",
}


def _missing_export_requirements() -> list[str]:
    """The distributions of the openvino-genai[export] extra that are not installed. Only looks the modules up, so
    the check is cheap and works before any of them is imported."""
    missing = [name for name, module in _EXPORT_REQUIREMENTS.items() if importlib.util.find_spec(module) is None]
    # Like uv_env.find_uv, a uv executable on PATH can be used instead of the uv package.
    if "uv" in missing and shutil.which("uv") is not None:
        missing.remove("uv")
    return missing


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser("genai-cli", description="OpenVINO GenAI command line tool.")
    commands = parser.add_subparsers(dest="command", metavar="{export}")
    commands.required = True
    export_parser = commands.add_parser(
        "export",
        help="Export a Hugging Face model to OpenVINO IR for OpenVINO GenAI pipelines.",
        description="Export a Hugging Face model to OpenVINO IR for OpenVINO GenAI pipelines.",
    )
    add_export_openvino_args(export_parser)
    return parser


# For FunASR checkpoints without config.json, map the model class in config.yaml
# to a model type handled by a built-in GenAI exporter.
_FUNASR_MODEL_TYPES = {"FsmnVADStreaming": "fsmn_vad"}


def _model_file(model: str, filename: str, cache_dir: str | None) -> Path | None:
    """``filename`` of a local or Hub model, None when the model has no such file."""
    path = Path(model)
    if path.is_dir():
        file_path = path / filename
        return file_path if file_path.is_file() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    try:
        return Path(hf_hub_download(model, filename, cache_dir=cache_dir))
    except (EntryNotFoundError, HfHubHTTPError, OSError, ValueError) as error:
        # No such file (e.g. config.json of a diffusers repository), offline or not a valid repository id: let
        # optimum_fallback.export_with_optimum_intel handle the model and report the problem.
        logger.debug(f"Cannot download {filename} of {model}: {error}")
        return None


def _read_model_config(model: str, cache_dir: str | None) -> dict | None:
    """Reads config.json of a local or Hub model as plain JSON (no remote code is run). Without config.json, a FunASR
    model with a native exporter gets ``{"model_type": ...}`` from its config.yaml."""
    config_path = _model_file(model, "config.json", cache_dir)
    if config_path is not None:
        return json.loads(config_path.read_text(encoding="utf-8"))
    return _read_funasr_model_config(model, cache_dir)


def _read_funasr_model_config(model: str, cache_dir: str | None) -> dict | None:
    config_path = _model_file(model, "config.yaml", cache_dir)
    if config_path is None:
        return None
    import yaml

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError) as error:
        logger.debug(f"Cannot parse config.yaml of {model}: {error}")
        return None
    funasr_model = config.get("model") if isinstance(config, dict) else None
    model_type = _FUNASR_MODEL_TYPES.get(funasr_model) if isinstance(funasr_model, str) else None
    return {"model_type": model_type} if model_type is not None else None


def native_exporter_for_model(model: str, library: str | None = None, cache_dir: str | None = None):
    """``(model_type, exporter class)`` for ``model`` (Hub id or local path); the exporter class is None when genai-cli
    exports the model through optimum-intel."""
    return _native_exporter_for_config(_read_model_config(model, cache_dir), library)


def _native_exporter_for_config(config: dict | None, library: str | None):
    from openvino_genai._cli.exporters import get_native_exporter

    model_type = config.get("model_type") if config else None
    return model_type, get_native_exporter(model_type, library)


def _export_openvino(parser: ArgumentParser, args, argv: list[str], export_argv: list[str]) -> int:
    from openvino_genai._cli.uv_env import installed_version, is_satisfied, reexec_with

    config = _read_model_config(args.model, args.cache_dir)
    model_type, exporter_cls = _native_exporter_for_config(config, args.library)
    if exporter_cls is None:
        from openvino_genai._cli.optimum_fallback import export_with_optimum_intel

        return export_with_optimum_intel(config, args.task, export_argv, argv)

    try:
        exporter_cls.validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if exporter_cls.TRANSFORMERS_VERSION is not None:
        transformers_requirement = f"transformers=={exporter_cls.TRANSFORMERS_VERSION}"
        if not is_satisfied(transformers_requirement):
            logger.info(
                f"`{model_type}` is exported with {transformers_requirement}, "
                f"transformers {installed_version('transformers')} is installed."
            )
            return reexec_with([transformers_requirement], argv)

    transformers_note = (
        f" (transformers {installed_version('transformers')})" if exporter_cls.TRANSFORMERS_VERSION is not None else ""
    )
    logger.info(f"Exporting `{model_type}` with the native genai-cli exporter{transformers_note}")
    exporter_cls(args).export()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Without the export extra, genai-cli export would fail later on a generic ModuleNotFoundError.
    if missing := _missing_export_requirements():
        parser.error(f"genai-cli export needs {', '.join(missing)}: pip install openvino-genai[export]")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[genai-cli] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Arguments after `export`, forwarded to `optimum-cli export openvino` as is.
    export_argv = argv[argv.index("export") + 1 :]
    return _export_openvino(parser, args, argv, export_argv)
