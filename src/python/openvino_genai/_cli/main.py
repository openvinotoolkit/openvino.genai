# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""genai-cli: exports Hugging Face models to OpenVINO IRs for OpenVINO GenAI.

Usage (the export options are the ones of ``optimum-cli export openvino``)::

    genai-cli export -m <model id or path> [options] <output dir>

Models listed in ``openvino_genai._cli.exporters`` are converted natively with torch.export; every other model is
exported through optimum-intel. In both cases the model is exported with the transformers version it is supported
with, provided by an ephemeral uv overlay when the installed version differs.
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


def _read_model_config(model: str, cache_dir: str | None) -> dict | None:
    """Reads config.json of a local or Hub model as plain JSON (no remote code is run)."""
    path = Path(model)
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.is_file():
            return None
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

        try:
            config_path = Path(hf_hub_download(model, "config.json", cache_dir=cache_dir))
        except (EntryNotFoundError, HfHubHTTPError, OSError, ValueError) as error:
            # No config.json (e.g. a diffusers repository), offline or not a valid repository id: let
            # optimum_fallback.export_with_optimum_intel handle the model and report the problem.
            logger.debug(f"Cannot download config.json of {model}: {error}")
            return None
    return json.loads(config_path.read_text(encoding="utf-8"))


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
    requirements = list(exporter_cls.EXTRA_REQUIREMENTS)
    if exporter_cls.TRANSFORMERS_VERSION is not None:
        requirements.insert(0, f"transformers=={exporter_cls.TRANSFORMERS_VERSION}")
    unsatisfied = [requirement for requirement in requirements if not is_satisfied(requirement)]
    if unsatisfied:
        logger.info(
            f"`{model_type}` is exported with {', '.join(requirements)}, "
            f"transformers {installed_version('transformers')} is installed."
        )
        return reexec_with(requirements, argv)

    logger.info(
        f"Exporting `{model_type}` with the native genai-cli exporter (transformers {installed_version('transformers')})"
    )
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
