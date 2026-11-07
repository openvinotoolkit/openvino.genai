# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""genai-cli: exports Hugging Face models to OpenVINO IRs for OpenVINO GenAI.

Usage (the export options are the ones of ``optimum-cli export openvino``)::

    genai-cli export -m <model id or path> [options] <output dir>

Models listed in ``openvino_genai._cli.exporters`` are converted natively with torch.export; every other model is
exported through optimum-intel. In both cases the model is exported with the transformers version it is supported
with, provided by an ephemeral uv overlay when the installed version differs.
"""

import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from openvino_genai._cli.args import add_export_openvino_args

logger = logging.getLogger("openvino_genai._cli")


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
            # No config.json (e.g. a diffusers repository), offline or not a valid repository id: let the fallback
            # (optimum-intel) handle the model and report the problem.
            logger.debug(f"Cannot download config.json of {model}: {error}")
            return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def native_exporter_for_model(model: str, library: str | None = None, cache_dir: str | None = None):
    """``(model_type, exporter class)`` for ``model`` (Hub id or local path); the exporter class is None when genai-cli
    exports the model through optimum-intel."""
    from openvino_genai._cli.exporters import get_native_exporter

    config = _read_model_config(model, cache_dir)
    model_type = config.get("model_type") if config else None
    return model_type, get_native_exporter(model_type, library)


def _export_openvino(parser: ArgumentParser, args, argv: list[str], export_argv: list[str]) -> int:
    from openvino_genai._cli.uv_env import installed_version, is_satisfied, reexec_with

    model_type, exporter_cls = native_exporter_for_model(args.model, args.library, args.cache_dir)
    if exporter_cls is None:
        from openvino_genai._cli.optimum_fallback import export_with_optimum_intel

        return export_with_optimum_intel(model_type, args.task, export_argv, argv)

    try:
        exporter_cls.validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    transformers_requirement = f"transformers=={exporter_cls.TRANSFORMERS_VERSION}"
    if not is_satisfied(transformers_requirement):
        logger.info(
            f"`{model_type}` is exported with {transformers_requirement}, "
            f"transformers {installed_version('transformers')} is installed."
        )
        return reexec_with([transformers_requirement], argv)

    logger.info(
        f"Exporting `{model_type}` with the native genai-cli exporter (transformers {installed_version('transformers')})"
    )
    exporter_cls(args).export()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[genai-cli] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Arguments after `export`, forwarded to `optimum-cli export openvino` as is.
    export_argv = argv[argv.index("export") + 1 :]
    return _export_openvino(parser, args, argv, export_argv)
