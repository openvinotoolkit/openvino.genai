# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model export for tests: genai-cli for the models it converts natively, optimum-cli for all others."""

import logging
import subprocess  # nosec B404
import sys
from functools import cache
from pathlib import Path

from openvino_genai._cli.main import native_exporter_for_model

logger = logging.getLogger(__name__)


@cache
def native_genai_cli_model_type(model: str, library: str | None = None) -> str | None:
    """The model type when genai-cli exports ``model`` (Hub id or local path) with a native exporter, else None."""
    model_type, exporter = native_exporter_for_model(str(model), library)
    return model_type if exporter is not None else None


def genai_cli_command(model: str, output: Path | str, options: list[str] | tuple[str, ...] = ()) -> list[str]:
    return [sys.executable, "-m", "openvino_genai._cli", "export", "--model", str(model), *options, str(output)]


def export_command(model: str, output: Path | str, options: list[str] | tuple[str, ...] = ()) -> list[str]:
    """``optimum-cli export openvino`` command for ``model``, or the equivalent genai-cli command when genai-cli
    converts the model natively. ``options`` are optimum-cli export options, genai-cli accepts the same ones."""
    options = list(options)
    library = options[options.index("--library") + 1] if "--library" in options else None
    if native_genai_cli_model_type(model, library) is not None:
        return genai_cli_command(model, output, options)
    return ["optimum-cli", "export", "openvino", "--model", str(model), *options, str(output)]


def export_with_genai_cli(model: str, output: Path | str, options: list[str] | tuple[str, ...] = ()) -> None:
    """Exports a model with genai-cli; raises CalledProcessError with the tool output on failure."""
    command = genai_cli_command(model, output, options)
    logger.info(f"Conversion command: {' '.join(command)}")
    result = subprocess.run(
        command, check=False, text=True, encoding="utf-8", stderr=subprocess.STDOUT, stdout=subprocess.PIPE
    )
    if result.returncode != 0:
        logger.error(f"genai-cli returned {result.returncode}. Output:\n{result.stdout}")
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
