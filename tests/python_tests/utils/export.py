# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model export for tests: genai-cli for the models it converts natively, optimum-cli for all others."""

import logging
import shutil
import subprocess  # nosec B404
import sys
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)


@cache
def native_genai_cli_model_type(model: str, library: str | None = None) -> str | None:
    """The model type when genai-cli exports ``model`` (Hub id or local path) with a native exporter, else None."""
    try:
        from openvino_genai._cli.main import native_exporter_for_model
    except ImportError:  # openvino_genai built without genai-cli
        return None
    model_type, exporter = native_exporter_for_model(str(model), library)
    return model_type if exporter is not None else None


def genai_cli_can_provide_transformers() -> bool:
    """genai-cli exports every native model with its pinned transformers version, which needs uv when it is not
    installed (see openvino_genai._cli.uv_env)."""
    try:
        from openvino_genai._cli.uv_env import find_uv

        find_uv()
        return True
    except (ImportError, RuntimeError):
        return shutil.which("uv") is not None


def export_command(model: str, output: Path | str, options: list[str] | tuple[str, ...] = ()) -> list[str]:
    """``optimum-cli export openvino`` command for ``model``, or the equivalent genai-cli command when genai-cli
    converts the model natively. ``options`` are optimum-cli export options, genai-cli accepts the same ones."""
    options = list(options)
    library = options[options.index("--library") + 1] if "--library" in options else None
    if native_genai_cli_model_type(model, library) is not None:
        tool = [sys.executable, "-m", "openvino_genai._cli", "export"]
    else:
        tool = ["optimum-cli", "export", "openvino"]
    return [*tool, "--model", str(model), *options, str(output)]


def export_with_genai_cli(model: str, output: Path | str, options: list[str] | tuple[str, ...] = ()) -> None:
    """Exports a model genai-cli converts natively; raises CalledProcessError with the tool output on failure."""
    command = [sys.executable, "-m", "openvino_genai._cli", "export", "--model", str(model), *options, str(output)]
    logger.info(f"Conversion command: {' '.join(command)}")
    result = subprocess.run(
        command, check=False, text=True, encoding="utf-8", stderr=subprocess.STDOUT, stdout=subprocess.PIPE
    )
    if result.returncode != 0:
        logger.error(f"genai-cli returned {result.returncode}. Output:\n{result.stdout}")
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
