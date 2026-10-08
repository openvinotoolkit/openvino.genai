# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export through optimum-intel for the models genai-cli does not convert natively yet."""

import hashlib
import importlib.util
import logging
import os
import sys
from functools import partial
from importlib import metadata
from typing import List, Optional

from openvino_genai._cli.uv_env import (
    cache_dir,
    compile_requirements,
    install_to_target,
    installed_version,
    is_satisfied,
    pinned_version,
    reexec_with,
    resolve_pinned_version,
)

logger = logging.getLogger(__name__)

# Requirement specifier used to provide optimum-intel when it is not installed, e.g.
# "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git" to export with the main branch.
_OPTIMUM_INTEL_SPEC_ENV = "GENAI_CLI_OPTIMUM_INTEL_SPEC"
# Heavy (and ABI-coupled) distributions that always come from the user's environment. uv overlays install the whole
# dependency closure of what they provide, so optimum-intel itself is not overlaid with `uv run --with`.
_ENVIRONMENT_PROVIDED = {"torch", "torchvision", "torchaudio", "openvino", "openvino-tokenizers", "openvino-genai", "nncf", "numpy"}


def _is_optimum_intel_available() -> bool:
    try:
        return importlib.util.find_spec("optimum.intel") is not None and installed_version("optimum-intel") is not None
    except ModuleNotFoundError:
        return False


def _provide_optimum_intel(argv: List[str]) -> int:
    """Re-runs genai-cli with optimum-intel available, leaving the user's environment untouched.

    optimum-intel and optimum (pure Python) are installed without dependencies into a cached directory added to
    PYTHONPATH; the light requirements the environment does not satisfy (transformers, huggingface-hub, ...) are
    overlaid with uv. torch, OpenVINO and NNCF of the environment are used as is.
    """
    spec = os.environ.get(_OPTIMUM_INTEL_SPEC_ENV, "optimum-intel")
    pins = compile_requirements([spec], exclude=_ENVIRONMENT_PROVIDED)
    packages = [pins["optimum-intel"], pins["optimum"]]
    target = cache_dir() / hashlib.sha256("\n".join(packages).encode()).hexdigest()[:16]
    logger.info(f"Providing {', '.join(packages)} from {target}")
    install_to_target(packages, target)

    overlay = [
        requirement
        for name, requirement in pins.items()
        if name not in ("optimum-intel", "optimum") and installed_version(name) != pinned_version(requirement)
    ]
    return reexec_with(overlay, argv, pythonpath=target)


def _transformers_requirement(model_type: Optional[str], task: str) -> Optional[str]:
    """Range of transformers versions optimum-intel supports for ``model_type``, or None if unconstrained."""
    if model_type is None:
        return None
    from optimum.exporters.openvino import model_configs  # noqa: F401 (registers the OpenVINO export configs)
    from optimum.exporters.tasks import TasksManager

    constructors = None
    for candidate in (model_type, model_type.replace("_", "-")):
        try:
            constructors = TasksManager.get_supported_tasks_for_model_type(
                candidate, exporter="openvino", library_name="transformers"
            )
            break
        except KeyError:
            continue
    if not constructors:
        return None
    constructor = constructors.get(task) or next(iter(constructors.values()))
    config_cls = constructor.func if isinstance(constructor, partial) else constructor

    specifiers = []
    min_version = getattr(config_cls, "MIN_TRANSFORMERS_VERSION", None)
    max_version = getattr(config_cls, "MAX_TRANSFORMERS_VERSION", None)
    if min_version is not None:
        specifiers.append(f">={min_version}")
    if max_version is not None:
        specifiers.append(f"<={max_version}")
    if not specifiers:
        return None
    # optimum-intel itself supports a limited transformers range, the pinned version must satisfy it too.
    from packaging.requirements import Requirement

    for requirement in map(Requirement, metadata.requires("optimum-intel") or []):
        if requirement.name == "transformers" and requirement.marker is None:
            specifiers += [str(spec) for spec in requirement.specifier]
    return "transformers" + ",".join(specifiers)


def export_with_optimum_intel(model_type: Optional[str], task: str, export_argv: List[str], argv: List[str]) -> int:
    """Runs ``optimum-cli export openvino <export_argv>`` with the transformers version supported for the model."""
    if not _is_optimum_intel_available():
        logger.info(f"Model type `{model_type}` is not converted natively by genai-cli, falling back to optimum-intel")
        return _provide_optimum_intel(argv)

    transformers_range = _transformers_requirement(model_type, task)
    if transformers_range is not None and not is_satisfied(transformers_range):
        logger.info(
            f"optimum-intel supports `{model_type}` with {transformers_range}, "
            f"transformers {installed_version('transformers')} is active."
        )
        return reexec_with([resolve_pinned_version(transformers_range)], argv)

    logger.info(
        f"Exporting `{model_type}` with optimum-intel {installed_version('optimum-intel')} "
        f"(transformers {installed_version('transformers')})"
    )
    from optimum.commands.optimum_cli import main as optimum_cli_main

    sys.argv = ["optimum-cli", "export", "openvino", *export_argv]
    optimum_cli_main()
    return 0
