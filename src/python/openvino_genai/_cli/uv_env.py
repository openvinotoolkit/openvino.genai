# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Re-execution of genai-cli inside an ephemeral ``uv`` overlay.

Every model is exported with the transformers version it is known to work with. Instead of
installing that version into the user's environment, genai-cli re-executes itself under::

    uv run --active --no-project --with transformers==<X> python -m openvino_genai._cli ...

uv layers the requested packages in a cached ephemeral environment on top of the active environment,
so the user's virtual environment is never modified.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# JSON list of requirement specifiers already overlaid by a parent genai-cli process.
_OVERLAY_ENV = "GENAI_CLI_UV_OVERLAY"
# sys.prefix of the user's environment that the overlays are layered on.
_BASE_PREFIX_ENV = "GENAI_CLI_BASE_PREFIX"


def installed_version(distribution: str) -> Optional[str]:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def is_satisfied(requirement: str) -> bool:
    """Whether the current environment satisfies a requirement specifier such as ``transformers==4.57.6``."""
    from packaging.requirements import Requirement

    req = Requirement(requirement)
    version = installed_version(req.name)
    return version is not None and req.specifier.contains(version, prereleases=True)


def _normalize(name: str) -> str:
    from packaging.utils import canonicalize_name

    return canonicalize_name(name)


def find_uv() -> str:
    try:
        from uv import find_uv_bin

        return find_uv_bin()
    except (ImportError, FileNotFoundError):
        pass
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "genai-cli needs `uv` to provide the transformers version required by the model. "
            "Install the export dependencies with `pip install openvino-genai[export]`."
        )
    return uv


def cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "openvino_genai" / "genai-cli"


def compile_requirements(
    requirements: List[str], no_deps: bool = False, exclude: Iterable[str] = ()
) -> Dict[str, str]:
    """Resolves requirements with ``uv pip compile`` (metadata only, nothing is installed).

    Distributions in ``exclude`` are dropped from the resolution together with their own dependencies.
    Returns pinned requirements (``name==version`` or ``name @ url``) keyed by normalized distribution name.
    """
    from packaging.requirements import Requirement

    cmd = [find_uv(), "pip", "compile", "-", "--python", sys.executable, "--no-header", "--no-annotate", "--quiet"]
    if no_deps:
        cmd.append("--no-deps")
    with tempfile.TemporaryDirectory() as tmp:
        if exclude:
            overrides = Path(tmp) / "overrides.txt"
            overrides.write_text("".join(f"{name}; sys_platform == 'never'\n" for name in exclude))
            cmd += ["--override", str(overrides)]
        result = subprocess.run(cmd, input="\n".join(requirements), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"uv cannot resolve {requirements}:\n{result.stderr}")
    pins = {}
    for line in result.stdout.splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            pins[_normalize(Requirement(line).name)] = line
    return pins


def pinned_version(requirement: str) -> Optional[str]:
    """``X`` for a ``name==X`` requirement, None otherwise (e.g. for direct URL requirements)."""
    from packaging.requirements import Requirement

    specifiers = list(Requirement(requirement).specifier)
    return specifiers[0].version if len(specifiers) == 1 and specifiers[0].operator == "==" else None


def resolve_pinned_version(requirement: str) -> str:
    """Resolves a requirement range (e.g. ``transformers>=4.51,<=5.2.99``) to ``name==<newest matching release>``."""
    from packaging.requirements import Requirement

    return compile_requirements([requirement], no_deps=True)[_normalize(Requirement(requirement).name)]


def install_to_target(requirements: List[str], target: Path) -> None:
    """Installs exact requirements without their dependencies into ``target`` (a directory for PYTHONPATH)."""
    marker = target / ".genai-cli-complete"
    if marker.is_file():
        return
    cmd = [find_uv(), "pip", "install", "--python", sys.executable, "--target", str(target), "--no-deps", *requirements]
    subprocess.run(cmd, check=True)
    marker.touch()


def reexec_with(requirements: List[str], argv: List[str], pythonpath: Optional[Path] = None) -> int:
    """Runs ``genai-cli <argv>`` again in a uv overlay providing ``requirements``; returns its exit code.

    A requirement replaces an already overlaid requirement on the same distribution.
    """
    from packaging.requirements import Requirement

    overlay = {_normalize(Requirement(req).name): req for req in json.loads(os.environ.get(_OVERLAY_ENV, "[]"))}
    changed = False
    for req in requirements:
        name = _normalize(Requirement(req).name)
        changed |= overlay.get(name) != req
        overlay[name] = req
    if not changed and pythonpath is None:
        raise RuntimeError(
            f"genai-cli is already running in a uv overlay with {list(overlay.values())}, but the requirements "
            f"{requirements} are still not satisfied."
        )

    env = dict(os.environ)
    env[_OVERLAY_ENV] = json.dumps(list(overlay.values()))
    if pythonpath is not None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(pythonpath), env.get("PYTHONPATH")]))
    base_prefix = os.environ.get(_BASE_PREFIX_ENV)
    if base_prefix is None and sys.prefix != sys.base_prefix:
        base_prefix = sys.prefix
    cmd = [find_uv(), "run"]
    if base_prefix is not None:
        # `--active` makes uv layer the overlay on top of the environment genai-cli was started from.
        env[_BASE_PREFIX_ENV] = base_prefix
        env["VIRTUAL_ENV"] = base_prefix
        cmd.append("--active")
    else:
        cmd += ["--python", sys.executable]
    cmd.append("--no-project")
    for requirement in overlay.values():
        cmd += ["--with", requirement]
    cmd += ["python", "-m", "openvino_genai._cli", *argv]

    logger.info(f"Re-running genai-cli with {', '.join(overlay.values())} provided by uv")
    logger.debug(" ".join(cmd))
    return subprocess.call(cmd, env=env)
