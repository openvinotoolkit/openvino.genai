# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``wwb run`` subcommand (CLI surface).

Tests run ``wwb`` as a subprocess and inspect exit codes / stdout / stderr.
No models are loaded — only ``--help`` and ``--dry-run`` paths are exercised.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 — invoking our own CLI for end-to-end tests
from pathlib import Path

MINIMAL_YAML: str = """
schema_version: 1
name: cli-test
models:
  base:
    path: org/base
    backend: hf
  target:
    path: /ov/target
    backend: genai
datasets:
  ds:
    type: builtin
tasks:
  - id: chat
    type: text
    base: base
    targets: [target]
    dataset: ds
"""


def run_wwb_cmd(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    """Invoke ``wwb`` with the given args and return the CompletedProcess.

    ``check=False`` so callers can assert on non-zero exits without raising.
    """
    base_env = {"PYTHONIOENCODING": "utf-8", **os.environ}
    if env:
        base_env.update(env)
    return subprocess.run(  # nosec B603 — fixed argv, no shell
        ["wwb"] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=base_env,
        check=False,
    )


def _write_yaml(tmp_path: Path, content: str = MINIMAL_YAML, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_wwb_legacy_help_still_works() -> None:
    """Regression guard — adding `run` must not break the legacy CLI surface."""
    result = run_wwb_cmd(["--help"])
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_wwb_run_help() -> None:
    """`wwb run --help` should show help mentioning scenarios."""
    result = run_wwb_cmd(["run", "--help"])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "scenario" in result.stdout.lower()


def test_wwb_run_dry_run(tmp_path: Path) -> None:
    """Dry-run validates the scenario and prints its name + tasks + base path."""
    yaml_path = _write_yaml(tmp_path)
    result = run_wwb_cmd(["run", str(yaml_path), "--dry-run"])

    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    combined = result.stdout
    assert "cli-test" in combined
    assert "chat" in combined
    assert "org/base" in combined


def test_wwb_run_dry_run_prints_matrix(tmp_path: Path) -> None:
    """Dry-run output must enumerate every (task, target) pair."""
    yaml_path = _write_yaml(tmp_path)
    result = run_wwb_cmd(["run", str(yaml_path), "--dry-run"])

    assert result.returncode == 0, f"stderr: {result.stderr}"
    output = result.stdout
    # The minimal scenario has one (task=chat, target=target) pair.
    assert "chat" in output
    assert "target" in output


def test_wwb_run_missing_scenario_file(tmp_path: Path) -> None:
    """Pointing at a nonexistent scenario path must exit non-zero."""
    missing = tmp_path / "does_not_exist.yaml"
    result = run_wwb_cmd(["run", str(missing)])
    assert result.returncode != 0


def test_wwb_run_invalid_yaml(tmp_path: Path) -> None:
    """Malformed YAML must exit non-zero with diagnostic output."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("bad: yaml: [{{", encoding="utf-8")
    result = run_wwb_cmd(["run", str(bad)])

    assert result.returncode != 0
    combined = (result.stdout or "") + (result.stderr or "")
    # Some signal of what failed must reach the user — stay loose on wording
    # so this doesn't tie us to a specific exception message.
    assert combined.strip(), "expected error output, got nothing"


def test_wwb_run_only_valid_task(tmp_path: Path) -> None:
    """`--only chat` should be accepted by dry-run and mention `chat`."""
    yaml_path = _write_yaml(tmp_path)
    result = run_wwb_cmd(["run", str(yaml_path), "--dry-run", "--only", "chat"])

    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert "chat" in result.stdout


def test_wwb_run_only_unknown_task(tmp_path: Path) -> None:
    """`--only no_such_task` must fail loudly."""
    yaml_path = _write_yaml(tmp_path)
    result = run_wwb_cmd(["run", str(yaml_path), "--dry-run", "--only", "no_such_task"])
    assert result.returncode != 0


def test_wwb_run_output_override(tmp_path: Path) -> None:
    """`--output` should be accepted in dry-run mode without error."""
    yaml_path = _write_yaml(tmp_path)
    override_dir = tmp_path / "override_out"
    result = run_wwb_cmd(["run", str(yaml_path), "--dry-run", "--output", str(override_dir)])
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"


def test_legacy_base_model_flag_still_works() -> None:
    """Legacy flags must remain visible in `wwb --help` after the run subcommand lands."""
    result = run_wwb_cmd(["--help"])
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "--base-model" in result.stdout
