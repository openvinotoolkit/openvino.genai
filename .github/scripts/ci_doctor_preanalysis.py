#!/usr/bin/env python3
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CI Doctor pre-analysis script.

Downloads logs for failed GitHub Actions jobs,
applies error-pattern heuristics, and writes a summary file
for the CI Doctor agent.

Requires the ``gh`` CLI to be installed and authenticated.

Required environment variables:
    GH_TOKEN  — GitHub token for API access
    RUN_ID    — Workflow run ID to investigate
    REPO      — Repository in "owner/repo" format
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

LOG_DIR = Path("/tmp/ci-doctor/logs")
FILTERED_DIR = Path("/tmp/ci-doctor/filtered")
SUMMARY_FILE = Path("/tmp/ci-doctor/summary.txt")

MAX_HINT_LINES = 30
GH_API_TIMEOUT_SEC = 120
GH_DOWNLOAD_TIMEOUT_SEC = 300
MAX_LOG_SIZE_MB = 2
MAX_LOG_SIZE_BYTES = MAX_LOG_SIZE_MB * 1024 * 1024

# Case-insensitive pattern matching common CI error indicators.
ERROR_PATTERN = re.compile(
    r"(error[: ]|fail|panic:|fatal[: ]|undefined[: ]|exception|exit status [^0])",
    re.IGNORECASE,
)


def gh_api_json(endpoint: str) -> dict[str, Any]:
    """Call ``gh api`` and return the parsed JSON response."""
    result = subprocess.run(
        ["gh", "api", endpoint],
        capture_output=True,
        text=True,
        check=True,
        timeout=GH_API_TIMEOUT_SEC,
    )
    return json.loads(result.stdout)


def gh_api_download(endpoint: str, dest: Path) -> None:
    """Call ``gh api`` and stream output to *dest*.

    Truncates the download at MAX_LOG_SIZE_BYTES to prevent
    excessively large files from consuming disk space.
    """
    proc = subprocess.Popen(
        ["gh", "api", endpoint],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        written = 0
        with dest.open("wb") as fout:
            while chunk := proc.stdout.read(64 * 1024):
                remaining = MAX_LOG_SIZE_BYTES - written
                if remaining <= 0:
                    print(f"  -> Truncated at {MAX_LOG_SIZE_MB} MB")
                    break
                fout.write(chunk[:remaining])
                written += len(chunk[:remaining])
        proc.wait(timeout=GH_API_TIMEOUT_SEC)
        if proc.returncode:
            stderr_output = proc.stderr.read().decode(errors="replace")
            raise subprocess.CalledProcessError(proc.returncode, proc.args, stderr=stderr_output)
    except Exception:
        proc.kill()
        proc.wait()
        raise


def count_lines(path: Path) -> int:
    """Return the number of lines in a text file, 0 on error."""
    try:
        return sum(1 for _ in path.open(errors="replace"))
    except OSError:
        return 0


def extract_hints(source: Path, hints_path: Path) -> list[str]:
    """Grep *source* for error-like lines, write up to MAX_HINT_LINES to *hints_path*.

    Returns the list of hint lines.
    """
    hints: list[str] = []
    try:
        for lineno, line in enumerate(source.open(), start=1):
            if ERROR_PATTERN.search(line):
                hints.append(f"{lineno}:{line}")
                if len(hints) >= MAX_HINT_LINES:
                    break
    except OSError:
        return []

    if hints:
        hints_path.write_text("\n".join(hints))
    return hints


def fetch_failed_jobs(repo: str, run_id: str) -> list[dict]:
    """Return a list of failed/cancelled jobs with their failed step names."""
    data = gh_api_json(f"repos/{repo}/actions/runs/{run_id}/jobs")

    failed_jobs: list[dict] = []
    for job in data.get("jobs", []):
        if job.get("conclusion") not in ("failed", "cancelled"):
            continue
        failed_steps = [step["name"] for step in job.get("steps", []) if step.get("conclusion") == "failed"]
        failed_jobs.append({"id": job["id"], "name": job["name"], "failed_steps": failed_steps})
    return failed_jobs


def download_job_logs(repo: str, failed_jobs: list[dict]) -> dict[int, list[str]]:
    """Download logs for each failed job and extract error hints."""
    job_hints: dict[int, list[str]] = {}
    for job in failed_jobs:
        job_id = job["id"]
        log_file = LOG_DIR / f"job-{job_id}.log"
        print(f"Downloading log for job {job_id}...")

        try:
            gh_api_download(f"repos/{repo}/actions/jobs/{job_id}/logs", log_file)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log_file.write_text("(log download failed)")
            print(f"  -> Log download failed: {exc}")
            continue

        lines = count_lines(log_file)
        print(f"  -> Saved {lines} lines to {log_file}")

        hints_file = FILTERED_DIR / f"job-{job_id}-hints.txt"
        hints = extract_hints(log_file, hints_file)
        job_hints[job_id] = hints
        if hints:
            print(f"  -> Pre-located {len(hints)} hint line(s) in {hints_file}")
        else:
            print(f"  -> No error hints found in {log_file}")
    return job_hints


def write_summary(run_id: str, failed_jobs: list[dict]) -> None:
    """Write a consolidated summary file for the CI Doctor agent."""
    lines: list[str] = [
        "=== CI Doctor Pre-Analysis ===",
        f"Run ID: {run_id}",
        "",
        f"Failed jobs (details in {LOG_DIR / 'failed-jobs.json'}):",
    ]

    for job in failed_jobs:
        steps = ", ".join(job["failed_steps"]) if job["failed_steps"] else "(none)"
        lines.append(f"  Job {job['id']}: {job['name']}")
        lines.append(f"    Failed steps: {steps}")

    lines.append("")
    lines.append(f"Downloaded log files ({LOG_DIR}):")
    for log_file in sorted(LOG_DIR.glob("job-*.log")):
        lines.append(f"  {log_file} ({count_lines(log_file)} lines)")

    lines.append("")
    lines.append(f"Filtered hint files ({FILTERED_DIR}):")
    for hints_file in sorted(FILTERED_DIR.glob("*-hints.txt")):
        if not hints_file.stat().st_size:
            continue
        hint_count = count_lines(hints_file)
        lines.append(f"  {hints_file} ({hint_count} matches)")
        # Show first 3 hint lines as preview.
        try:
            for i, line in enumerate(hints_file.open()):
                if i >= 3:
                    break
                lines.append(f"    {line.rstrip()}")
        except OSError:
            pass

    summary_text = "\n".join(lines) + "\n"
    SUMMARY_FILE.write_text(summary_text)
    print(summary_text)
    print(f"Pre-analysis complete. Agent should start with {SUMMARY_FILE}")


def _validate_env() -> tuple[str, str]:
    """Read and validate required environment variables.

    Returns (repo, run_id) on success; exits with an error message otherwise.
    """
    repo = os.environ.get("REPO", "")
    run_id = os.environ.get("RUN_ID", "")

    if not repo or not run_id:
        raise ValueError("REPO and RUN_ID environment variables are required.")

    REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

    if not REPO_PATTERN.match(repo):
        raise ValueError(f"REPO has invalid format (expected 'owner/name'): {repo}")

    RUN_ID_PATTERN = re.compile(r"^[0-9]+$")

    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"RUN_ID must be numeric: {run_id}")

    if not os.environ.get("GH_TOKEN"):
        print("Warning: GH_TOKEN is not set; gh CLI may fail to authenticate.", file=sys.stderr)

    return repo, run_id


def main() -> None:
    repo, run_id = _validate_env()

    for directory in (LOG_DIR, FILTERED_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"=== CI Doctor: Pre-downloading logs for run {run_id} ===")

    failed_jobs = fetch_failed_jobs(repo, run_id)
    failed_jobs_path = LOG_DIR / "failed-jobs.json"
    failed_jobs_path.write_text(json.dumps(failed_jobs, indent=2))

    print(f"Found {len(failed_jobs)} failed job(s)")
    if not failed_jobs:
        print("No failed jobs found, skipping log download")
        return

    print("Failed jobs:")
    print(failed_jobs)

    download_job_logs(repo, failed_jobs)
    write_summary(run_id, failed_jobs)


if __name__ == "__main__":
    main()
