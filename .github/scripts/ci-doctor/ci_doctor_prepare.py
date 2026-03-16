# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from zipfile import ZipFile
import tempfile

import requests
from github.WorkflowRun import WorkflowRun
from requests.packages.urllib3.util.retry import Retry
import argparse
from requests.adapters import HTTPAdapter
from github import Github, Auth

import os
import re
import logging


def init_logger():
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=LOGLEVEL, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d-%Y %H:%M:%S"
    )


init_logger()

LOGGER = logging.getLogger("ci-doctor-preanalysis")

CI_DOCTOR_DIR = Path("/tmp/ci-doctor/")


def get_arguments() -> argparse.Namespace:
    def repository_name(value: str) -> str:
        if not re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", value):
            raise argparse.ArgumentTypeError(f"Invalid format (expected 'owner/name'): {value}")
        return value

    def run_id(value: str) -> int:
        if not re.match(r"^[0-9]+$", value):
            raise argparse.ArgumentTypeError(f"Run ID must be a positive integer: {value}")
        return int(value)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--repository-name",
        type=repository_name,
        required=True,
        help="Repository name in the OWNER/REPOSITORY format",
    )
    parser.add_argument("--run-id", type=run_id, required=True, help="Workflow Run ID")
    return parser.parse_args()


def collect_logs_for_run(run: WorkflowRun, logs_dir: Path, GITHUB_TOKEN: str, session: requests.Session):
    """
    Downloads logs of a given Workflow Run,
    saves them to a specified path, and returns that path.

    We don't need successful job logs, so we remove them.
    We could've just downloaded logs for failed jobs only,
    but when you download all logs from a workflow run,
    GitHub includes "system.txt" files for each job, which can also
    contain errors on which we might want to trigger rerun.

    Example log archive structure:
    .
    ├── 10_Pytorch Layer Tests _ PyTorch Layer Tests.txt
    ├── 11_CPU functional tests _ CPU functional tests.txt
    ├── 12_C++ unit tests _ C++ unit tests.txt
    ├── 13_OpenVINO tokenizers extension _ OpenVINO tokenizers extension.txt
    ├── C++ unit tests _ C++ unit tests
    │   └── system.txt
    ├── CPU functional tests _ CPU functional tests
    │   └── system.txt
    ├── OpenVINO tokenizers extension _ OpenVINO tokenizers extension
    │   └── system.txt
    ├── Pytorch Layer Tests _ PyTorch Layer Tests
        └── system.txt

    Sometimes though, directories contain log files for each individual step,
    IN ADDITION to the full log in root of the directory:
    .
    ├── 1_Build.txt
    └── Build
        ├── 13_Upload build logs.txt
        ├── 1_Set up job.txt
        ├── 24_Post Clone vcpkg.txt
        ├── 25_Post Clone OpenVINO.txt
        ├── 26_Stop containers.txt
        ├── 27_Complete job.txt
        ├── 2_Initialize containers.txt
        ├── 3_Clone OpenVINO.txt
        ├── 4_Get VCPKG version and put it into GitHub ENV.txt
        ├── 5_Init submodules for non vcpkg dependencies.txt
        ├── 6_Clone vcpkg.txt
        ├── 7_System info.txt
        ├── 8_Build vcpkg.txt
        ├── 9_CMake - configure.txt
        └── system.txt

    In that case, we need only 'system.txt' file from each directory
    """
    # Get failed jobs
    failed_jobs = [job for job in run.jobs() if job.conclusion == "failure"]
    LOGGER.info(f"FAILED JOBS: {[job.name for job in failed_jobs]}")

    with tempfile.NamedTemporaryFile(suffix=".zip") as temp_file:
        log_archive_path = Path(temp_file.name)

        # Download logs archive
        with open(file=log_archive_path, mode="wb") as log_archive:
            LOGGER.info(f"DOWNLOADING LOGS FOR RUN ID {run.id}")
            # PyGitHub does not expose the "/repos/{owner}/{repo}/actions/runs/{run_id}/logs" endpoint so we have to use requests
            LOGGER.debug(f"Downloading logs from {run.logs_url}")
            response = session.get(url=run.logs_url, headers={"Authorization": f"Bearer {GITHUB_TOKEN}"})
            response.raise_for_status()
            log_archive.write(response.content)

        # Unpack it
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_temp_dir = Path(temp_dir)

            with ZipFile(file=log_archive_path, mode="r") as zip_file:
                zip_file.extractall(logs_temp_dir)

            # Traverse the unpacked logs to find the ones of failed jobs
            for job in failed_jobs:
                job_filename = job.name.replace("/", "_")
                LOGGER.debug(f"Looking for failed job logs with filename: {job_filename}")

                for p in logs_temp_dir.iterdir():
                    # Move failed jobs' logs to the final destination
                    if p.is_dir() and p.name == job_filename:
                        LOGGER.debug(f"Keeping system.txt from directory {p} for failed job {job.name}")
                        (p / "system.txt").rename(logs_dir / f"{job_filename}__system.txt")
                    elif p.is_file() and p.name.endswith(f"{job_filename}.txt"):
                        LOGGER.debug(f"Keeping file {p} for failed job {job.name}")
                        p.rename(logs_dir / p.name)

    LOGGER.info(f"COLLECTED LOGS FOR {run.id} IN {logs_dir}")


# Lines that match ERROR_PATTERN but are known false positives.
NOISE_PATTERN = re.compile(
    r"(-o pipefail|xfail|XFAIL|Defaulting to unsafe serialization|SCCACHE_IGNORE_SERVER_IO_ERROR)",
)

# Case-insensitive pattern matching common CI error indicators.
ERROR_PATTERN = re.compile(
    r"("
    r"\berror[\s:\[)]"
    r"|\bfail(?:ed|ure|ing|s)?\b"
    r"|panic:"
    r"|\bfatal[\s:]"
    r"|\bundefined[\s:]"
    r"|\bexception\b"
    r"|exit status [^0]"
    r")",
    re.IGNORECASE,
)

MAX_HINT_LINES = 30


def extract_hints(logs_dir: Path, hints_dir: Path) -> None:
    """Extracts lines matching ERROR_PATTERN from log files, writes them to separate hint files."""

    for log_file in logs_dir.iterdir():
        if not log_file.is_file() or not log_file.name.endswith(".txt"):
            continue
        hints: list[str] = []
        for lineno, line in enumerate(log_file.open(), start=1):
            if NOISE_PATTERN.search(line) or not ERROR_PATTERN.search(line):
                continue
            hints.append(f"{lineno}:{line.strip()}")
            if len(hints) >= MAX_HINT_LINES:
                break

        hints_file_path = hints_dir / f"{log_file.name}-hints.txt"
        if hints:
            hints_file_path.write_text("\n".join(hints))


def count_lines(file_path: Path) -> int:
    with file_path.open() as f:
        return sum(1 for _ in f)


def write_summary(run: WorkflowRun, logs_dir: Path, hints_dir: Path) -> None:
    """Write a consolidated summary file for the CI Doctor agent."""
    lines: list[str] = [
        "=== Failed Jobs Summary ===",
        f"Run ID: {run.id}",
        "",
    ]

    failed_jobs = [job for job in run.jobs() if job.conclusion == "failure"]
    for job in failed_jobs:
        failed_steps = ", ".join([step.name for step in job.steps if step.conclusion == "failure"])
        lines.append(f"  Job {job.id} {job.name} {job.url}:")
        lines.append(f"    Failed steps: {failed_steps if failed_steps else '(none)'}")

    lines.append("")
    lines.append(f"Downloaded log files ({logs_dir}):")
    for log_file in sorted(logs_dir.glob("*.txt")):
        lines.append(f"  {log_file}")

    lines.append("")
    lines.append(f"Hint files ({hints_dir}):")
    for hints_file in sorted(hints_dir.glob("*-hints.txt")):
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

    SUMMARY_FILE = logs_dir.parent / "summary.txt"
    SUMMARY_FILE.write_text(summary_text)
    print(summary_text)
    print(f"Pre-analysis complete. Agent should start with {SUMMARY_FILE}")


PATTERNS_TO_FILTER_OUT = [
    # 2026-03-13T13:42:55.9786288Z Received 35870 data chunks (chunk size: 16384 bytes), time passed: 30784ms
    re.compile(r"Received \d+ data chunks \(chunk size: \d+ bytes\), time passed: \d+ms"),
]


def filter_logs(job_logs_dir: Path):
    """Remove lines matching patterns in PATTERNS_TO_FILTER_OUT from log files in LOG_DIR."""
    for log_file in job_logs_dir.glob("*.txt"):
        filtered_lines: list[str] = []
        for line in log_file.open():
            if any(pattern.search(line) for pattern in PATTERNS_TO_FILTER_OUT):
                continue
            filtered_lines.append(line.rstrip())

        log_file.write_text("\n".join(filtered_lines) + "\n")


def main():
    args = get_arguments()
    run_id = args.run_id
    repository_name = args.repository_name

    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

    session = requests.Session()
    retry_strategy = Retry(total=5, backoff_factor=3, backoff_jitter=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://github.com", HTTPAdapter(max_retries=retry_strategy))

    github = Github(auth=Auth.Token(token=GITHUB_TOKEN))
    gh_repo = github.get_repo(full_name_or_id=repository_name)
    run = gh_repo.get_workflow_run(id_=run_id)

    if run.conclusion != "failure":
        LOGGER.warning(
            f"Run {run_id} in {repository_name} did not fail (conclusion: {run.conclusion}). No logs will be collected."
        )
        return

    RUN_DIR = CI_DOCTOR_DIR / f"run_{run_id}"
    LOGS_DIR = RUN_DIR / "logs"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    collect_logs_for_run(run=run, logs_dir=LOGS_DIR, GITHUB_TOKEN=GITHUB_TOKEN, session=session)
    filter_logs(job_logs_dir=LOGS_DIR)

    HINTS_DIR = RUN_DIR / "hints"
    HINTS_DIR.mkdir(exist_ok=True, parents=True)

    extract_hints(logs_dir=LOGS_DIR, hints_dir=HINTS_DIR)

    write_summary(run=run, logs_dir=LOGS_DIR, hints_dir=HINTS_DIR)


if __name__ == "__main__":
    main()
