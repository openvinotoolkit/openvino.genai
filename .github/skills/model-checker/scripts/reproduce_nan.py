#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the WWB NaN similarity case for 20 references and no predictions.

Prerequisites:
* An openvino.genai checkout with the WWB CLI installed as ``wwb``.
* A WWB text ground-truth CSV with exactly 20 rows and ``prompts``/``answers`` columns.
* The WWB similarity encoder (default: sentence-transformers/all-mpnet-base-v2),
  available locally or downloadable from Hugging Face.

Run from any directory (replace the paths with your local files):

    python /path/to/openvino.genai/.github/skills/model-checker/scripts/reproduce_nan.py \\
        --repo /path/to/openvino.genai \\
        --reference-csv /path/to/reference-20.csv \\
        --work-dir /tmp/model-checker-nan-repro

The supported WWB ``--target-data`` option accepts a precomputed predictions
CSV. This script passes it a header-only CSV (zero predictions), while keeping
all 20 reference rows. WWB should report NaN; the old ``nan < 0.95`` comparison
would pass it, while both current model-checker hooks should reject it. This is
an invalid-input check, not an accuracy measurement. Since this uses precomputed
predictions, it does not independently run either inference backend; it feeds
the same real WWB output through both checker hooks.
"""

import argparse
import csv
import importlib.util
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the openvino.genai checkout containing tools/who_what_benchmark.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        required=True,
        help="WWB text ground-truth CSV with exactly 20 rows and prompts/answers columns.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/model-checker-nan-repro"),
        help="Directory for generated inputs, WWB outputs, and logs (default: %(default)s).",
    )
    parser.add_argument(
        "--data-encoder",
        default="sentence-transformers/all-mpnet-base-v2",
        help="WWB similarity encoder model ID or local path (default: %(default)s).",
    )
    return parser.parse_args()


def _validate_reference_csv(repo: Path, reference_csv: Path) -> None:
    if not (repo / "tools/who_what_benchmark").is_dir():
        raise SystemExit(f"Not an openvino.genai checkout: {repo}")
    if not reference_csv.is_file():
        raise SystemExit(f"Reference CSV does not exist: {reference_csv}")

    with reference_csv.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"prompts", "answers"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise SystemExit("Reference CSV must contain 'prompts' and 'answers' columns.")
        row_count = sum(1 for _ in reader)
    if row_count != 20:
        raise SystemExit(f"Reference CSV must contain exactly 20 data rows; found {row_count}.")


def _load_checker_module():
    checker_path = Path(__file__).resolve().parent / "check_model.py"
    if not checker_path.is_file():
        raise FileNotFoundError(f"Model checker not found at {checker_path}")
    spec = importlib.util.spec_from_file_location("_reproduce_nan_check_model", checker_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model checker from {checker_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = _arguments()
    repo = args.repo.expanduser().resolve()
    reference_csv = args.reference_csv.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    wwb_cli = shutil.which("wwb")

    _validate_reference_csv(repo, reference_csv)
    if wwb_cli is None:
        raise SystemExit("WWB CLI 'wwb' is not installed or not on PATH; install this checkout's WWB dependencies first.")

    checker = _load_checker_module()
    checker_tools = (
        checker.OptimumWWBTargetEvaluationTool,
        checker.GenAIWWBTargetEvaluationTool,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    empty_predictions = work_dir / "zero-predictions.csv"
    with empty_predictions.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerow(("prompts", "answers"))

    logging.basicConfig(level=logging.INFO)
    output_dir = work_dir / "wwb-output"
    command = [
        wwb_cli,
        "--gt-data",
        str(reference_csv),
        "--target-data",
        str(empty_predictions),
        "--model-type",
        "text",
        "--data-encoder",
        args.data_encoder,
        "--output",
        str(output_dir),
    ]

    print("Running real WWB evaluation with 20 references and zero predictions:")
    print(" ".join(command))
    completed = subprocess.run(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode:
        print(f"WWB exited with status {completed.returncode}; no reproduction conclusion.", file=sys.stderr)
        return completed.returncode

    stdout_lines = completed.stdout.splitlines()
    metrics = checker.parse_wwb_metrics_value(stdout_lines)
    if metrics is None or not math.isnan(metrics):
        print(f"Expected a NaN WWB similarity, got {metrics!r}; reproduction did not match.", file=sys.stderr)
        return 1

    print(f"Observed WWB similarity: {metrics!r}")
    print(f"Pre-fix comparison result: {not metrics < 0.95} (True means the old guard would pass)")

    # The same real WWB metric output is checked by both production hooks.
    result = subprocess.CompletedProcess(command, completed.returncode, stdout_lines)
    for tool_class in checker_tools:
        tool = tool_class.__new__(tool_class)
        tool.name = tool_class.__name__
        tool.logger_prefix = f"[{tool.name}]"
        tool.work_dir = work_dir
        hook_result = tool._post_run_hook(result)
        print(f"{tool.name}: success={hook_result.success}; error={hook_result.err_msg}")
        if hook_result.success or "similarity is NaN" not in (hook_result.err_msg or ""):
            print(f"{tool.name} did not reject NaN as expected.", file=sys.stderr)
            return 1

    print("Reproduction confirmed: WWB returns NaN; the old comparison passes it; both current hooks reject it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
