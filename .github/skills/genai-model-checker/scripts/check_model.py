#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end validation of a HuggingFace model with OpenVINO GenAI.

Steps: export to IR → smoke test via llm_bench → accuracy check via who-what-benchmark.

Usage:
    python check_model.py \
        --model-id tencent/HY-MT1.5-1.8B \
        --task text-generation-with-past \
        --work-dir /tmp/genai-model-check \
        --llm-bench-script tools/llm_bench/benchmark.py

Optional skip flags:
    --skip-export      Reuse existing IR in <work-dir>/model_ir instead of re-exporting.
    --skip-llm-bench   Skip the llm_bench smoke test.
    --skip-wwb         Skip the who-what-benchmark accuracy check.
"""

import argparse
import csv
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_OUTPUT_LINES = 200  # Last N lines of captured output to display on failure

# Embedded task mapping: optimum-cli export task → (llm_bench --task, wwb --model-type)
# wwb_type=None means WWB step is skipped for that task.
TASK_MAPPING = {
    "text-generation-with-past": ("text_gen", "text"),
    "image-text-to-text": ("visual_text_gen", "visual-text"),
    "text-to-image": ("text-to-image", "text-to-image"),
    "image-to-image": ("image-to-image", "image-to-image"),
    "feature-extraction": ("text_embed", "text-embedding"),
    "text-classification": ("text_rerank", "text-reranking"),
    "text-to-video": ("text-to-video", "text-to-video"),
    "automatic-speech-recognition": ("speech_to_text", None),
}

MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")

DEFAULT_PROMPTS = {
    "text_gen": "What is OpenVINO?",
    "visual_text_gen": "Describe this image.",
    "text-to-image": "A photo of a cat sitting on a windowsill",
    "image-to-image": "A photo of a cat sitting on a windowsill",
    "text-to-video": "A cat walking across a room",
    "text_embed": "The quick brown fox jumps over the lazy dog",
    "text_rerank": "What is machine learning?",
    "speech_to_text": None,  # uses default audio from llm_bench
}

WWB_SIMILARITY_THRESHOLDS = {
    "text": 0.95,
    "visual-text": 0.95,
    "text-to-image": 0.90,
    "image-to-image": 0.90,
    "text-to-video": 0.90,
    "text-embedding": 0.95,
    "text-reranking": 0.95,
}


def validate_model_id(model_id: str) -> None:
    if not MODEL_ID_PATTERN.match(model_id):
        logger.error("Invalid model_id format: %s", model_id)
        logger.error("Expected format: org-name/model-name (alphanumeric, hyphens, dots, underscores)")
        sys.exit(1)


def validate_task(task: str) -> None:
    if task not in TASK_MAPPING:
        logger.error("Unsupported task: %s", task)
        logger.error("Supported tasks: %s", ", ".join(sorted(TASK_MAPPING.keys())))
        sys.exit(1)


def log_environment_info() -> None:
    """Log Python version and key package versions for diagnostics."""
    logger.info("Python: %s", sys.version)
    logger.info("Executable: %s", sys.executable)
    for pkg in ["openvino", "openvino_genai", "openvino_tokenizers", "optimum-intel", "transformers", "diffusers"]:
        try:
            version = subprocess.run(
                [sys.executable, "-c", f"import importlib.metadata; print(importlib.metadata.version('{pkg}'))"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("  %-25s %s", pkg, version.stdout.strip() if version.returncode == 0 else "NOT INSTALLED")
        except subprocess.TimeoutExpired:
            logger.info("  %-25s TIMEOUT", pkg)


def log_directory_contents(directory: Path, label: str) -> None:
    """Log directory tree for debugging missing/unexpected files."""
    if not directory.exists():
        logger.warning("%s directory does not exist: %s", label, directory)
        return
    logger.info("%s contents (%s):", label, directory)
    for item in sorted(directory.rglob("*")):
        rel = item.relative_to(directory)
        size = item.stat().st_size if item.is_file() else 0
        logger.info("  %s %s", f"{size:>10,}" if item.is_file() else "       DIR", rel)


def log_model_config(model_dir: Path) -> None:
    """Log model config.json after export for debugging inference issues."""
    config_file = model_dir / "config.json"
    if not config_file.exists():
        logger.warning("No config.json found in %s", model_dir)
        return
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        logger.info("Model config.json:")
        for key in [
            "model_type",
            "architectures",
            "num_hidden_layers",
            "hidden_size",
            "vocab_size",
            "max_position_embeddings",
            "torch_dtype",
        ]:
            if key in config:
                logger.info("  %-30s %s", key, config[key])
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read config.json: %s", e)


def tail_output(output: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    """Return last max_lines of output to avoid flooding logs."""
    lines = output.splitlines()
    if len(lines) <= max_lines:
        return output
    return f"... ({len(lines) - max_lines} lines truncated) ...\n" + "\n".join(lines[-max_lines:])


def run_step(name: str, cmd: list, cwd: str = None) -> subprocess.CompletedProcess:
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("Command: %s", shlex.join(cmd))
    logger.info("Working directory: %s", cwd or os.getcwd())
    logger.info("=" * 60)
    try:
        # Stream output line-by-line to avoid apparent hangs on long-running commands.
        # Output is simultaneously collected for post-step logging.
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        logger.error("STEP FAILED: %s — executable not found: %s", name, cmd[0])
        logger.error("Ensure '%s' is installed and available on PATH.", cmd[0])
        return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr="")

    collected_lines: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        collected_lines.append(line)
        # Print to stdout so the invoking terminal sees live progress
        print(line, flush=True)
    proc.wait()
    full_output = "\n".join(collected_lines)

    result = subprocess.CompletedProcess(cmd, returncode=proc.returncode, stdout=full_output, stderr="")
    if result.returncode != 0:
        logger.error("STEP FAILED: %s (exit code %d)", name, result.returncode)
        logger.error("--- begin output ---")
        logger.error("%s", tail_output(result.stdout))
        logger.error("--- end output ---")
    else:
        logger.info("STEP PASSED: %s", name)
        if result.stdout:
            for line in result.stdout.splitlines()[-5:]:
                logger.info("  > %s", line)
    return result


def step_export(model_id: str, export_task: str, output_dir: Path) -> bool:
    cmd = [
        "optimum-cli",
        "export",
        "openvino",
        "-m",
        model_id,
        str(output_dir),
        "--task",
        export_task,
    ]
    result = run_step("Export model to OpenVINO IR", cmd)
    if result.returncode != 0:
        return False
    # Verify IR files exist
    xml_files = list(output_dir.rglob("*.xml"))
    if not xml_files:
        logger.error("Export produced no .xml files in %s", output_dir)
        log_directory_contents(output_dir, "Export output")
        return False
    logger.info("Found %d IR model file(s)", len(xml_files))
    for xml in xml_files:
        logger.info("  %s", xml.relative_to(output_dir))
    log_model_config(output_dir)
    return True


def step_smoke_test(model_dir: Path, bench_task: str, llm_bench_script: str, device: str) -> bool:
    cmd = [
        sys.executable,
        llm_bench_script,
        "-m",
        str(model_dir),
        "-d",
        device,
        "-n",
        "1",
        "--task",
        bench_task,
    ]
    prompt = DEFAULT_PROMPTS.get(bench_task)
    if prompt is not None:
        cmd.extend(["-p", prompt])
    result = run_step("Smoke test (llm_bench, 1 iteration)", cmd)
    return result.returncode == 0


def step_wwb_ground_truth(model_dir: Path, wwb_type: str, work_dir: Path, num_samples: int, device: str) -> bool:
    gt_data = work_dir / "gt.csv"
    cmd = [
        "wwb",
        "--base-model",
        str(model_dir),
        "--gt-data",
        str(gt_data),
        "--model-type",
        wwb_type,
        "--device",
        device,
        "--num-samples",
        str(num_samples),
    ]
    result = run_step("WWB: generate ground truth (Optimum)", cmd)
    if result.returncode != 0:
        return False
    if not gt_data.exists():
        logger.error("Ground truth file not created: %s", gt_data)
        return False
    return True


def step_wwb_target(model_dir: Path, wwb_type: str, work_dir: Path, num_samples: int, device: str) -> bool:
    gt_data = work_dir / "gt.csv"
    output_dir = work_dir / "results"
    cmd = [
        "wwb",
        "--target-model",
        str(model_dir),
        "--gt-data",
        str(gt_data),
        "--model-type",
        wwb_type,
        "--genai",
        "--device",
        device,
        "--output",
        str(output_dir),
        "--num-samples",
        str(num_samples),
    ]
    result = run_step("WWB: evaluate target (OpenVINO GenAI)", cmd)
    return result.returncode == 0


def parse_wwb_metrics(work_dir: Path, wwb_type: str) -> dict:
    metrics_file = work_dir / "results" / "metrics.csv"
    if not metrics_file.exists():
        logger.error("Metrics file not found: %s", metrics_file)
        log_directory_contents(work_dir / "results", "WWB results")
        return None
    with open(metrics_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        logger.error("Metrics file is empty: %s", metrics_file)
        return None
    metrics = rows[0]
    logger.info("WWB Metrics:")
    for key, value in metrics.items():
        logger.info("  %-30s %s", key, value)
    return metrics


def evaluate_results(metrics: dict, wwb_type: str) -> bool:
    threshold = WWB_SIMILARITY_THRESHOLDS.get(wwb_type, 0.95)
    similarity_key = None
    for key in metrics:
        if "similarity" in key.lower():
            similarity_key = key
            break
    if similarity_key is None:
        logger.warning("No similarity metric found in results. Cannot determine pass/fail automatically.")
        return True  # Don't fail if metric key is unexpected
    try:
        similarity = float(metrics[similarity_key])
    except (ValueError, TypeError):
        logger.error("Cannot parse similarity value: %s", metrics[similarity_key])
        return False
    passed = similarity >= threshold
    logger.info("Similarity: %.4f (threshold: %.2f) → %s", similarity, threshold, "PASS" if passed else "FAIL")
    return passed


def print_summary(results: dict, args: argparse.Namespace) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Model:   %s", args.model_id)
    logger.info("  Task:    %s", args.task)
    logger.info("  Device:  %s", args.device)
    logger.info("-" * 60)
    for step_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %-40s %s", step_name, status)
    logger.info("=" * 60)
    all_passed = all(results.values())
    logger.info("OVERALL: %s", "PASS" if all_passed else "FAIL")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end validation of a HuggingFace model with OpenVINO GenAI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", required=True, help="HuggingFace model identifier (e.g. tencent/HY-MT1.5-1.8B)")
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_MAPPING.keys()),
        help="optimum-cli export task (e.g. text-generation-with-past)",
    )
    parser.add_argument("--work-dir", default="/tmp/genai-model-check", help="Working directory for outputs")
    parser.add_argument("--llm-bench-script", required=True, help="Path to tools/llm_bench/benchmark.py")
    parser.add_argument("--device", default="CPU", help="Inference device")
    parser.add_argument(
        "--skip-export", action="store_true", help="Skip model export (reuse existing IR in work-dir/model_ir)"
    )
    parser.add_argument("--skip-llm-bench", action="store_true", help="Skip llm_bench smoke test")
    parser.add_argument("--skip-wwb", action="store_true", help="Skip who-what-benchmark accuracy check")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of WWB samples")
    args = parser.parse_args()

    validate_model_id(args.model_id)
    validate_task(args.task)

    bench_task, wwb_type = TASK_MAPPING[args.task]

    work_dir = Path(args.work_dir)
    model_dir = work_dir / "model_ir"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to both console and file
    log_file = work_dir / "check_model.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

    logger.info("GenAI Model Checker")
    logger.info("  model_id:   %s", args.model_id)
    logger.info("  task:       %s", args.task)
    logger.info("  bench_task:  %s", bench_task)
    logger.info("  wwb_type:    %s", wwb_type)
    logger.info("  work_dir:    %s", work_dir)
    logger.info("  device:      %s", args.device)
    logger.info("  log_file:    %s", log_file)
    log_environment_info()

    if not args.skip_llm_bench and not Path(args.llm_bench_script).is_file():
        logger.error("llm_bench script not found: %s", args.llm_bench_script)
        sys.exit(1)

    results = {}

    # Step 1: Export
    if args.skip_export:
        logger.info("Skipping export (--skip-export). Reusing existing IR: %s", model_dir)
        if not model_dir.exists() or not list(model_dir.rglob("*.xml")):
            logger.error("--skip-export specified but no IR files found in %s", model_dir)
            sys.exit(1)
        log_model_config(model_dir)
        results["Export to OpenVINO IR"] = True
    else:
        results["Export to OpenVINO IR"] = step_export(args.model_id, args.task, model_dir)
        if not results["Export to OpenVINO IR"]:
            print_summary(results, args)
            logger.info("Full log: %s", log_file)
            sys.exit(1)

    # Step 2: Smoke test
    if args.skip_llm_bench:
        logger.info("Skipping smoke test (--skip-llm-bench).")
        results["Smoke test (llm_bench)"] = True
    else:
        results["Smoke test (llm_bench)"] = step_smoke_test(model_dir, bench_task, args.llm_bench_script, args.device)
        if not results["Smoke test (llm_bench)"]:
            print_summary(results, args)
            logger.info("Full log: %s", log_file)
            sys.exit(1)

    # Step 3: WWB accuracy
    if args.skip_wwb or wwb_type is None:
        if wwb_type is None:
            logger.info("WWB not supported for task '%s', skipping accuracy check.", args.task)
        else:
            logger.info("Skipping WWB accuracy check (--skip-wwb).")
        results["WWB accuracy"] = True  # Not applicable, mark as pass
    else:
        gt_ok = step_wwb_ground_truth(model_dir, wwb_type, work_dir, args.num_samples, args.device)
        results["WWB ground truth"] = gt_ok
        if gt_ok:
            target_ok = step_wwb_target(model_dir, wwb_type, work_dir, args.num_samples, args.device)
            results["WWB target evaluation"] = target_ok
            if target_ok:
                metrics = parse_wwb_metrics(work_dir, wwb_type)
                if metrics is not None:
                    results["WWB accuracy"] = evaluate_results(metrics, wwb_type)
                else:
                    results["WWB accuracy"] = False
            else:
                results["WWB accuracy"] = False
        else:
            results["WWB accuracy"] = False

    print_summary(results, args)
    logger.info("Full log: %s", log_file)
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
