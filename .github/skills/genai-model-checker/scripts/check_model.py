#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validation of a HuggingFace model with OpenVINO GenAI."""

import argparse
from dataclasses import dataclass
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

FAIL_TAIL_LINES = 10  # Lines of output to show on failure (full output goes to per-step log file)

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


WWB_SIMILARITY_THRESHOLD = 0.95


def log_header(args: argparse.Namespace, bench_task: str, wwb_task: str, work_dir: str, log_file: str) -> None:
    logger.info("GenAI Model Checker")
    logger.info("  model_id:    %s", args.model_id)
    logger.info("  task:        %s", args.task)
    logger.info("  device:      %s", args.device)
    logger.info("  bench_task:  %s", bench_task)
    logger.info("  wwb_type:    %s", wwb_task)
    logger.info("  work_dir:    %s", work_dir)
    logger.info("  log_file:    %s", log_file)
    logger.info("Python: %s", sys.version)
    logger.info("Executable: %s", sys.executable)
    for pkg in ["openvino", "openvino_genai", "openvino_tokenizers", "optimum-intel", "transformers", "diffusers"]:
        version = subprocess.run(
            [sys.executable, "-c", f"import importlib.metadata; print(importlib.metadata.version('{pkg}'))"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        logger.info("  %-25s %s", pkg, version.stdout.strip() if version.returncode == 0 else "NOT INSTALLED")


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


@dataclass
class ToolResult:
    name: str
    success: bool
    err_msg: str | None = None

    def raise_if_failed(self) -> None:
        if not self.success and self.err_msg:
            raise RuntimeError(self.err_msg)


class ToolWrapper:
    """Utility to run a command-line tool and capture output for logging."""

    def __init__(self, name: str, commands_list: list, work_dir: Path):
        self.name = name
        self.commands_list = commands_list
        self.work_dir = work_dir

        if self.work_dir:
            self.work_dir.mkdir(parents=True, exist_ok=True)

        self.logger_prefix = f"[{self.name}]"
        self.log_path = self.work_dir / f"{self.name}.log"
        self.log_path.touch(exist_ok=True)

    def _post_run_hook(self, result: subprocess.CompletedProcess) -> ToolResult:
        """Process results"""
        if result.returncode == 0:
            return ToolResult(name=self.name, success=True)

        err_messages = []
        err_messages.append(f"{self.logger_prefix}: failed with exit code {result.returncode}")
        err_messages.append(f"{self.logger_prefix}: Last lines from {self.log_path}:")
        for line in result.stdout[-FAIL_TAIL_LINES:]:
            err_messages.append(f"  {line}")
        err_msg = "\n".join(err_messages)
        return ToolResult(name=self.name, success=False, err_msg=err_msg)

    def run(self) -> ToolResult:
        command = shlex.join(self.commands_list)
        logger.info(f"{self.logger_prefix}: {command}")
        logger.info(f"{self.logger_prefix}: Log path: {self.log_path.resolve()}")

        with open(self.log_path, "w") as lf:
            lf.write(f"Command: {command}\n")
            lf.write(f"work_dir: {self.work_dir.resolve() if self.work_dir else Path(os.getcwd()).resolve()}\n")
            lf.write("Tool output:\n")
            lf.flush()

            start = time.perf_counter()
            # Using Popen directly to stream output line-by-line to log file
            proc = subprocess.Popen(
                self.commands_list,
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=10,
            )

            tool_logs: list[str] = []
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                tool_logs.append(line.rstrip("\n"))
            proc.wait()

            elapsed = time.perf_counter() - start

        result = subprocess.CompletedProcess(self.commands_list, returncode=proc.returncode, stdout=tool_logs)
        logger.info(f"{self.logger_prefix}: took {elapsed:.2f} seconds")

        return self._post_run_hook(result)


class OptimumExportTool(ToolWrapper):
    def __init__(self, model_id: str, task: str, model_dir: Path, work_dir: Path):
        cmd = [
            "optimum-cli",
            "export",
            "openvino",
            "-m",
            model_id,
            str(model_dir),
            "--task",
            task,
        ]
        super().__init__(name="optimum_export", commands_list=cmd, work_dir=work_dir)


class LlmBenchTool(ToolWrapper):
    def __init__(self, model_dir: Path, task: str, device: str, work_dir: Path):
        llm_bench_script_path = Path("tools/llm_bench/benchmark.py")

        if not llm_bench_script_path.is_file():
            raise FileNotFoundError(f"llm_bench script not found: {llm_bench_script_path}")
        cmd = [
            sys.executable,
            str(llm_bench_script_path),
            "-m",
            str(model_dir),
            "-d",
            device,
            "-n",
            "1",
            "--task",
            task,
        ]
        super().__init__(name="llm_bench", commands_list=cmd, work_dir=work_dir)

    def _post_run_hook(self, result):
        if result.returncode != 0:
            return super()._post_run_hook(result)
        last_llm_bench_line = result.stdout[-1]
        logger.info(f"{self.logger_prefix}: llm_bench metrics: {last_llm_bench_line}")
        return ToolResult(name=self.name, success=True)


class HFWWBGroundTruthTool(ToolWrapper):
    def __init__(self, model_id: str, task: str, work_dir: Path, num_samples: int, device: str):
        cmd = [
            "wwb",
            "--base-model",
            model_id,
            "--gt-data",
            str(work_dir / "gt.csv"),
            "--model-type",
            task,
            "--device",
            device,
            "--num-samples",
            str(num_samples),
            "--hf",  # Use HuggingFace backend for ground truth generation
        ]
        super().__init__(name="wwb_hf_ground_truth", commands_list=cmd, work_dir=work_dir)


def parse_wwb_metrics_value(stdout: list[str]) -> float | None:
    # example output, parse similarity
    # INFO:whowhatbench.wwb:Metrics for model: model-check/model_ir
    # INFO:whowhatbench.wwb:   similarity
    # 0    0.957204
    if len(stdout) < 1:
        return None
    metric_value_str = stdout[-1].strip().split()[1]
    return float(metric_value_str)


class OptimumWWBTargetEvaluationTool(ToolWrapper):
    def __init__(self, model_dir: Path, task: str, work_dir: Path, num_samples: int, device: str):
        cmd = [
            "wwb",
            "--target-model",
            str(model_dir),
            "--gt-data",
            str(work_dir / "gt.csv"),
            "--model-type",
            task,
            "--device",
            device,
            "--num-samples",
            str(num_samples),
            # Optimum backend used by default
            "--output",
            str(work_dir / "optimum"),
        ]
        super().__init__(name="wwb_optimum_target_eval", commands_list=cmd, work_dir=work_dir)

    def _post_run_hook(self, result: subprocess.CompletedProcess) -> ToolResult:
        if result.returncode != 0:
            return super()._post_run_hook(result)

        metrics_value = parse_wwb_metrics_value(result.stdout)
        if metrics_value is None:
            return ToolResult(
                name=self.name,
                success=False,
                err_msg=f"{self.logger_prefix}: Failed to parse WWB metrics value from output.",
            )
        if metrics_value < WWB_SIMILARITY_THRESHOLD:
            return ToolResult(
                name=self.name,
                success=False,
                err_msg=(
                    f"{self.logger_prefix}: WWB similarity {metrics_value:.4f} is below threshold of "
                    f"{WWB_SIMILARITY_THRESHOLD:.2f}."
                ),
            )
        logger.info(
            f"{self.logger_prefix}: optimum similarity {metrics_value:.4f} meets threshold of {WWB_SIMILARITY_THRESHOLD:.2f}."
        )
        return ToolResult(name=self.name, success=True)


class GenAIWWBTargetEvaluationTool(ToolWrapper):
    def __init__(self, model_dir: Path, task: str, work_dir: Path, num_samples: int, device: str):
        cmd = [
            "wwb",
            "--target-model",
            str(model_dir),
            "--gt-data",
            str(work_dir / "gt.csv"),
            "--model-type",
            task,
            "--device",
            device,
            "--num-samples",
            str(num_samples),
            "--genai",  # Use GenAI backend for target evaluation
            "--output",
            str(work_dir / "genai"),
        ]
        super().__init__(name="wwb_genai_target_eval", commands_list=cmd, work_dir=work_dir)

    def _post_run_hook(self, result: subprocess.CompletedProcess) -> ToolResult:
        if result.returncode != 0:
            return super()._post_run_hook(result)

        metrics_value = parse_wwb_metrics_value(result.stdout)
        if metrics_value is None:
            return ToolResult(
                name=self.name,
                success=False,
                err_msg=f"{self.logger_prefix}: Failed to parse WWB metrics value from output.",
            )
        if metrics_value < WWB_SIMILARITY_THRESHOLD:
            return ToolResult(
                name=self.name,
                success=False,
                err_msg=(
                    f"{self.logger_prefix}: WWB similarity {metrics_value:.4f} is below threshold of "
                    f"{WWB_SIMILARITY_THRESHOLD:.2f}."
                ),
            )
        logger.info(
            f"{self.logger_prefix}: genai similarity {metrics_value:.4f} meets threshold of {WWB_SIMILARITY_THRESHOLD:.2f}."
        )
        return ToolResult(name=self.name, success=True)


def _setup_exception_logging() -> None:
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def _setup_logging(log_file: Path) -> None:
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
    _setup_exception_logging()


def _get_arguments() -> argparse.Namespace:
    def model_id_validator(model_id: str) -> str:
        MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
        if not MODEL_ID_PATTERN.match(model_id):
            raise RuntimeError(
                f"Invalid model_id format: {model_id}\n"
                "Expected format: org-name/model-name (alphanumeric, hyphens, dots, underscores)"
            )
        return model_id

    class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "End-to-end validation of a HuggingFace model with OpenVINO GenAI.\n\n"
            "Steps:\n"
            "  1. Export model to OpenVINO IR via optimum-cli.\n"
            "  2. Smoke test via llm_bench (1 iteration).\n"
            f"  3. Accuracy check via who-what-benchmark (similarity threshold: {WWB_SIMILARITY_THRESHOLD}):\n"
            "       a. Generate ground truth with HuggingFace backend.\n"
            "       b. Evaluate with Optimum backend.\n"
            "       c. Evaluate with GenAI backend."
        ),
        epilog=(
            "Examples:\n"
            "  python check_model.py \\\n"
            "      --model-id tencent/HY-MT1.5-1.8B \\\n"
            "      --task text-generation-with-past \\\n"
            "      --work-dir /tmp/genai-model-check\n\n"
            "  # Reuse existing IR, skip accuracy check:\n"
            "  python check_model.py \\\n"
            "      --model-id tencent/HY-MT1.5-1.8B \\\n"
            "      --task text-generation-with-past \\\n"
            "      --skip-export --skip-wwb"
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=model_id_validator,
        required=True,
        help="HuggingFace model identifier (e.g. tencent/HY-MT1.5-1.8B)",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(TASK_MAPPING.keys()),
        help="optimum-cli export task (e.g. text-generation-with-past)",
    )

    parser.add_argument("--work-dir", default="/tmp/genai-model-check", help="Working directory for outputs")
    parser.add_argument("--device", default="CPU", help="Inference device")
    parser.add_argument(
        "--skip-export", action="store_true", help="Skip model export (reuse existing IR in work-dir/model_ir)"
    )
    parser.add_argument("--skip-llm-bench", action="store_true", help="Skip llm_bench smoke test")
    parser.add_argument("--skip-wwb", action="store_true", help="Skip who-what-benchmark accuracy check")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of WWB samples")
    return parser.parse_args()


def main():
    args = _get_arguments()

    bench_task, wwb_task = TASK_MAPPING[args.task]

    work_dir = Path(args.work_dir)
    model_dir = work_dir / "model_ir"
    work_dir.mkdir(parents=True, exist_ok=True)

    log_file = work_dir / "check_model.log"
    _setup_logging(log_file)

    log_header(args, bench_task, wwb_task, work_dir, log_file)

    # Step 1: Export
    if args.skip_export:
        logger.info("Skipping model export. Reusing existing IR: %s", model_dir)
    else:
        optimum_export_work_dir = work_dir / "optimum_export"
        result = OptimumExportTool(args.model_id, args.task, model_dir, optimum_export_work_dir).run()
        result.raise_if_failed()

    # Step 2: Smoke test
    if args.skip_llm_bench:
        logger.info("Skipping llm_bench test")
    else:
        llm_bench_work_dir = work_dir / "llm_bench"
        result = LlmBenchTool(model_dir, bench_task, args.device, llm_bench_work_dir).run()
        result.raise_if_failed()

    # Step 3: WWB accuracy
    if args.skip_wwb or wwb_task is None:
        logger.info("Skipping wwb accuracy check")
    else:
        wwb_work_dir = work_dir / "wwb"
        hf_gt_result = HFWWBGroundTruthTool(args.model_id, wwb_task, wwb_work_dir, args.num_samples, args.device).run()
        hf_gt_result.raise_if_failed()

        optimum_result = OptimumWWBTargetEvaluationTool(
            model_dir, wwb_task, wwb_work_dir, args.num_samples, args.device
        ).run()
        optimum_result.raise_if_failed()

        genai_result = GenAIWWBTargetEvaluationTool(
            model_dir, wwb_task, wwb_work_dir, args.num_samples, args.device
        ).run()
        genai_result.raise_if_failed()


if __name__ == "__main__":
    main()
