# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for the tool_call_benchmark."""

import argparse
import json
import logging
import os
import sys
import time

from toolcallbench.dataset import DATASET_SHA256, DATASET_VERSION, _sha256, load_dataset
from toolcallbench.evaluator import ToolCallEvaluator
from toolcallbench.model_loaders import load_pipeline
from toolcallbench.verdict import THRESHOLDS

logger = logging.getLogger("toolcallbench")


def get_argparser():
    parser = argparse.ArgumentParser(
        prog="tcb",
        description="Tool-call capability benchmark for coding-agent style LLMs "
                    "(deterministic, no sandbox, frozen 50-case dataset).")
    parser.add_argument("--model", required=True,
                        help="OpenVINO IR directory with the exported model")
    parser.add_argument("--tokenizer", default=None,
                        help="HF tokenizer directory with the chat template (default: --model)")
    parser.add_argument("--device", default="CPU", help="OpenVINO device (default: CPU)")
    parser.add_argument("--ov-config", default=None,
                        help="OpenVINO properties as a JSON string or a path to a JSON file")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="generation budget per turn (default: 1024)")
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated case ids to run (default: all)")
    parser.add_argument("--skip-categories", default=None,
                        help="comma-separated category names to skip (default: none)")
    parser.add_argument("--dataset", default=None,
                        help="dataset JSONL path (default: packaged coding_agent_v1)")
    parser.add_argument("--output", default=None,
                        help="output directory; writes report.json and cases.jsonl")
    return parser


def parse_ov_config(value):
    if not value:
        return None
    if os.path.exists(value):
        with open(value) as f:
            return json.load(f)
    return json.loads(value)


def main(argv=None):
    args = get_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from transformers import AutoTokenizer

    tokenizer_dir = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    ov_config = parse_ov_config(args.ov_config)
    pipeline = load_pipeline(args.model, args.device, ov_config)

    case_ids = [c for c in args.case_ids.split(",") if c] if args.case_ids else None
    skip_categories = ([c for c in args.skip_categories.split(",") if c]
                       if args.skip_categories else None)

    evaluator = ToolCallEvaluator(pipeline, tokenizer,
                                  max_new_tokens=args.max_new_tokens,
                                  dataset_path=args.dataset)
    start = time.time()
    report = evaluator.evaluate(case_ids=case_ids, skip_categories=skip_categories)
    elapsed = round(time.time() - start, 1)

    print(f"\nverdict:       {report['verdict']}")
    print(f"overall:       {report['overall']}")
    print(f"format-valid:  {report['format_valid']}")
    print(f"unsafe acts:   {report['unsafe_acts']}")
    for name, score in report["categories"].items():
        print(f"{name:<20} {score}")
    print(f"elapsed:       {elapsed}s")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        import openvino_genai
        meta, _ = load_dataset(args.dataset)
        dataset_version = meta.get("version", DATASET_VERSION)
        dataset_sha = _sha256(args.dataset) if args.dataset else DATASET_SHA256
        report["run"] = {
            "model": args.model,
            "tokenizer": tokenizer_dir,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "dataset_version": dataset_version,
            "dataset_sha256": dataset_sha,
            "openvino_genai_version": getattr(openvino_genai, "__version__", "unknown"),
            "thresholds": THRESHOLDS,
            "elapsed_sec": elapsed,
        }
        with open(os.path.join(args.output, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(args.output, "cases.jsonl"), "w") as f:
            for r in report["results"]:
                f.write(json.dumps(r) + "\n")
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
