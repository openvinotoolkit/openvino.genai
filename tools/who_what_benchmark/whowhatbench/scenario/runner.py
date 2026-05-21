# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from whowhatbench.model_loaders import load_model
from whowhatbench.scenario.args_builder import build_args_namespace
from whowhatbench.scenario.gt_cache import GTCache
from whowhatbench.scenario.result_store import ResultStore, TaskResult
from whowhatbench.scenario.schema import DatasetConfig, DatasetTypeEnum, ModelConfig, Scenario, TaskConfig
from whowhatbench.wwb import create_evaluator

logger = logging.getLogger(__name__)


class ScenarioRunner:
    def __init__(self, scenario: Scenario, output_dir: Path) -> None:
        self._scenario = scenario
        self._output_dir = Path(output_dir)
        self._gt_cache = GTCache(self._output_dir / "_gt_cache")

    def run(self, only_task_ids: Optional[list[str]] = None) -> ResultStore:
        store = ResultStore()

        if only_task_ids is not None:
            unknown = set(only_task_ids) - {t.id for t in self._scenario.tasks}
            if unknown:
                raise ValueError(
                    f"Unknown task IDs: {sorted(unknown)}. Available: {[t.id for t in self._scenario.tasks]}"
                )

        for task in self._scenario.tasks:
            if only_task_ids is not None and task.id not in only_task_ids:
                continue

            dataset_cfg = self._scenario.datasets[task.dataset]
            base_model_cfg = self._scenario.models[task.base]

            # Per-task work hoisted out of the per-target loop: dataset
            # materialisation and GT cache key computation only depend on
            # base/dataset/task fields shared across targets.
            test_data = self._resolve_test_data(dataset_cfg, task)
            gt_path, gt_cache_hit_first = self._prepare_gt(task, base_model_cfg, dataset_cfg, test_data)

            # First target after a fresh generation sees a cache miss; later
            # targets sharing the same GT file effectively hit the cache.
            for index, target_id in enumerate(task.targets):
                logger.info("Running task %r against target %r", task.id, target_id)
                gt_cache_hit = gt_cache_hit_first or index > 0
                try:
                    result = self._run_one(task, target_id, test_data, gt_path, gt_cache_hit)
                    store.add(result)
                except Exception:
                    logger.exception("Task %r target %r failed — continuing", task.id, target_id)

        return store

    def _run_one(
        self,
        task: TaskConfig,
        target_id: str,
        test_data: Optional[dict[str, Any]],
        gt_path: str,
        gt_cache_hit: bool,
    ) -> TaskResult:
        task_out = self._output_dir / "tasks" / task.id / target_id

        args = build_args_namespace(self._scenario, task, target_id, task_out, gt_path)
        target_model_cfg = self._scenario.models[target_id]

        logger.info("Loading target model %r on %s", target_model_cfg.path, args.device)
        t0 = time.monotonic()
        target_model = load_model(
            task.type,
            target_model_cfg.path,
            args.device,
            args.ov_config,
            args.hf,
            args.genai,
            use_llamacpp=args.llamacpp,
            from_onnx=args.from_onnx,
            gguf_file=args.gguf_file,
            cb_config=args.cb_config,
            adapters=args.adapters,
            alphas=args.alphas,
            empty_adapters=args.empty_adapters,
            draft_model=args.draft_model,
            draft_device=args.draft_device,
            draft_cb_config=args.draft_cb_config,
            vocoder_path=args.vocoder_path,
        )

        evaluator = create_evaluator(None, args, test_data=test_data)
        gen_fn = (
            evaluator.get_generation_fn() if (args.genai or args.llamacpp or task.type == "speech-generation") else None
        )

        task_out.mkdir(parents=True, exist_ok=True)
        per_q_raw, metrics_raw = evaluator.score(target_model, gen_fn, output_dir=str(task_out), verbose=False)
        runtime_s = time.monotonic() - t0
        evaluator.dump_predictions(str(task_out / "target.csv"))
        del target_model
        del evaluator

        per_question = _normalize_per_question(per_q_raw)
        metrics = _normalize_metrics(metrics_raw)

        return TaskResult(
            task_id=task.id,
            target_id=target_id,
            metrics=metrics,
            per_question=per_question,
            runtime_s=runtime_s,
            output_dir=task_out,
            gt_cache_hit=gt_cache_hit,
        )

    def _prepare_gt(
        self,
        task: TaskConfig,
        base_model_cfg: ModelConfig,
        dataset_cfg: DatasetConfig,
        test_data: Optional[dict[str, Any]],
    ) -> tuple[str, bool]:
        if task.gt_data:
            return task.gt_data, False

        gt_key = self._gt_cache.key(task, base_model_cfg, dataset_cfg, self._scenario)
        cached = self._gt_cache.get(gt_key)
        if cached is not None:
            logger.info("GT cache hit for task %r (key=%s)", task.id, gt_key)
            return str(cached), True

        gt_path = self._gt_cache.allocate_path(gt_key)
        logger.info(
            "GT cache miss for task %r — generating with base model %r",
            task.id,
            base_model_cfg.path,
        )

        # Build the args namespace from the BASE model id so backend flags
        # (hf/genai/llamacpp) reflect the model that actually produces GT.
        # Using the target id here would route the base model through the wrong
        # backend code path inside the evaluator.
        # gt_data_path is set to the allocated cache path so evaluators that
        # derive their reference directory from it (embeddings, reranking)
        # can place per-sample artifacts alongside the GT CSV.
        task_out = self._output_dir / "tasks" / task.id / "_gt"
        args_for_gt = build_args_namespace(self._scenario, task, task.base, task_out, str(gt_path))

        # Base model always runs on CPU for GT to avoid device contention with
        # the target evaluation device, and to keep GT deterministic.
        base_model = load_model(
            task.type,
            base_model_cfg.path,
            "CPU",
            None,
            args_for_gt.hf,
            args_for_gt.genai,
            use_llamacpp=args_for_gt.llamacpp,
            from_onnx=args_for_gt.from_onnx,
            gguf_file=args_for_gt.gguf_file,
            cb_config=args_for_gt.cb_config,
            adapters=args_for_gt.adapters,
            alphas=args_for_gt.alphas,
            empty_adapters=args_for_gt.empty_adapters,
            draft_model=args_for_gt.draft_model,
            draft_device=args_for_gt.draft_device,
            draft_cb_config=args_for_gt.draft_cb_config,
            vocoder_path=args_for_gt.vocoder_path,
        )

        evaluator = create_evaluator(base_model, args_for_gt, test_data=test_data)
        evaluator.dump_gt(str(gt_path))

        # Free the base model before loading the target to limit peak memory.
        del base_model
        del evaluator

        self._gt_cache.put(gt_key, gt_path)
        return str(gt_path), False

    def _resolve_test_data(
        self,
        dataset_cfg: DatasetConfig,
        task: TaskConfig,
    ) -> Optional[dict[str, Any]]:
        """Return a test_data dict for inline/CSV datasets, else None.

        Builtin and HuggingFace datasets are loaded by the evaluator itself
        from args.dataset / args.split, so the runner passes test_data=None.
        """
        if dataset_cfg.type == DatasetTypeEnum.inline:
            if dataset_cfg.prompts is not None:
                return {"prompts": dataset_cfg.prompts}
            if dataset_cfg.passages is not None:
                return {"passages": dataset_cfg.passages}
            if dataset_cfg.chats is not None:
                return {"chats": dataset_cfg.chats}
            raise ValueError("Inline dataset has no prompts/passages/chats payload.")
        if dataset_cfg.type == DatasetTypeEnum.csv:
            return _load_csv_dataset(dataset_cfg, task.num_samples)
        return None


def _load_csv_dataset(dataset_cfg: DatasetConfig, num_samples: Optional[int]) -> dict[str, list[Any]]:
    if dataset_cfg.path is None:
        raise ValueError("CSV dataset requires a 'path' field.")
    field = dataset_cfg.field
    # Bound the read to num_samples (when present) and the requested column —
    # avoids loading huge CSVs into memory when only a small slice is needed.
    df = pd.read_csv(dataset_cfg.path, usecols=[field], nrows=num_samples, dtype=str)
    if field not in df.columns:
        raise ValueError(
            f"Field {field!r} not found in CSV {dataset_cfg.path!r}. Available columns: {list(df.columns)}"
        )
    return {"prompts": df[field].tolist()}


def _normalize_per_question(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records") if len(value) > 0 else []
    if isinstance(value, dict):
        if not value:
            return []
        return pd.DataFrame(value).to_dict("records")
    return []


def _normalize_metrics(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        return value.iloc[0].to_dict() if len(value) > 0 else {}
    if isinstance(value, dict):
        return {k: (v[0] if isinstance(v, list) and v else v) for k, v in value.items()}
    return {}
