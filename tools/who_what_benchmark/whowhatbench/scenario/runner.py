# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from whowhatbench.scenario.args_builder import build_args_namespace
from whowhatbench.scenario.gt_cache import GTCache
from whowhatbench.scenario.result_store import ResultStore, TaskResult
from whowhatbench.scenario.schema import DatasetConfig, DatasetTypeEnum, ModelConfig, Scenario, TaskConfig

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
            for target_id in task.targets:
                logger.info("Running task %r against target %r", task.id, target_id)
                result = self._run_one(task, target_id)
                store.add(result)

        return store

    def _run_one(self, task: TaskConfig, target_id: str) -> TaskResult:
        # Deferred: importing the full evaluator stack (which pulls in openvino_genai)
        # only happens when a task is actually executed, not at scenario load time.
        from whowhatbench.model_loaders import load_model  # noqa: PLC0415
        from whowhatbench.wwb import create_evaluator  # noqa: PLC0415

        base_model_cfg = self._scenario.models[task.base]
        dataset_cfg = self._scenario.datasets[task.dataset]
        task_out = self._output_dir / "tasks" / task.id / target_id

        # Inline / CSV datasets must be materialised into a test_data dict
        # before constructing args, since the evaluator receives them directly.
        test_data = self._resolve_test_data(dataset_cfg)

        gt_path, gt_cache_hit = self._prepare_gt(task, target_id, task_out, base_model_cfg, dataset_cfg, test_data)

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
        )

        evaluator = create_evaluator(None, args, test_data=test_data)
        gen_fn = (
            evaluator.get_generation_fn() if (args.genai or args.llamacpp or task.type == "speech-generation") else None
        )

        task_out.mkdir(parents=True, exist_ok=True)
        per_q_raw, metrics_raw = evaluator.score(target_model, gen_fn, output_dir=str(task_out), verbose=False)
        runtime_s = time.monotonic() - t0
        evaluator.dump_predictions(str(task_out / "target.csv"))

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
        target_id: str,
        task_out: Path,
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

        gt_path = str(self._gt_cache._dir / f"{gt_key}.csv")
        logger.info(
            "GT cache miss for task %r — generating with base model %r",
            task.id,
            base_model_cfg.path,
        )
        args_for_gt = build_args_namespace(self._scenario, task, target_id, task_out, None)

        # Deferred imports — same cache hit as in _run_one; free after first call.
        from whowhatbench.model_loaders import load_model  # noqa: PLC0415
        from whowhatbench.wwb import create_evaluator  # noqa: PLC0415

        # Base model always runs on CPU for GT to avoid device contention with
        # the target evaluation device, and to keep GT deterministic.
        base_model = load_model(
            task.type,
            base_model_cfg.path,
            "CPU",
            None,
            base_model_cfg.backend.value == "hf",
            base_model_cfg.backend.value == "genai",
        )

        evaluator = create_evaluator(base_model, args_for_gt, test_data=test_data)
        evaluator.dump_gt(gt_path)

        # Free the base model before loading the target to limit peak memory.
        del base_model
        del evaluator

        self._gt_cache.put(gt_key, Path(gt_path))
        return gt_path, False

    def _resolve_test_data(self, dataset_cfg: DatasetConfig) -> Optional[dict[str, Any]]:
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
            return _load_csv_dataset(dataset_cfg)
        return None


def _load_csv_dataset(dataset_cfg: DatasetConfig) -> dict[str, list[Any]]:
    if dataset_cfg.path is None:
        raise ValueError("CSV dataset requires a 'path' field.")
    df = pd.read_csv(dataset_cfg.path)
    field = dataset_cfg.field
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
        return {k: (v[0] if isinstance(v, list) else v) for k, v in value.items()}
    return {}
