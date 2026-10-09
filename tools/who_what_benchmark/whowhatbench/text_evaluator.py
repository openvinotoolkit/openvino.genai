# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import pandas as pd

from tqdm import tqdm
from typing import Any, Union
from importlib.resources import files

from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import TextDivergency, TextSimilarity, KLDivergency
from .text_generation_strategies import GenerationStrategy
from .text_metrics_collection import Metrics, ArtifactsSchema, ArtifactsManager


PROMPTS_FILE = 'text_prompts.yaml'
LONG_PROMPTS_FILE = 'text_long_prompts.yaml'


@register_evaluator("text")
class TextEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics: str = None,
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens=128,
        crop_question=True,
        num_samples=None,
        language="en",
        gen_answer_fn=None,
        generation_config=None,
        seqs_per_request=None,
        use_chat_template=None,
        long_prompt=True,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
        kl_input_tokens=None,
        kld_ctx=None,
        kld_chunk=None,
        metrics_list: list = [],
        is_genai=False,
        is_llamacpp=False,
        output_dir=None,
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self._crop_question = crop_question
        self.num_samples = num_samples
        self.generation_config = generation_config
        self.seqs_per_request = seqs_per_request
        self.generation_fn = gen_answer_fn
        self.use_chat_template = use_chat_template
        self.num_assistant_tokens = num_assistant_tokens
        self.assistant_confidence_threshold = assistant_confidence_threshold
        self.generation_config_extra = generation_config_extra or {}
        self.kld_ctx = kld_ctx
        self.kld_chunk = kld_chunk
        if self.generation_config is not None:
            assert self.seqs_per_request is not None
        self.empty_adapters = empty_adapters

        # Take language from the base model if provided
        self.language = language

        self.long_prompt = long_prompt

        self.output_dir = output_dir
        self._generation_pass_idx = 0

        self.metrics_list = metrics_list

        # to support previous version of the code, where metrics was passed as a single string
        if self.metrics and (not self.metrics_list or self.metrics not in self.metrics_list):
            if not self.metrics_list:
                self.metrics_list = []
            self.metrics_list.append(self.metrics)

        if not self.metrics_list:
            self.metrics_list = [Metrics.SIMILARITY.value]

        self.is_genai = is_genai
        self.is_llamacpp = is_llamacpp
        self.generation_strategy = GenerationStrategy.create(
            self.metrics_list, self.is_genai, self.is_llamacpp, self.kld_ctx, self.kld_chunk
        )
        self.requisted_artifacts = ArtifactsSchema(self.metrics_list, self.long_prompt)

        if base_model:
            self.gt_data = self._generate_data(
                base_model,
                gen_answer_fn,
                generation_config=generation_config,
                output_dir=self.output_dir or "reference",
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        # Take language ground truth if no base model provided
        if self.language is None and "language" in self.gt_data.columns:
            self.language = self.gt_data["language"].values[0]

        if "prompt_length_type" in self.gt_data.columns:
            self.long_prompt = self.gt_data["prompt_length_type"].values[0] == 'long'

        self.similarity = None
        self.divergency = None
        if "similarity" in self.metrics_list:
            self.similarity = TextSimilarity(similarity_model_id)
        if "divergency" in self.metrics_list:
            assert tokenizer is not None
            self.divergency = TextDivergency(tokenizer)
        if Metrics.KL_DIVERGENCY.value in self.metrics_list:
            self.kl_divergency = KLDivergency()

        self.last_cmp = None

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_answer_fn=None, output_dir="wwb_target", **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            reference_token_ids_path = None
            if isinstance(self.gt_data, pd.DataFrame) and "prompt_input_ids_path" in self.gt_data.columns:
                reference_token_ids_path = []
                for _, row in self.gt_data.iterrows():
                    if isinstance(row.get("prompt_input_ids_path"), str):
                        reference_token_ids_path.append(row["prompt_input_ids_path"])

            predictions = self._generate_data(
                model_or_data,
                gen_answer_fn,
                self.generation_config,
                reference_token_ids_path=reference_token_ids_path,
                output_dir=output_dir,
            )
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(self.gt_data, predictions)
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(self.gt_data, predictions)
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if Metrics.KL_DIVERGENCY.value in self.metrics_list:
            kl_metric_dict, kl_metric_per_question = self.kl_divergency.evaluate(self.gt_data, predictions)
            all_metrics.update(kl_metric_dict)
            all_metrics_per_prompt.update(kl_metric_per_question)

        compared_rows = min(len(self.gt_data), len(predictions))
        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompts"].values[:compared_rows]
        self.last_cmp["source_model"] = (
            self.gt_data["answers"].values[:compared_rows] if "answers" in self.gt_data else [""] * compared_rows
        )
        self.last_cmp["optimized_model"] = (
            predictions["answers"].values[:compared_rows] if "answers" in self.predictions else [""] * compared_rows
        )
        self.last_cmp = pd.DataFrame(self.last_cmp)
        self.last_cmp.rename(columns={"prompts": "prompt"}, inplace=True)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        if metric in ["SDT", "SDT norm", "kl_divergency"]:
            res = self.last_cmp.nlargest(top_k, metric)
        else:
            res = self.last_cmp.nsmallest(top_k, metric)

        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(
        self,
        model,
        gen_answer_fn=None,
        generation_config=None,
        reference_token_ids_path=None,
        output_dir=None,
    ):
        gen_answer_fn = gen_answer_fn or self.generation_strategy.run_generation

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "prompts" in self.test_data
                    data = dict(self.test_data)
                else:
                    data = {"prompts": list(self.test_data)}
                data = pd.DataFrame.from_dict(data)
        else:
            prompts_file_path = LONG_PROMPTS_FILE if self.long_prompt else PROMPTS_FILE
            data_path = files('whowhatbench.prompts').joinpath(prompts_file_path)
            prompt_data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
            data = pd.DataFrame.from_dict(prompt_data[self.language])

        prompt_data = data["prompts"]
        prompts = prompt_data.values if self.num_samples is None else prompt_data.values[: self.num_samples]

        artifacts_collector = ArtifactsManager(self.requisted_artifacts, output_dir)
        generated_token_ids = artifacts_collector.collect_metadata_for_generation(reference_token_ids_path)

        if generation_config is None:
            for sample_idx, p in enumerate(tqdm(prompts, desc="Evaluate pipeline")):
                reference_tokens = None
                if generated_token_ids is not None and sample_idx < len(generated_token_ids):
                    reference_tokens = generated_token_ids[sample_idx]

                output = gen_answer_fn(
                    model,
                    self.tokenizer,
                    p,
                    reference_tokens,
                    self.max_new_tokens,
                    self._crop_question,
                    self.use_chat_template,
                    self.empty_adapters,
                    self.num_assistant_tokens,
                    self.assistant_confidence_threshold,
                    generation_config_extra=self.generation_config_extra,
                )

                artifacts_collector.collect(
                    sample_idx,
                    p,
                    output,
                )
        else:
            if self.generation_config_extra:
                for k, v in self.generation_config_extra.items():
                    if hasattr(generation_config, k):
                        setattr(generation_config, k, v)
            with tqdm(total=len(prompt_data.values), desc="Evaluate pipeline") as progress_bar:
                batch = []
                for p_idx, p in enumerate(prompt_data.values):
                    progress_bar.update(1)
                    batch.append(p)
                    if (
                        len(batch) == self.seqs_per_request
                        or p_idx == len(prompt_data.values) - 1
                    ):
                        ans_batch = model.generate(
                            batch, [generation_config] * len(batch)
                        )
                        for ans in ans_batch:
                            artifacts_collector.collect(
                                p_idx,
                                p,
                                ans.m_generation_ids[0],
                            )

                        batch.clear()

        return artifacts_collector.build_csv(self.language, self.long_prompt, self.metrics_list)
