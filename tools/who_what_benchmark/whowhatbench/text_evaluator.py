# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import torch
import logging
import numpy as np
import pandas as pd

from enum import Enum
from tqdm import tqdm
from pathlib import Path
from typing import Any, Union
from abc import ABC, abstractmethod
from importlib.resources import files

from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import TextDivergency, TextSimilarity, KLDivergency
from .utils import patch_awq_for_inference, get_ignore_parameters_flag
import inspect
from collections import OrderedDict

PROMPTS_FILE = 'text_prompts.yaml'
LONG_PROMPTS_FILE = 'text_long_prompts.yaml'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Metrics(Enum):
    SIMILARITY = "similarity"
    DIVERGENCY = "divergency"
    KL_DIVERGENCY = "kl_divergency"
    TOKEN_SIMILARITY = "token_similarity"


class MetricStorageKind(Enum):
    CSV = "csv"
    NPY = "npy"
    TEXT_FILE = "text_file"


class GenerationResults:
    def __init__(self, answer_text, prompt_input_ids=None, logits=None, generated_token_ids=None):
        self.answer_text = answer_text
        self.prompt_input_ids = prompt_input_ids
        self.logits = logits
        self.generated_token_ids = generated_token_ids


class ArtifactsSchema:
    def __init__(self, metrics_list, long_prompts=True):
        self.artifacts_schema = OrderedDict([("prompts", MetricStorageKind.CSV)])
        self.output_dir_required = False

        if long_prompts:
            self.artifacts_schema["prompts"] = MetricStorageKind.TEXT_FILE
            self.output_dir_required = True

        if Metrics.SIMILARITY.value in metrics_list or Metrics.DIVERGENCY.value in metrics_list:
            self.artifacts_schema["answers"] = MetricStorageKind.CSV

        if Metrics.TOKEN_SIMILARITY.value in metrics_list:
            self.artifacts_schema["prompt_input_ids"] = MetricStorageKind.NPY
            self.artifacts_schema["generated_token_ids"] = MetricStorageKind.NPY
            self.output_dir_required = True

        if Metrics.KL_DIVERGENCY.value in metrics_list:
            self.artifacts_schema["prompt_input_ids"] = MetricStorageKind.NPY
            self.artifacts_schema["generated_token_ids"] = MetricStorageKind.NPY
            self.artifacts_schema["logits"] = MetricStorageKind.NPY
            self.output_dir_required = True


class ArtifactsManager:
    DEFAULT_ARTIFACT_ROOT = "wwb_output"

    def __init__(self, artifacts_schema: ArtifactsSchema, artifact_root: Path | None):
        self.artifacts_schema = artifacts_schema.artifacts_schema
        self.artifact_root = None
        if artifacts_schema.output_dir_required:
            self.artifact_root = artifact_root or self.DEFAULT_ARTIFACT_ROOT
            self.artifact_root = Path(self.artifact_root)
            if self.artifact_root.exists():
                logger.warning(f"Artifact root {self.artifact_root} already exists. The data will be overwritten.")
            else:
                self.artifact_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"All artifacts will be stored to {self.artifact_root}.")

        self.artifacts = {name: [] for name in self.artifacts_schema}

    def collect(self, sample_idx: int, prompt: str, output: GenerationResults):
        for name, storage_kind in self.artifacts_schema.items():
            if name == "prompts":
                value = prompt
            elif name == "answers":
                value = output.answer_text
            else:
                value = getattr(output, name, None)

            if value is not None:
                self.artifacts[name].append(self.store(value, sample_idx, storage_kind, name))

    def collect_metadata_for_generation(self, paths_to_data=None):
        if not paths_to_data:
            return
        data = []
        for gen_token_np_path in paths_to_data:
            data.append(np.load(gen_token_np_path))
        return data

    def store(self, value, sample_idx, storage_kind, name):
        if MetricStorageKind.CSV == storage_kind:
            return value
        if MetricStorageKind.NPY == storage_kind:
            sample_dir = self.artifact_root / f"sample_{sample_idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = sample_dir / f"{name}.npy"
            tensor = value.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                # NumPy has no native bfloat16 support, so it must be upcast before conversion
                tensor = tensor.float()
            np.save(artifact_path, tensor.numpy())
            return str(artifact_path)
        if MetricStorageKind.TEXT_FILE == storage_kind:
            sample_dir = self.artifact_root / f"sample_{sample_idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = sample_dir / f"{name}_{sample_idx}.txt"
            artifact_path.write_text(value, encoding="utf-8")
            return str(artifact_path)
        raise ValueError(f"Unsupported storage kind: {storage_kind}")

    def build_csv(self, language: str, long_prompt: bool, metrcis_list: list = None):
        res_data = {"prompts": list(self.artifacts["prompts"])}
        for key in [
            "logits",
            "prompt_input_ids",
            "generated_token_ids",
            "answers",
        ]:
            if self.artifacts.get(key):
                res_data_key = key if self.artifacts_schema[key] == MetricStorageKind.CSV else f"{key}_path"
                res_data[res_data_key] = self.artifacts[key]

        df = pd.DataFrame(res_data)
        df["language"] = language
        df["prompt_length_type"] = "long" if long_prompt else "short"
        df["metrics"] = ";".join(metrcis_list) if metrcis_list else ""
        return df


class BaseGenerationStrategy(ABC):
    def __init__(self, metrcis_list: list = None):
        self.metrcis_list = metrcis_list

    @property
    def produced_fields(self) -> frozenset:
        fields = {"answer_text"}
        if self.metrcis_list and Metrics.TOKEN_SIMILARITY.value in self.metrcis_list:
            fields.update({"prompt_input_ids", "generated_token_ids"})
        return frozenset(fields)

    @abstractmethod
    def run_generation(
        self,
        model,
        tokenizer,
        question,
        reference_tokens,
        max_new_tokens,
        skip_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        pass


class LlamaCPPSelfSufficientGeneration(BaseGenerationStrategy):
    def __init__(self, metrcis_list: list = None):
        super().__init__(metrcis_list)

    def run_generation(
        self,
        model,
        tokenizer,
        question,
        reference_tokens,
        max_new_tokens,
        skip_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        output_text = None
        if use_chat_template:
            output = model.create_chat_completion(
                messages=[{"role": "user", "content": question}], max_tokens=max_new_tokens, temperature=0.0
            )
            output_text = output["choices"][0]["message"]["content"]
        else:
            output = model(question, max_tokens=max_new_tokens, echo=False, temperature=0.0)
            output_text = output["choices"][0]["text"]

        return GenerationResults(answer_text=output_text)


class GenAISelfSufficientGeneration(BaseGenerationStrategy):
    def __init__(self, metrcis_list: list = None):
        super().__init__(metrcis_list)

    def run_generation(
        self,
        model,
        tokenizer,
        question,
        reference_tokens,
        max_new_tokens,
        skip_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        if Metrics.TOKEN_SIMILARITY.value in self.metrcis_list:
            gen_fn = self.genai_generation_encoded
        else:
            gen_fn = self.genai_generation

        return gen_fn(
            model,
            tokenizer,
            question,
            reference_tokens,
            max_new_tokens,
            skip_question,
            use_chat_template=use_chat_template,
            empty_adapters=empty_adapters,
            num_assistant_tokens=num_assistant_tokens,
            assistant_confidence_threshold=assistant_confidence_threshold,
            generation_config_extra=generation_config_extra,
        )

    def genai_generation(
        self,
        model,
        tokenizer,
        question,
        reference_tokens,
        max_new_tokens,
        skip_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        kwargs = {}
        if empty_adapters:
            import openvino_genai

            kwargs["adapters"] = openvino_genai.AdapterConfig()
        if generation_config_extra:
            kwargs.update(generation_config_extra)

        output = model.generate(
            question,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            apply_chat_template=use_chat_template,
            num_assistant_tokens=num_assistant_tokens,
            assistant_confidence_threshold=assistant_confidence_threshold,
            **kwargs,
        )

        return GenerationResults(answer_text=output)

    def genai_generation_encoded(
        self,
        model,
        tokenizer,
        question,
        reference_tokens,
        max_new_tokens,
        skip_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        kwargs = {}
        if empty_adapters:
            import openvino_genai

            kwargs["adapters"] = openvino_genai.AdapterConfig()
        if generation_config_extra:
            kwargs.update(generation_config_extra)

        tokenizer = model.get_tokenizer()

        prompt = question
        if use_chat_template:
            import openvino_genai

            chat_history = openvino_genai.ChatHistory()
            chat_history.append({"role": "user", "content": prompt})
            prompt = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True)

        tokenized_input = tokenizer.encode(prompt)
        output = model.generate(
            tokenized_input,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            apply_chat_template=use_chat_template,
            num_assistant_tokens=num_assistant_tokens,
            assistant_confidence_threshold=assistant_confidence_threshold,
            **kwargs,
        )

        return GenerationResults(
            answer_text=tokenizer.decode(output.tokens[0]),
            prompt_input_ids=torch.from_numpy(tokenized_input.input_ids.data),
            generated_token_ids=torch.Tensor(output.tokens[0]),
        )


class SelfSufficientGenerationStrategy(BaseGenerationStrategy):
    def __init__(self, metrcis_list: list = None):
        super().__init__(metrcis_list)

    def run_generation(
        self,
        model,
        tokenizer,
        prompt,
        reference_tokens,
        max_new_tokens,
        crop_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        is_awq = getattr(model, "is_awq", None) is not None
        device = "cpu"
        if hasattr(model, "device"):
            device = model.device

        if use_chat_template:
            message = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                message, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

        if "token_type_ids" in inputs and "token_type_ids" not in list(
            inspect.signature(model.forward).parameters.keys()
        ):
            inputs.pop("token_type_ids")

        if is_awq:
            with patch_awq_for_inference(is_awq):
                tokens = model.generate(
                    **inputs, do_sample=False, max_new_tokens=max_new_tokens, **get_ignore_parameters_flag()
                )
        else:
            tokens = model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens, **get_ignore_parameters_flag()
            )
        if crop_question:
            tokens = tokens[:, inputs["input_ids"].shape[-1] :]

        return GenerationResults(
            answer_text=tokenizer.batch_decode(tokens, skip_special_tokens=True)[0],
            prompt_input_ids=inputs["input_ids"],
            generated_token_ids=tokens,
        )


class ReferenceBaseGenerationStrategy(BaseGenerationStrategy):
    def __init__(self, metrcis_list: list = None, kld_ctx: int = None, kld_chunk: int = None):
        super().__init__(metrcis_list)
        self.kld_ctx = kld_ctx
        self.kld_chunk = kld_chunk
        self.keep_text_answer = True

    @property
    def produced_fields(self) -> frozenset:
        return frozenset({"prompt_input_ids", "logits"})

    def _score_kl_prompt_tokens_chunked(self, model, current_input_ids):
        tokens = current_input_ids[0]
        ctx_size = self.kld_ctx or tokens.numel()
        chunk_size = self.kld_chunk or 1

        collected_logits = []
        # collected_eval_tokens = []
        for i, start in enumerate(range(0, tokens.numel(), ctx_size)):
            if start + ctx_size > tokens.numel() or i >= chunk_size:
                break

            end = start + ctx_size
            chunk_len = end - start

            chunk_input_ids = current_input_ids[:, start:end]

            model_inputs = {
                "input_ids": chunk_input_ids,
                "return_dict": True,
                "attention_mask": torch.ones(chunk_input_ids.shape, dtype=torch.long, device=chunk_input_ids.device),
            }
            outputs = model(**model_inputs)

            logits_start = chunk_len // 2
            step_logits = outputs.logits[:, logits_start:, :].to(dtype=torch.float32).squeeze(0)
            collected_logits.append(step_logits)
            logger.debug(
                "KL DIVERGENCY TOKEN LOGITS chunk idx %s, start idx %s, end idx %s, chunk_len %s, "
                "step_logits shape %s, logit start idx %s",
                i,
                start,
                end,
                chunk_len,
                step_logits.shape,
                logits_start,
            )

        if not collected_logits:
            logger.warning(
                "No logits were collected for the given input tokens, input token size: %s, ctx_size %s.",
                tokens.shape,
                ctx_size,
            )
            return torch.tensor([], dtype=torch.float32)
        return torch.cat(collected_logits, dim=0)

    def run_generation(
        self,
        model,
        tokenizer,
        prompt,
        reference_tokens,
        max_new_tokens,
        crop_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        is_awq = getattr(model, "is_awq", None) is not None
        device = "cpu"
        if hasattr(model, "device"):
            device = model.device

        prompt_input_ids = None
        if reference_tokens is not None:
            prompt_input_ids = torch.from_numpy(reference_tokens).to(dtype=torch.long)
        else:
            if use_chat_template:
                message = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(
                    message,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                ).to(device)
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_input_ids = inputs["input_ids"]

        if is_awq:
            with patch_awq_for_inference(is_awq):
                logits_tensor = self._score_kl_prompt_tokens_chunked(model, prompt_input_ids)
        else:
            logits_tensor = self._score_kl_prompt_tokens_chunked(model, prompt_input_ids)

        return GenerationResults(
            answer_text="",
            prompt_input_ids=prompt_input_ids,
            generated_token_ids=None,
            logits=logits_tensor,
        )


class LlamaCPPReferenceBaseGenerationStrategy(ReferenceBaseGenerationStrategy):
    def _score_kl_prompt_tokens_chunked(self, model, current_input_ids):
        tokens = current_input_ids[0].tolist() if current_input_ids.dim() > 1 else current_input_ids.tolist()
        ctx_size = self.kld_ctx or len(tokens)
        chunk_size = self.kld_chunk or 1

        collected_logits = []
        for i, start in enumerate(range(0, len(tokens), ctx_size)):
            if start + ctx_size > len(tokens) or i >= chunk_size:
                break

            end = min(start + ctx_size, len(tokens))
            chunk_tokens = tokens[start:end]
            model.reset()
            model.eval(chunk_tokens)
            chunk_len = len(chunk_tokens)
            new_scores = model.scores[model.n_tokens - chunk_len : model.n_tokens]

            logits_start = chunk_len // 2
            step_logits = torch.from_numpy(np.array(new_scores[logits_start:], dtype=np.float32))
            collected_logits.append(step_logits)
            logger.debug(
                "KL DIVERGENCY TOKEN LOGITS chunk idx %s, start idx %s, end idx %s, chunk_len %s, "
                "step_logits shape %s, logit start idx %s",
                i,
                start,
                end,
                chunk_len,
                step_logits.shape,
                logits_start,
            )

        return torch.cat(collected_logits, dim=0)

    def run_generation(
        self,
        model,
        tokenizer,
        prompt,
        reference_tokens,
        max_new_tokens,
        crop_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        prompt_input_ids = None
        if reference_tokens is not None:
            prompt_input_ids = torch.from_numpy(reference_tokens).to(dtype=torch.long)
        else:
            prompt_ids = model.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)
            prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long)

        logits_tensor = self._score_kl_prompt_tokens_chunked(model, prompt_input_ids)

        return GenerationResults(
            answer_text="",
            prompt_input_ids=prompt_input_ids,
            generated_token_ids=None,
            logits=logits_tensor,
        )


class CompositeGenerationStrategy(BaseGenerationStrategy):
    """Runs several strategies one after another and merges the fields each of them is responsible for."""

    def __init__(self, strategies: list, metrcis_list: list = None):
        super().__init__(metrcis_list)
        field_owners = {}
        for strategy in strategies:
            for field in strategy.produced_fields:
                if field not in field_owners:
                    field_owners[field] = strategy
        self.strategies = strategies

    @property
    def produced_fields(self) -> frozenset:
        return frozenset().union(*(strategy.produced_fields for strategy in self.strategies))

    def run_generation(
        self,
        model,
        tokenizer,
        question,
        reference_tokens,
        max_new_tokens,
        skip_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        merged = GenerationResults(answer_text="")
        for strategy in self.strategies:
            output = strategy.run_generation(
                model,
                tokenizer,
                question,
                reference_tokens,
                max_new_tokens,
                skip_question,
                use_chat_template=use_chat_template,
                empty_adapters=empty_adapters,
                num_assistant_tokens=num_assistant_tokens,
                assistant_confidence_threshold=assistant_confidence_threshold,
                generation_config_extra=generation_config_extra,
            )
            for field in strategy.produced_fields:
                setattr(merged, field, getattr(output, field))
        return merged


class GenerationStrategy:
    TEXT_ANSWER_METRICS = (Metrics.SIMILARITY.value, Metrics.DIVERGENCY.value, Metrics.TOKEN_SIMILARITY.value)

    @staticmethod
    def create(metrics_list, is_genai, is_llamacpp, kld_ctx=None, kld_chunk=None) -> BaseGenerationStrategy:
        is_kl_requested = Metrics.KL_DIVERGENCY.value in metrics_list
        is_answer_requested = any(metric in metrics_list for metric in GenerationStrategy.TEXT_ANSWER_METRICS)

        strategies = []
        if is_answer_requested:
            strategies.append(GenerationStrategy._create_self_sufficient(metrics_list, is_genai, is_llamacpp))
        if is_kl_requested:
            strategies.append(
                GenerationStrategy._create_reference(metrics_list, is_genai, is_llamacpp, kld_ctx, kld_chunk)
            )

        if len(strategies) == 1:
            return strategies[0]
        return CompositeGenerationStrategy(strategies, metrics_list)

    @staticmethod
    def _create_self_sufficient(metrics_list, is_genai, is_llamacpp) -> BaseGenerationStrategy:
        if is_genai:
            return GenAISelfSufficientGeneration(metrics_list)
        if is_llamacpp:
            return LlamaCPPSelfSufficientGeneration(metrics_list)
        return SelfSufficientGenerationStrategy(metrics_list)

    @staticmethod
    def _create_reference(metrics_list, is_genai, is_llamacpp, kld_ctx, kld_chunk) -> BaseGenerationStrategy:
        if is_genai:
            raise ValueError("KL divergency is not supported for GenAI models.")
        if is_llamacpp:
            return LlamaCPPReferenceBaseGenerationStrategy(metrics_list, kld_ctx, kld_chunk)
        return ReferenceBaseGenerationStrategy(metrics_list, kld_ctx, kld_chunk)


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
        # self.kl_input_tokens = self._load_kl_input_tokens(kl_input_tokens)
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
