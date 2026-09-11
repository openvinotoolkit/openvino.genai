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
    def __init__(
        self, answer_text, prompt_input_ids=None, topk_log_probs=None, topk_token_ids=None, generated_token_ids=None
    ):
        self.answer_text = answer_text
        self.prompt_input_ids = prompt_input_ids
        self.topk_log_probs = topk_log_probs
        self.topk_token_ids = topk_token_ids
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
            self.artifacts_schema["generated_token_ids"] = MetricStorageKind.NPY
            self.artifacts_schema["topk_log_probs"] = MetricStorageKind.NPY
            self.artifacts_schema["topk_token_ids"] = MetricStorageKind.NPY
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
            "topk_log_probs",
            "topk_token_ids",
            "prompt_input_ids",
            "generated_token_ids",
            "topk_token_ids",
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

    @abstractmethod
    def run_generation(
        self,
        model,
        tokenizer,
        question,
        reference_tokens,
        topk_token_ids,
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
        topk_token_ids,
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
        topk_token_ids,
        max_new_tokens,
        skip_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        if Metrics.TOKEN_SIMILARITY in self.metrcis_list or Metrics.KL_DIVERGENCY in self.metrcis_list:
            gen_fn = self.genai_generation_encoded
        else:
            gen_fn = self.genai_generation

        return gen_fn(
            model,
            tokenizer,
            question,
            reference_tokens,
            topk_token_ids,
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
        topk_token_ids,
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
        topk_token_ids,
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

        tokenized_input = tokenizer.encode(question)
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
            answer_text=tokenizer.decode(output.tokens)[0],
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
        topk_token_ids,
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
    DEF_TOP_N = 16

    def __init__(self, metrcis_list: list = None):
        super().__init__(metrcis_list)

    def collect_topk_from_step_logits(self, step_logits):
        log_probs = torch.log_softmax(step_logits, dim=-1)
        top_k = min(self.DEF_TOP_N, log_probs.shape[-1])
        topk_log_probs, topk_ids = torch.topk(log_probs, k=top_k, dim=-1)
        return torch.softmax(topk_log_probs, dim=-1), topk_ids

    def collect_topk_from_logits(self, output):
        all_topk_probs = []
        all_topk_ids = []
        logits_per_step = output.logits if output.logits is not None else []
        logger.debug(f"BASE MODEL TOPK LOG PROBS")
        for i, step_logits in enumerate(logits_per_step):
            topk_probs, topk_ids = self.collect_topk_from_step_logits(step_logits)
            logger.debug(f"Step {i}:\n topk_probs={topk_probs}\n, topk_ids={topk_ids}")
            all_topk_probs.append(topk_probs)
            all_topk_ids.append(topk_ids)
        return all_topk_probs, all_topk_ids

    def collect_probs_at_reference_ids_from_step_logits(self, step_logits, reference_topk_ids):
        log_probs = torch.log_softmax(step_logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, dim=-1, index=reference_topk_ids)
        return torch.softmax(gathered_log_probs, dim=-1)

    def run_base_mode(
        self, model, tokenizer, inputs, reference_tokens, reference_topk_ids, max_new_tokens, crop_question
    ):
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            **get_ignore_parameters_flag(),
            return_dict_in_generate=True,
            output_logits=True,
        )

        all_topk_probs, all_topk_ids = self.collect_topk_from_logits(output)

        sequences = output.sequences
        topk_probs_tensor = torch.stack(all_topk_probs, dim=0).squeeze(1)
        topk_ids_tensor = torch.stack(all_topk_ids, dim=0).squeeze(1)
        generated_token_ids = sequences
        if crop_question:
            generated_token_ids = sequences[:, inputs["input_ids"].shape[-1] :]

        return GenerationResults(
            answer_text=tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0],
            prompt_input_ids=inputs["input_ids"][0],
            generated_token_ids=generated_token_ids,
            topk_log_probs=topk_probs_tensor,
            topk_token_ids=topk_ids_tensor,
        )

    def run_target_mode(
        self, model, tokenizer, inputs, reference_tokens, reference_topk_ids, max_new_tokens, crop_question
    ):
        ref_tokens = torch.as_tensor(reference_tokens, dtype=torch.long, device=inputs["input_ids"].device).view(-1)
        steps = min(max_new_tokens, len(reference_tokens))
        ref_tokens = ref_tokens[: steps + 1]

        all_topk_probs = []
        all_topk_ids = []

        current_input_ids = inputs["input_ids"]
        current_attention_mask = inputs.get("attention_mask", None)
        past_key_values = None

        logger.debug(f"TARGET MODEL TOPK LOG PROBS")
        for step_idx in range(steps):
            model_inputs = {
                "input_ids": current_input_ids,
                "use_cache": True,
                "return_dict": True,
            }
            if current_attention_mask is not None:
                model_inputs["attention_mask"] = current_attention_mask
            if past_key_values is not None:
                if "transformers" in str(type(model)):
                    model_inputs["past_key_values"] = past_key_values
                else:
                    # for optimum-intel stateful model past_key_values are not used explicitly, instead they are handled inside the model
                    # to avoid taking into account past_key_values, will set it to [None]
                    model_inputs["past_key_values"] = [None]

            outputs = model(**model_inputs)
            step_logits = outputs.logits[:, -1, :]
            step_reference_topk_ids = torch.as_tensor(
                reference_topk_ids[step_idx], dtype=torch.long, device=step_logits.device
            ).unsqueeze(0)
            topk_probs = self.collect_probs_at_reference_ids_from_step_logits(step_logits, step_reference_topk_ids)
            all_topk_probs.append(topk_probs)
            all_topk_ids.append(step_reference_topk_ids)

            logger.debug(
                f"Step {step_idx}:\n current input ids={current_input_ids} \n topk_probs={topk_probs}\n, topk_ids={step_reference_topk_ids}"
            )

            forced_token = ref_tokens[step_idx].view(1, 1)
            current_input_ids = forced_token
            if current_attention_mask is not None:
                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        torch.ones(
                            (current_attention_mask.shape[0], 1),
                            dtype=current_attention_mask.dtype,
                            device=current_attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
            past_key_values = getattr(outputs, "past_key_values", None)

        topk_probs_tensor = torch.stack(all_topk_probs, dim=0).squeeze(1)
        topk_topk_ids = torch.stack(all_topk_ids, dim=0).squeeze(1)

        return GenerationResults(
            answer_text="",
            prompt_input_ids=inputs["input_ids"][0],
            generated_token_ids=None,
            topk_log_probs=topk_probs_tensor,
            topk_token_ids=topk_topk_ids,
        )

    def run_generation(
        self,
        model,
        tokenizer,
        prompt,
        reference_tokens,
        reference_topk_ids,
        max_new_tokens,
        crop_question,
        use_chat_template=False,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        generation_config_extra=None,
    ):
        _ = generation_config_extra
        is_awq = getattr(model, "is_awq", None) is not None
        device = "cpu"
        if hasattr(model, "device"):
            device = model.device

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

        if "token_type_ids" in inputs and "token_type_ids" not in list(
            inspect.signature(model.forward).parameters.keys()
        ):
            inputs.pop("token_type_ids")

        gen_result = None
        run_generation_fn = self.run_base_mode if reference_tokens is None else self.run_target_mode
        if is_awq:
            with patch_awq_for_inference(is_awq):
                gen_result = run_generation_fn(
                    model, tokenizer, inputs, reference_tokens, reference_topk_ids, max_new_tokens, crop_question
                )
        else:
            gen_result = run_generation_fn(
                model, tokenizer, inputs, reference_tokens, reference_topk_ids, max_new_tokens, crop_question
            )

        return gen_result


class GenerationStrategy:
    @staticmethod
    def create(metrics_list, is_genai, is_llamacpp) -> BaseGenerationStrategy:
        if Metrics.KL_DIVERGENCY.value in metrics_list:
            if is_genai or is_llamacpp:
                raise ValueError("KL divergency is not supported for GenAI models or LLamaCPP backend.")
            return ReferenceBaseGenerationStrategy(metrics_list)
        if Metrics.TOKEN_SIMILARITY.value in metrics_list and (is_genai or is_llamacpp):
            raise ValueError("Token similarity is not supported for GenAI models or LLamaCPP backend.")
        if is_genai:
            return GenAISelfSufficientGeneration(metrics_list)
        elif is_llamacpp:
            return LlamaCPPSelfSufficientGeneration(metrics_list)
        return SelfSufficientGenerationStrategy(metrics_list)


@register_evaluator("text")
class TextEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
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
        metrics_list: list = [Metrics.SIMILARITY.value],
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
        # if self.metrics and self.metrics not in self.metrics_list:
        #     if not self.metrics_list:
        #         self.metrics_list = []
        #     self.metrics_list.append(self.metrics)

        self.is_genai = is_genai
        self.is_llamacpp = is_llamacpp
        self.generation_strategy = GenerationStrategy.create(self.metrics_list, self.is_genai, self.is_llamacpp)
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

            if not self.metrics_list and "metrics_list" in self.gt_data.columns:
                self.metrics_list = self.gt_data["metrics_list"].values[0].split(";")

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
            generated_token_ids_path = None
            topk_token_ids_paths = None
            if isinstance(self.gt_data, pd.DataFrame) and "generated_token_ids_path" in self.gt_data.columns:
                generated_token_ids_path = []
                topk_token_ids_paths = []
                for _, row in self.gt_data.iterrows():
                    if isinstance(row.get("generated_token_ids_path"), str):
                        generated_token_ids_path.append(row["generated_token_ids_path"])
                    if isinstance(row.get("topk_token_ids_path"), str):
                        topk_token_ids_paths.append(row["topk_token_ids_path"])

            predictions = self._generate_data(
                model_or_data,
                gen_answer_fn,
                self.generation_config,
                generated_token_ids_path=generated_token_ids_path,
                topk_token_ids=topk_token_ids_paths,
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
            kl_metric = KLDivergency()
            kl_metric_dict, kl_metric_per_question = kl_metric.evaluate(self.gt_data, predictions)
            all_metrics.update(kl_metric_dict)
            all_metrics_per_prompt.update(kl_metric_per_question)

        if Metrics.TOKEN_SIMILARITY.value in self.metrics_list:
            # to be done
            pass

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
        generated_token_ids_path=None,
        topk_token_ids=None,
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
        generated_token_ids = artifacts_collector.collect_metadata_for_generation(generated_token_ids_path)
        topk_token_ids = artifacts_collector.collect_metadata_for_generation(topk_token_ids)

        if generation_config is None:
            extra_kwargs = (
                {"generation_config_extra": self.generation_config_extra} if self.generation_config_extra else {}
            )
            for sample_idx, p in enumerate(tqdm(prompts, desc="Evaluate pipeline")):
                reference_tokens = None
                if generated_token_ids is not None and sample_idx < len(generated_token_ids):
                    reference_tokens = generated_token_ids[sample_idx][0].tolist()
                reference_topk_ids = None
                if topk_token_ids is not None and sample_idx < len(topk_token_ids):
                    reference_topk_ids = topk_token_ids[sample_idx]

                output = gen_answer_fn(
                    model,
                    self.tokenizer,
                    p,
                    reference_tokens,
                    reference_topk_ids,
                    self.max_new_tokens,
                    self._crop_question,
                    self.use_chat_template,
                    self.empty_adapters,
                    self.num_assistant_tokens,
                    self.assistant_confidence_threshold,
                    **extra_kwargs,
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
