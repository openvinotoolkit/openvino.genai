# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import logging
import inspect

import numpy as np
from abc import ABC, abstractmethod

from .utils import patch_awq_for_inference, get_ignore_parameters_flag
from .text_metrics_collection import GenerationResults, Metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    @property
    def produced_fields(self) -> frozenset:
        return frozenset({"prompt_input_ids", "logits"})

    def _score_kl_prompt_tokens_chunked(self, model, current_input_ids):
        tokens = current_input_ids[0]
        ctx_size = min(self.kld_ctx or tokens.numel(), tokens.numel())
        if self.kld_ctx is not None and self.kld_ctx > tokens.numel():
            logger.warning(
                "KLD context size (%s) is larger than the number of tokens (%s). Using token size.",
                self.kld_ctx,
                tokens.numel(),
            )
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
        ctx_size = min(self.kld_ctx or len(tokens), len(tokens))
        if self.kld_ctx is not None and self.kld_ctx > len(tokens):
            logger.warning(
                "KLD context size (%s) is larger than the number of tokens (%s). Using token size.",
                self.kld_ctx,
                len(tokens),
            )
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
