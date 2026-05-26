# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import Any, Union, List, TypedDict, Optional

from .registry import register_evaluator
from .text_evaluator import TextEvaluator
from .whowhat_metrics import TextSimilarity
from .utils import get_ignore_parameters_flag, load_image, fix_phi3_v_eos_token_id
from .inputs_preprocessors import MODEL_TYPE_TO_CLS_MAPPING
from .chat_text_evaluator import _find_common_prefix_length, _get_kv_cache_seq_len, _trim_kv_cache

# Keys in the preprocess_inputs dict that carry visual (image / video) tensors.
# These are stripped when reusing the KV cache for text-only turns because the
# corresponding visual tokens are already encoded in past_key_values.
_VISUAL_KEYS: frozenset = frozenset(
    {
        "pixel_values",
        "images",
        "image_sizes",
        "image_grid_thw",
        "video_grid_thw",
        "pixel_values_videos",
        "image_features",
    }
)


class VisualTextChatInput(TypedDict):
    prompt: str
    images: Optional[List[np.ndarray | Image.Image]]
    videos: Optional[List[np.ndarray | Image.Image]]


def vlm_gen_answer_with_kv_cache(
    model,
    inputs,
    processor,
    tokenizer,
    max_new_tokens,
    pruning_ratio,
    relevance_weight,
):
    """Generate answers for a multi-turn VLM conversation reusing the KV cache.

    On text-only turns the full tokenised history is compared token-by-token with
    cached_tokens (mirroring align_cache_and_history in lm_encoding.cpp) and only
    the new tokens are forwarded to model.generate() with all visual keys removed,
    because the corresponding image tokens are already encoded in past_key_values.

    For turns that introduce new images or videos the full preprocess_inputs are
    used without a cached past_key_values, because splitting pixel_values to match
    only the new image tokens in the suffix is model-specific.  The resulting KV
    cache is still saved so that subsequent text-only turns can benefit from it.

    Falls back silently to full re-encoding when the backend does not expose
    past_key_values (e.g. stateful OVModelForCausalLM).
    """
    if model.config.model_type not in MODEL_TYPE_TO_CLS_MAPPING:
        raise ValueError(
            f"WWB doesn't support models with type '{model.config.model_type}' to evaluation in chat mode."
        )

    inputs_processor = MODEL_TYPE_TO_CLS_MAPPING[model.config.model_type](chat_mode=True)
    answers = []

    past_key_values = None  # KV cache accumulated across turns
    cached_tokens: list = []  # flat token-id list matching what is in past_key_values

    for input_case in inputs:
        preprocess_inputs = inputs_processor.preprocess_inputs(
            input_case["prompt"],
            input_case["images"],
            processor,
            tokenizer,
            config=model.config,
            video=input_case["videos"],
        )

        has_new_visuals = input_case["images"] is not None or input_case["videos"] is not None
        has_input_ids = "input_ids" in preprocess_inputs

        # KV cache is reused only for text-only turns (no new visual tokens injected)
        # when the model exposes input_ids for token-level comparison.
        use_kv_cache = past_key_values is not None and has_input_ids and not has_new_visuals

        if use_kv_cache:
            full_input_ids = preprocess_inputs["input_ids"]
            full_token_list = full_input_ids[0].tolist()
            full_seq_len = full_input_ids.shape[1]

            # --- align cache with new tokenised history (align_cache_and_history) ---
            prefix_len = _find_common_prefix_length(full_token_list, cached_tokens)
            if prefix_len < len(cached_tokens):
                past_key_values = _trim_kv_cache(past_key_values, prefix_len)
            cached_tokens = cached_tokens[:prefix_len]

            # Strip visual keys: image/video tokens are already in the KV cache.
            generate_kwargs = {k: v for k, v in preprocess_inputs.items() if k not in _VISUAL_KEYS}
            generate_kwargs["input_ids"] = full_input_ids[:, prefix_len:]
            generate_kwargs["attention_mask"] = full_input_ids.new_ones(1, full_seq_len)
            generate_kwargs["past_key_values"] = past_key_values
        else:
            generate_kwargs = dict(preprocess_inputs)
            # New visual turn: the accumulated pixel_values span all previous images
            # but new_input_ids would only cover the new image tokens; splitting
            # pixel_values correctly is model-specific, so reset the cache.
            if has_new_visuals:
                past_key_values = None
                cached_tokens = []

        generate_kwargs.update(fix_phi3_v_eos_token_id(model.config.model_type, tokenizer))
        generate_kwargs["do_sample"] = False
        generate_kwargs["max_new_tokens"] = max_new_tokens
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["tokenizer"] = tokenizer
        generate_kwargs.update(get_ignore_parameters_flag())

        output = model.generate(**generate_kwargs)

        gen_input_len = generate_kwargs["input_ids"].shape[1]

        # --- parse output (models may return different types) ---
        if isinstance(output, tuple) and isinstance(output[0], list) and isinstance(output[0][0], str):
            # MiniCPM-o: (decoded_answers, GenerateDecoderOnlyOutput)
            answer_text = output[0][0]
            raw_output = output[1] if len(output) > 1 else None
        elif hasattr(output, "sequences"):
            generated_ids = output.sequences[:, gen_input_len:]
            answer_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            raw_output = output
        else:
            # Plain tensor fallback.
            tokens = output
            gen_input_ids = generate_kwargs["input_ids"]
            if tokens.shape[-1] > gen_input_len and torch.equal(tokens[:, :gen_input_len], gen_input_ids):
                generated_ids = tokens[:, gen_input_len:]
            else:
                generated_ids = tokens
            answer_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            raw_output = None

        # --- update KV cache state for the next turn ---
        new_past = getattr(raw_output, "past_key_values", None) if raw_output is not None else None
        if new_past is not None and has_input_ids:
            past_key_values = new_past
            full_ids_list = preprocess_inputs["input_ids"][0].tolist()
            if hasattr(raw_output, "sequences"):
                generated_ids_list = raw_output.sequences[0, gen_input_len:].tolist()
            else:
                generated_ids_list = tokenizer.encode(answer_text, add_special_tokens=False)
            # The cache holds: all full-history tokens + generated tokens.
            cached_tokens = full_ids_list + generated_ids_list
            actual_cache_len = _get_kv_cache_seq_len(past_key_values)
            if actual_cache_len > 0 and len(cached_tokens) != actual_cache_len:
                cached_tokens = cached_tokens[:actual_cache_len]
        else:
            past_key_values = None
            cached_tokens = []

        inputs_processor.update_chat_history_with_answer(answer_text)
        answers.append(answer_text)

    return answers


@register_evaluator("visual-text-chat")
class ChatVisualTextEvaluator(TextEvaluator):
    CHAT_PROMPTS_FILE = "visualtext_image_chat_prompts.json"

    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        processor: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens=128,
        num_samples=None,
        gen_answer_fn=None,
        pruning_ratio=None,
        relevance_weight=None,
    ) -> None:
        if base_model is None and gt_data is None:
            raise ValueError("Text generation pipeline for evaluation or ground truth data must be defined")

        self.test_data = test_data
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.generation_fn = gen_answer_fn
        self.pruning_ratio = pruning_ratio
        self.relevance_weight = relevance_weight
        self.processor = processor

        self.gt_dir = Path(gt_data or "").parent
        if base_model:
            self.gt_data = self._generate_data(base_model, gen_answer_fn, self.gt_dir / "reference")
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        self.similarity = None
        self.divergency = None
        self.similarity = TextSimilarity(similarity_model_id)
        self.last_cmp = None

    def get_generation_fn(self):
        return self.generation_fn

    def read_data_to_evaluation_scores(self, data):
        text_data = {"answers": [], "prompts": []}

        for path in data["answers"].values:
            with open(path, "r", encoding="utf-8") as f:
                text_data["answers"].append(json.load(f))

        for path in data["prompts"].values:
            with open(path, "r", encoding="utf-8") as f:
                text_data["prompts"].append(json.load(f))

        df = pd.DataFrame(text_data)
        return df

    def score(self, model_or_data, gen_answer_fn=None, output_dir=None, **kwargs):
        result_folder = Path(output_dir or self.gt_dir) / "target"
        if isinstance(model_or_data, str) and Path(model_or_data).exists():
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn, result_folder)
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        # gt_data/predictions contains path to the prompts/answers, let's collect text
        gt_data_text = self.read_data_to_evaluation_scores(self.gt_data)
        predictions_text = self.read_data_to_evaluation_scores(self.predictions)

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(gt_data_text, predictions_text)
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(gt_data_text, predictions_text)
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions_text["prompts"].values
        self.last_cmp["source_model"] = ["\n\n".join(val) for val in gt_data_text["answers"].values]
        self.last_cmp["optimized_model"] = ["\n\n".join(val) for val in predictions_text["answers"].values]
        self.last_cmp = pd.DataFrame(self.last_cmp)
        self.last_cmp.rename(columns={"prompts": "prompt"}, inplace=True)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        if metric in ["SDT", "SDT norm"]:
            res = self.last_cmp.nlargest(top_k, metric)
        else:
            res = self.last_cmp.nsmallest(top_k, metric)

        res = list(row for idx, row in res.iterrows())

        return res

    def prepare_default_data(self, custom_path: str = None) -> List[List[VisualTextChatInput]]:
        from importlib.resources import files

        data_path = custom_path or files("whowhatbench.prompts").joinpath(self.CHAT_PROMPTS_FILE)
        data = []
        with open(data_path) as input_file:
            data = json.load(input_file)

        def_chat_inputs = []
        for chat in data:
            chat_input = []
            for item in chat:
                chat_input.append(
                    {
                        "prompt": item["text"],
                        "images": [load_image(image) for image in item["images"]] if item["images"] else None,
                        "videos": None,
                    }
                )
            def_chat_inputs.append(chat_input)
        return def_chat_inputs

    def _generate_data(self, model, gen_answer_fn=None, result_dir="reference"):
        def default_gen_answer(
            model,
            inputs,
            processor,
            tokenizer,
            max_new_tokens,
            pruning_ratio,
            relevance_weight,
        ):
            if model.config.model_type not in MODEL_TYPE_TO_CLS_MAPPING:
                raise ValueError(
                    f"WWB doesn't support models with type '{model.config.model_type}' to evaluation in chat mode."
                )

            inputs_processor = MODEL_TYPE_TO_CLS_MAPPING[model.config.model_type](chat_mode=True)
            answers = []
            for input_case in inputs:
                preprocess_inputs = inputs_processor.preprocess_inputs(
                    input_case["prompt"],
                    input_case["images"],
                    processor,
                    tokenizer,
                    config=model.config,
                    video=input_case["videos"],
                )
                tokens = model.generate(
                    **preprocess_inputs,
                    **fix_phi3_v_eos_token_id(model.config.model_type, tokenizer),
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    tokenizer=tokenizer,
                    **get_ignore_parameters_flag(),
                )
                if isinstance(tokens, tuple) and isinstance(tokens[0], list) and isinstance(tokens[0][0], str):
                    # Some models return a decoded output, like miniCPM-o
                    # The output tuple has format (<list of decoded outputs without question/prompt>, <GenerateDecoderOnlyOutput>)
                    answer_text = tokens[0][0]
                else:
                    # Some models includes the input_ids in the generated tokens, some - not, so we need to check and remove them if needed
                    inputs_num = preprocess_inputs["input_ids"].shape[-1]
                    if tokens.shape[-1] > inputs_num and torch.equal(
                        tokens[:, :inputs_num], preprocess_inputs["input_ids"]
                    ):
                        answer_tokens = tokens[:, preprocess_inputs["input_ids"].shape[-1] :]
                    else:
                        answer_tokens = tokens
                    answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)[0]

                inputs_processor.update_chat_history_with_answer(answer_text)
                answers.append(answer_text)

            return answers

        gen_answer_fn = gen_answer_fn or vlm_gen_answer_with_kv_cache

        if self.test_data:
            if isinstance(self.test_data, str):
                input_data = self.prepare_default_data(self.test_data)
            elif isinstance(self.test_data, list):
                input_data = self.test_data
        else:
            input_data = self.prepare_default_data()

        answers = []
        prompts_paths = []
        input_data = input_data if self.num_samples is None else input_data[: self.num_samples]

        Path(result_dir).mkdir(parents=True, exist_ok=True)

        prompts_dir = Path(result_dir) / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        inputs: List[VisualTextChatInput]
        for i, inputs in tqdm(
            enumerate(input_data),
            desc="Evaluate pipeline",
        ):
            answer = gen_answer_fn(
                model,
                inputs,
                self.processor,
                self.tokenizer,
                self.max_new_tokens,
                self.pruning_ratio,
                self.relevance_weight,
            )

            result_path = Path(result_dir) / f"chat_vlm_output_{i}.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(answer, f, ensure_ascii=False, indent=4)
            answers.append(result_path)

            prompt_path = Path(prompts_dir) / f"chat_vlm_prompts_{i}.json"
            with open(prompt_path, "w", encoding="utf-8") as f:
                prompts = [input["prompt"] for input in inputs]
                json.dump(prompts, f, ensure_ascii=False, indent=4)
            prompts_paths.append(prompt_path)

        res_data = {"prompts": prompts_paths, "answers": answers}
        df = pd.DataFrame(res_data)

        return df
