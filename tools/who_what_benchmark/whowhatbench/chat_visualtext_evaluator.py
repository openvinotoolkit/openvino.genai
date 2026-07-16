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
from .utils import OMNI_MODEL_TYPES, get_ignore_parameters_flag, load_image, fix_phi3_v_eos_token_id
from .inputs_preprocessors import MODEL_TYPE_TO_CLS_MAPPING
from .chat_utils import find_common_prefix_length, get_kv_cache_seq_len, trim_kv_cache, get_kv_axes_pos

from transformers import __version__
from packaging.version import Version

TRANSFORMERS_VERSION = Version(__version__)


class VisualTextChatInput(TypedDict):
    prompt: str
    images: Optional[List[np.ndarray | Image.Image]]
    videos: Optional[List[np.ndarray | Image.Image]]


full_chat_types = [
    "qwen2_vl",
    "qwen2_5_vl",
    "llava",
    "phi4mm",
    "qwen3_vl",
    "qwen3_5",
    "qwen3_5_moe",
    "llava_next",
    "llava-qwen2",
    "qwen3_omni",
    "qwen3_omni_moe",
]


def default_gen_answer(
    model,
    inputs,
    processor,
    tokenizer,
    max_new_tokens,
    pruning_ratio,
    relevance_weight,
    kv_axes_pos,
    crop_question=False,
    full_chat=False,
):
    if model.config.model_type not in MODEL_TYPE_TO_CLS_MAPPING:
        raise ValueError(
            f"WWB doesn't support models with type '{model.config.model_type}' to evaluation in chat mode."
        )

    inputs_processor = MODEL_TYPE_TO_CLS_MAPPING[model.config.model_type](chat_mode=True, model=model)
    answers = []

    past_key_values = None
    tokenized_history: list = []
    for input_case in inputs:
        preprocess_inputs = inputs_processor.preprocess_inputs(
            input_case["prompt"],
            input_case["images"],
            processor,
            tokenizer,
            config=model.config,
            video=input_case["videos"],
        )

        full_input_ids = preprocess_inputs["input_ids"]
        full_token_list = full_input_ids[0].tolist()

        if not full_chat and len(tokenized_history) > 0:
            prefix_len = find_common_prefix_length(full_token_list, tokenized_history, inputs_processor)
            if prefix_len < len(tokenized_history):
                language_model = model.language_model if hasattr(model, "language_model") else model
                past_key_values = trim_kv_cache(language_model, past_key_values, prefix_len, kv_axes_pos)
            tokenized_history = tokenized_history[:prefix_len]
        else:
            prefix_len = 0
            past_key_values = None

        generate_kwargs = {}
        if past_key_values is not None:
            preprocess_inputs = inputs_processor.align_inputs_with_cache(
                model, preprocess_inputs, full_input_ids, prefix_len
            )

            if "transformers" in str(type(model)):
                generate_kwargs["past_key_values"] = past_key_values
            else:
                # for optimum-intel stateful model past_key_values are not used explicitly, instead they are handled inside the model
                # to avoid taking into account past_key_values, will set it to [None]
                states = model.language_model.request.query_state()
                shape = None
                for state in states:
                    old_tensor = state.state
                    # [BATCH_SIZE, num_kv_heads, seq_len, head_size]
                    data = np.array(old_tensor.data)
                    shape = data.shape
                    break

                if prefix_len > 0:
                    fake_past_key_values = ((np.ones(shape, dtype=data.dtype), np.ones(shape, dtype=data.dtype)),)
                    if TRANSFORMERS_VERSION > Version("5.2.0"):
                        from transformers.cache_utils import DynamicCache

                        fake_layer = (
                            torch.from_numpy(np.ones(shape, dtype=data.dtype)),
                            torch.from_numpy(np.ones(shape, dtype=data.dtype)),
                        )
                        fake_past_key_values = DynamicCache(ddp_cache_data=(fake_layer,))
                    generate_kwargs["past_key_values"] = fake_past_key_values
                else:
                    generate_kwargs["past_key_values"] = None

        if model.config.model_type not in ["internvl_chat"]:
            generate_kwargs["use_cache"] = True
        if model.config.model_type not in ["phi4mm"]:
            generate_kwargs["tokenizer"] = tokenizer

        is_optimum_ov = "openvino" in str(type(model)).lower()
        is_hf_omni = model.config.model_type in OMNI_MODEL_TYPES and not is_optimum_ov
        if is_hf_omni:
            # Qwen3-Omni's generate() bounds text output via thinker_max_new_tokens
            # and would synthesize audio unless return_audio is disabled.
            generate_kwargs["thinker_max_new_tokens"] = max_new_tokens
            generate_kwargs["return_audio"] = False
        else:
            generate_kwargs["max_new_tokens"] = max_new_tokens

        output = model.generate(
            **preprocess_inputs,
            **fix_phi3_v_eos_token_id(model.config.model_type, tokenizer),
            do_sample=False,
            **get_ignore_parameters_flag(),
            return_dict_in_generate=True,
            **generate_kwargs,
        )
        if is_hf_omni and isinstance(output, tuple):
            # (text_ids, audio) is returned when return_audio=false on transformers <= 4.57
            output = output[0]
        if isinstance(output, tuple) and isinstance(output[0], list) and isinstance(output[0][0], str):
            # Some models return a decoded output, like miniCPM-o
            # The output tuple has format (<list of decoded outputs without question/prompt>, <GenerateDecoderOnlyOutput>)
            output = output[1]

        tokens = output.sequences
        if crop_question:
            input_ids_len = full_input_ids.shape[-1]
            tokens = tokens[:, input_ids_len:]

        new_past_key_values = getattr(output, "past_key_values", None) if output is not None else None
        if new_past_key_values is not None and not full_chat:
            past_key_values = new_past_key_values
            generated_ids_list = tokens[0].tolist()
            tokenized_history = full_token_list + generated_ids_list
            actual_cache_len = get_kv_cache_seq_len(model, past_key_values, tokenized_history)
            if actual_cache_len > 0 and len(tokenized_history) != actual_cache_len:
                tokenized_history = tokenized_history[:actual_cache_len]
        else:
            past_key_values = None
            tokenized_history = []

        answer_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
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
        crop_question=True,
        device="CPU",
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
        self._crop_question = crop_question
        self._full_chat = device == "NPU"

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
        gen_answer_fn = gen_answer_fn or default_gen_answer

        # applicable for ov model only, 2 is just a default value
        kv_axes_pos = 2
        if "optimum" in str(type(model)):
            language_model = model.language_model if hasattr(model, "language_model") else model
            kv_axes_pos = get_kv_axes_pos(language_model.model)

        _full_chat = (
            self._full_chat
            or "transformers" in str(type(model))
            or (model and model.config.model_type in full_chat_types)
        )

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
                kv_axes_pos,
                self._crop_question,
                _full_chat,
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
