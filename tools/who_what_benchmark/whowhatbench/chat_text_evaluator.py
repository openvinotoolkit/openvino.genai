# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

import os
import yaml
import json
import pandas as pd
from tqdm import tqdm
from importlib.resources import files
from .registry import register_evaluator
from .text_evaluator import TextEvaluator
from .whowhat_metrics import TextDivergency, TextSimilarity
from .utils import patch_awq_for_inference, get_ignore_parameters_flag
from .chat_utils import find_common_prefix_length, trim_kv_cache, get_kv_cache_seq_len, get_kv_axes_pos
import inspect


def default_gen_answer(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    _empty_adapters=False,
    _num_assistant_tokens=0,
    _assistant_confidence_threshold=0.0,
    full_chat=False,
    kv_axes_pos=2,
    generation_config_extra=None,
):
    is_awq = getattr(model, "is_awq", None) is not None
    device = "cpu"
    if hasattr(model, "device"):
        device = model.device

    chat_history = []
    answers = []

    # transformers manage kv_cache via past_key_values
    # for optimum-intel statefull model ((),)
    past_key_values = None
    tokenized_history: list = []
    for prompt in prompts:
        chat_history.append({"role": "user", "content": prompt})

        full_tokenized_chat = tokenizer.apply_chat_template(
            chat_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        if "token_type_ids" in full_tokenized_chat and "token_type_ids" not in list(
            inspect.signature(model.forward).parameters.keys()
        ):
            full_tokenized_chat.pop("token_type_ids")

        full_input_ids = full_tokenized_chat["input_ids"]
        full_input_ids_list = full_input_ids[0].tolist()

        if len(tokenized_history) > 0 and not full_chat:
            prefix_len = find_common_prefix_length(full_input_ids_list, tokenized_history)
            if prefix_len < len(tokenized_history):
                past_key_values = trim_kv_cache(model, past_key_values, prefix_len, kv_axes_pos)
            tokenized_history = tokenized_history[:prefix_len]
        else:
            prefix_len = 0

        new_input_ids = full_input_ids
        attention_mask = full_input_ids.new_ones(1, full_input_ids.shape[1])

        generate_kwargs = dict(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            **get_ignore_parameters_flag(),
            use_cache=True,
        )

        if past_key_values is not None:
            if "transformers" in str(type(model)):
                generate_kwargs["past_key_values"] = past_key_values
            else:
                # for optimum-intel stateful model past_key_values are not used explicitly, instead they are handled inside the model
                # to avoid taking into account past_key_values, will set it to [None]
                generate_kwargs["past_key_values"] = [None]

        if is_awq:
            with patch_awq_for_inference(is_awq):
                output = model.generate(**generate_kwargs)
        else:
            output = model.generate(**generate_kwargs)

        # output.sequences shape: [1, new_input_len + generated_len]
        generated_ids = output.sequences[:, new_input_ids.shape[1] :]
        answer_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Update the KV cache state for the next turn.
        new_past_key_values = getattr(output, "past_key_values", None)
        if new_past_key_values is not None and not full_chat:
            past_key_values = new_past_key_values
            tokenized_history = full_input_ids_list + generated_ids[0].tolist()
            actual_cache_len = get_kv_cache_seq_len(model, past_key_values, tokenized_history)
            if actual_cache_len > 0 and len(tokenized_history) != actual_cache_len:
                tokenized_history = tokenized_history[:actual_cache_len]
        else:
            past_key_values = None
            tokenized_history = []

        chat_history.append({"role": "assistant", "content": answer_text[0]})
        answers.append(answer_text[0])

    return answers


@register_evaluator("text-chat")
class ChatTextEvaluator(TextEvaluator):
    CHAT_PROMPTS_FILE = "text_chat_prompts.yaml"

    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens=128,
        num_samples=None,
        gen_answer_fn=None,
        empty_adapters=False,
        num_assistant_tokens=0,
        assistant_confidence_threshold=0.0,
        device="CPU",
        generation_config_extra=None,
    ) -> None:
        if base_model is None and gt_data is None:
            raise ValueError("Text generation pipeline for evaluation or ground truth data must be defined")

        self.test_data = test_data
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.generation_fn = gen_answer_fn
        self.num_assistant_tokens = num_assistant_tokens
        self.assistant_confidence_threshold = assistant_confidence_threshold
        self.empty_adapters = empty_adapters
        self.generation_config_extra = generation_config_extra or {}
        self.full_chat = device == "NPU"

        self.gt_dir = os.path.dirname(gt_data or "")
        if base_model:
            self.gt_data = self._generate_data(base_model, gen_answer_fn, os.path.join(self.gt_dir, "reference"))
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        self.similarity = None
        self.divergency = None
        if "similarity" in self.metrics:
            self.similarity = TextSimilarity(similarity_model_id)
        if "divergency" in self.metrics:
            assert tokenizer is not None
            self.divergency = TextDivergency(tokenizer)

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
                text_data["prompts"].append(f.read())

        df = pd.DataFrame(text_data)
        return df

    def score(self, model_or_data, gen_answer_fn=None, output_dir=None, **kwargs):
        if output_dir is None:
            result_folder = os.path.join(self.gt_dir, "target")
        else:
            result_folder = os.path.join(output_dir, "target")

        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
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

        self.last_cmp["source_model"] = [
            "".join([f"Answer {i}:\n{rep}\n" for i, rep in enumerate(val)]) for val in gt_data_text["answers"].values
        ]
        self.last_cmp["optimized_model"] = [
            "".join([f"Answer {i}:\n{rep}\n" for i, rep in enumerate(val)])
            for val in predictions_text["answers"].values
        ]
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

    def _generate_data(self, model, gen_answer_fn=None, result_dir="reference"):
        gen_answer_fn = gen_answer_fn or default_gen_answer

        # applicable for ov model only, 2 is just a default value
        kv_axes_pos = 2
        if "optimum" in str(type(model)):
            kv_axes_pos = get_kv_axes_pos(model.model)

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
            # default data were created based on flpelerin/ChatAlpaca-10k and LDJnr/Puffin datasets
            data_path = files("whowhatbench.prompts").joinpath(self.CHAT_PROMPTS_FILE)
            prompt_data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            data = pd.DataFrame.from_dict(prompt_data)

        prompt_data = data["prompts"]

        answers = []
        prompts_paths = []
        prompts = prompt_data.values if self.num_samples is None else prompt_data.values[: self.num_samples]

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        prompts_dir = os.path.join(result_dir, "prompts")
        if not os.path.exists(prompts_dir):
            os.makedirs(prompts_dir)

        extra_kwargs = {"generation_config_extra": self.generation_config_extra} if self.generation_config_extra else {}
        for i, p in tqdm(enumerate(prompts), desc="Evaluate pipeline"):
            answer = gen_answer_fn(
                model,
                self.tokenizer,
                p,
                self.max_new_tokens,
                self.empty_adapters,
                self.num_assistant_tokens,
                self.assistant_confidence_threshold,
                self.full_chat,
                kv_axes_pos,
                **extra_kwargs,
            )

            result_path = os.path.join(result_dir, f"chat_output_{i}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(answer, f, ensure_ascii=False, indent=4)
            answers.append(result_path)

            prompt_path = os.path.join(prompts_dir, f"chat_prompts_{i}.txt")
            with open(prompt_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(p))
            prompts_paths.append(prompt_path)

        res_data = {"prompts": prompts_paths, "answers": answers}
        df = pd.DataFrame(res_data)

        return df
