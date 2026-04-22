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
from .utils.utils import patch_awq_for_inference, get_ignore_parameters_flag
import inspect


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
        def default_gen_answer(
            model,
            tokenizer,
            prompts,
            max_new_tokens,
            _empty_adapters=False,
            _num_assistant_tokens=0,
            _assistant_confidence_threshold=0.0,
        ):
            is_awq = getattr(model, "is_awq", None) is not None
            device = "cpu"
            if hasattr(model, "device"):
                device = model.device

            chat_history = []
            answers = []
            for prompt in prompts:
                chat_history.append({"role": "user", "content": prompt})
                inputs = tokenizer.apply_chat_template(
                    chat_history, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
                ).to(device)

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

                answer_tokens = tokens[:, inputs["input_ids"].shape[-1] :]
                answer_text = tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
                chat_history.append({"role": "assistant", "content": answer_text[0]})
                answers.append(answer_text[0])

            return answers

        gen_answer_fn = gen_answer_fn or default_gen_answer

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

        for i, p in tqdm(enumerate(prompts), desc="Evaluate pipeline"):
            answer = gen_answer_fn(
                model,
                self.tokenizer,
                p,
                self.max_new_tokens,
                self.empty_adapters,
                self.num_assistant_tokens,
                self.assistant_confidence_threshold,
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
