from typing import Any, Union

import os
import yaml
import pandas as pd
from tqdm import tqdm
from importlib.resources import files
from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import TextDivergency, TextSimilarity
from .utils import patch_awq_for_inference, get_ignore_parameters_flag
import inspect

PROMPTS_FILE = 'text_prompts.yaml'
LONG_PROMPTS_FILE = 'text_long_prompts.yaml'


@register_evaluator(
    "text"
)
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
        generation_config_base=None,
        seqs_per_request=None,
        use_chat_template=None,
        long_prompt=False
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
        self.generation_config_base = generation_config
        self.seqs_per_request = seqs_per_request
        self.generation_fn = gen_answer_fn
        self.use_chat_template = use_chat_template
        if self.generation_config is not None:
            assert self.seqs_per_request is not None

        # Take language from the base model if provided
        self.language = language

        self.long_prompt = long_prompt

        if base_model:
            self.gt_data = self._generate_data(
                base_model, gen_answer_fn, generation_config=generation_config
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
        if "similarity" in self.metrics:
            self.similarity = TextSimilarity(similarity_model_id)
        if "divergency" in self.metrics:
            assert tokenizer is not None
            self.divergency = TextDivergency(tokenizer)

        self.last_cmp = None

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_answer_fn=None, **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn, self.generation_config)
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompts"].values
        self.last_cmp["source_model"] = self.gt_data["answers"].values
        self.last_cmp["optimized_model"] = predictions["answers"].values
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

    def _generate_data(self, model, gen_answer_fn=None, generation_config=None):
        def default_gen_answer(model, tokenizer, prompt, max_new_tokens, crop_question, use_chat_template=False):
            is_awq = getattr(model, "is_awq", None) is not None
            device = "cpu"
            if hasattr(model, "device"):
                device = model.device

            if use_chat_template:
                message = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            if 'token_type_ids' in inputs and 'token_type_ids' not in list(inspect.signature(model.forward).parameters.keys()):
                inputs.pop('token_type_ids')

            if is_awq:
                with patch_awq_for_inference(is_awq):
                    tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, **get_ignore_parameters_flag())
            else:
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, **get_ignore_parameters_flag())
            if crop_question:
                tokens = tokens[:, inputs["input_ids"].shape[-1] :]
            return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

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
            prompts_file_path = LONG_PROMPTS_FILE if self.long_prompt else PROMPTS_FILE
            data_path = files('whowhatbench.prompts').joinpath(prompts_file_path)
            prompt_data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
            data = pd.DataFrame.from_dict(prompt_data[self.language])

        prompt_data = data["prompts"]

        answers = []
        prompts = (
            prompt_data.values
            if self.num_samples is None
            else prompt_data.values[: self.num_samples]
        )

        if generation_config is None:
            for p in tqdm(prompts, desc="Evaluate pipeline"):
                answers.append(
                    gen_answer_fn(
                        model,
                        self.tokenizer,
                        p,
                        self.max_new_tokens,
                        self._crop_question,
                        self.use_chat_template
                    )
                )
        else:
            with tqdm(total=len(prompt_data.values)) as progress_bar:
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
                            answers.append(ans.m_generation_ids[0])

                        batch.clear()

        res_data = {"prompts": list(prompts), "answers": answers}
        df = pd.DataFrame(res_data)
        df["language"] = self.language
        df["prompt_length_type"] = 'long' if self.long_prompt else 'short'

        return df
