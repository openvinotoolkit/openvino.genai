from typing import Any, Union

import pandas as pd
from tqdm import tqdm

from .whowhat_metrics import DivergencyMetric, SimilarityMetric

default_data = {
    "questions": [
        "Who is Mark Twain?",
        "Who is William Shakespeare?",
        "Who is Agatha Christie?",
        "Who is Barbara Cartland?",
        "Who is Danielle Steel?",
        "Who is Harold Robbins?",
        "Who is Georges Simenon?",
        "Who is Enid Blyton?",
        "Who is Sidney Sheldon?",
        "Who is Akira Toriyama?",
        "Who is Leo Tolstoy?",
        "Who is Alexander Pushkin?",
        "Who is Stephen King?",
        "What is C++?",
        "What is Python?",
        "What is Java?",
        "What is JavaScript?",
        "What is Perl?",
        "What is OpenCV?",
        "Who is the most famous writer?",
        "Who is the most famous inventor?",
        "Who is the most famous mathematician?",
        "Who is the most famous composer?",
        "Who is the most famous programmer?",
        "Who is the most famous athlete?",
        "Who is the most famous ancient Greek scientist?",
        "What color will you get when you mix blue and yellow?",
    ]
}


class Evaluator:
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics=("similarity", "divergency"),
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens=128,
        crop_question=True,
        num_samples=None,
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

        if base_model:
            self.gt_data = self._generate_data(base_model)
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        self.similarity = None
        self.divergency = None
        if "similarity" in self.metrics:
            self.similarity = SimilarityMetric(similarity_model_id)
        if "divergency" in self.metrics:
            assert tokenizer is not None
            self.divergency = DivergencyMetric(tokenizer)

        self.last_cmp = None

    def dump_gt(self, csv_name: str):
        self.gt_data.to_csv(csv_name)

    def score(self, model, gen_answer_fn=None):
        predictions = self._generate_data(model, gen_answer_fn)

        all_metrics_per_question = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_question.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_question.update(metric_per_question)

        self.last_cmp = all_metrics_per_question
        self.last_cmp["questions"] = predictions["questions"].values
        self.last_cmp["source_model"] = self.gt_data["answers"].values
        self.last_cmp["optimized_model"] = predictions["answers"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)
        self.last_cmp.rename(columns={"questions": "prompt"}, inplace=True)

        return pd.DataFrame(all_metrics_per_question), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        if metric in ["SDT", "SDT norm"]:
            res = self.last_cmp.nlargest(top_k, metric)
        else:
            res = self.last_cmp.nsmallest(top_k, metric)

        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(self, model, gen_answer_fn=None):
        def default_gen_answer(model, tokenizer, question, max_new_tokens, crop_question):
            inputs = self.tokenizer(question, return_tensors="pt")
            tokens = model.generate(**inputs, max_new_tokens=max_new_tokens)
            out = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            return out[len(question) :] if crop_question else out

        gen_answer_fn = gen_answer_fn or default_gen_answer

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "questions" in self.test_data
                    data = dict(self.test_data)
                else:
                    data = {"questions": list(self.test_data)}
                data = pd.DataFrame.from_dict(data)
        else:
            data = pd.DataFrame.from_dict(default_data)

        questions = data["questions"]

        answers = []
        prompts = questions.values if self.num_samples is None else questions.values[:self.num_samples]

        for q in tqdm(prompts, desc="Evaluate pipeline"):
            answers.append(gen_answer_fn(model, self.tokenizer, q, self.max_new_tokens, self._crop_question))

        res_data = {"questions": list(prompts), "answers": answers}
        df = pd.DataFrame(res_data)

        return df
