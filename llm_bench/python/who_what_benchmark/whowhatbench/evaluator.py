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
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer

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

    def score(self, model):
        predictions = self._generate_data(model)

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

    def _generate_data(self, model):
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

        for q in tqdm(questions.values, desc="Evaluate pipeline"):
            inputs = self.tokenizer(q, return_tensors="pt")
            tokens = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            out = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            answers.append(out[len(q) :])

        res_data = {"questions": list(questions.values), "answers": answers}
        df = pd.DataFrame(res_data)

        return df
