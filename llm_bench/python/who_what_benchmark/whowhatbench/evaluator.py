from typing import Any, Union

import pandas as pd
from tqdm import tqdm

from .whowhat_metrics import DivergencyMetric, SimilarityMetric

default_data = {
    "en" : {
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
        ],
    },
    "cn": {
        "questions": [
            "马克吐温是谁?",
            "谁是威廉-莎士比亚?",
            "阿加莎-克里斯蒂是谁?",
            "芭芭拉-卡特兰是谁?",
            "丹妮尔-斯蒂尔是谁?"
            "谁是哈罗德-罗宾斯?",
            "乔治-西默农是谁?",
            "伊妮德-布莱顿是谁?",
            "西德尼-谢尔顿是谁?",
            "鸟山明是谁?",
            "谁是列夫-托尔斯泰?",
            "亚历山大-普希金是谁?",
            "斯蒂芬-金是谁?",
            "C++是什么?",
            "Python是什么?",
            "什么是 Java?",
            "JavaScript是什么?",
            "什么是 Perl?",
            "什么是 OpenCV?",
            "谁是最著名的作家?",
            "谁是最有名的发明家?",
            "谁是最著名的数学家?",
            "最著名的作曲家是谁?",
            "谁是最有名的程序员?",
            "谁是最著名的运动员?",
            "谁是最著名的古希腊科学家?",
            "蓝色和黄色混合会得到什么颜色?",
        ],
    },
}


def autodetect_language(model):
    model2language = {
        "chatglm": "cn",
        "qwen2": "cn",
        "qwen": "cn",
        "baichuan": "cn",
        "minicpmv": "cn",
        "internlm": "cn",
    }

    return model2language.get(model.config.model_type, "en")


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
        language=None
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

        # Take language from the base model if provided
        self.language = language
        if self.language is None:
            if base_model is not None:
                self.language = autodetect_language(base_model)

        if base_model:
            self.gt_data = self._generate_data(base_model)
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        # Take language ground truth if no base model provided
        if self.language is None and "language" in self.gt_data.columns:
            self.language = self.gt_data["language"].values[0]

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
            if self.language is None:
                print("No language detecting in the base model or ground truth data. Taking language from target model.")
                self.language = autodetect_language(model)
            data = pd.DataFrame.from_dict(default_data[self.language])

        questions = data["questions"]

        answers = []
        prompts = questions.values if self.num_samples is None else questions.values[:self.num_samples]

        for q in tqdm(prompts, desc="Evaluate pipeline"):
            answers.append(gen_answer_fn(model, self.tokenizer, q, self.max_new_tokens, self._crop_question))

        res_data = {"questions": list(prompts), "answers": answers}
        df = pd.DataFrame(res_data)
        df["language"] = self.language

        return df
