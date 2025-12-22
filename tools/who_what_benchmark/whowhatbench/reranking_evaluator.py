from typing import Any, Union

import os
import scipy
import torch
import pandas as pd
from tqdm import tqdm
from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import RerankingSimilarity
from transformers import set_seed
import datasets
import numpy as np


# we would like to evalute score for all documents
# In GenAI, top_k is set at the pipeline creation stage, but data is collected after the pipeline creation stage
# let's set potential big top_k(avg amount of documents in default dataset is 10) so that all documents are included in the final ranking
DEFAULT_TOP_K = 100
DEFAULT_MAX_LENGTH = 200
DEFAULT_MAX_LENGTH_QWEN = 8192


def is_qwen3(config):
    return config.model_type == "qwen3"


def is_qwen3_causallm(config):
    return is_qwen3(config) and "Qwen3ForCausalLM" in config.architectures


def preprocess_fn(example):
    return {
        "query": example["query"],
        "passages": example["passages"]["passage_text"],
    }


def prepare_default_data(num_samples=None):
    DATASET_NAME = "microsoft/ms_marco"
    NUM_SAMPLES = num_samples if num_samples else 24
    set_seed(42)
    default_dataset = datasets.load_dataset(
        DATASET_NAME, 'v2.1', split="test", streaming=True
    ).shuffle(42).take(NUM_SAMPLES)
    return default_dataset.map(
        lambda x: preprocess_fn(x), remove_columns=default_dataset.column_names
    )


@register_evaluator(
    "text-reranking"
)
class RerankingEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        num_samples=None,
        gen_rerank_fn=None
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.generation_fn = gen_rerank_fn
        self.gt_dir = os.path.dirname(gt_data)

        if base_model:
            self.gt_data = self._generate_data(
                base_model, gen_rerank_fn, os.path.join(self.gt_dir, "reference")
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        self.similarity = RerankingSimilarity()
        self.last_cmp = None

    def get_generation_fn(self):
        return self.generation_fn

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

        all_metrics, all_metrics_per_query = self.similarity.evaluate(
            self.gt_data, predictions
        )

        self.last_cmp = all_metrics_per_query
        self.last_cmp["query"] = predictions["query"].values
        self.last_cmp["passages"] = predictions["passages"].values
        self.last_cmp["source_model"] = self.gt_data["top_n_scores_path"].values
        self.last_cmp["optimized_model"] = predictions["top_n_scores_path"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)

        return pd.DataFrame(all_metrics_per_query), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None
        res = self.last_cmp.nsmallest(top_k, metric)
        res = list(row for idx, row in res.iterrows())
        return res

    def _apply_qwen3_template(self, query, passages):
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
            + 'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        task = "Given a web search query, retrieve relevant passages that answer the query"

        qwen3_query = f"{prefix}<Instruct>: {task}\n<Query>: {query}\n"
        qwen3_passages = [f"<Document>: {doc}{suffix}" for doc in passages]
        return qwen3_query, qwen3_passages

    def _generate_data(self, model, gen_answer_fn=None, result_dir="reference"):
        def default_gen_answer(model, tokenizer, query, passages):
            # post/pre processing for qwen models added according to transformers Qwen3-Embedding-0.6B model card:
            # https://huggingface.co/Qwen/Qwen3-Reranker-0.6B#transformers-usage
            if is_qwen3(model.config):
                pairs = [f"{query}{passage}" for passage in passages]
                input_data = tokenizer(pairs, padding=True, truncation=True, max_length=DEFAULT_MAX_LENGTH_QWEN, return_tensors="pt", padding_side="left")
            else:
                tokenizer_kwargs = {"truncation": True, "padding": True, "max_length": DEFAULT_MAX_LENGTH}
                inputs = [query] * len(passages)
                input_data = tokenizer(inputs, passages, return_tensors="pt", **tokenizer_kwargs)

            with torch.no_grad():
                outputs = model(**input_data).logits

            if is_qwen3_causallm(model.config):
                batch_scores = outputs[:, -1, :]
                token_false_id = tokenizer.convert_tokens_to_ids("no")
                token_true_id = tokenizer.convert_tokens_to_ids("yes")
                true_vector = batch_scores[:, token_true_id]
                false_vector = batch_scores[:, token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp()
            else:
                if outputs.shape[1] > 1:
                    scores = outputs[:, 1]
                else:
                    scores = outputs.flatten()
                scores = scipy.special.expit(scores)
            ordered_scores = []
            for index, (score, _) in enumerate(zip(scores, passages)):
                ordered_scores.append(np.array([index, score.numpy()]))
            return np.array(ordered_scores[:DEFAULT_TOP_K])

        gen_answer_fn = gen_answer_fn or default_gen_answer

        # TODO: add possibility to use custom dataset/csv
        df = pd.DataFrame.from_dict(prepare_default_data(self.num_samples))

        scores_path = []
        passages = []
        queries = []
        inputs = (
            df.values
            if self.num_samples is None
            else df.values[: self.num_samples]
        )

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for i, data in tqdm(enumerate(inputs), desc="Evaluate pipeline"):
            query = data[0]
            documents = data[1]
            if is_qwen3(model.config):
                query, documents = self._apply_qwen3_template(data[0], data[1])
            result = gen_answer_fn(model, self.tokenizer, query, documents)

            queries.append(data[0])
            passages.append(data[1])
            result_path = os.path.join(result_dir, f"scores_{i}.npy")
            with open(result_path, 'wb') as f:
                np.save(f, result)
            scores_path.append(result_path)

        res_data = {"query": queries, "passages": passages, "top_n_scores_path": scores_path}
        df_result = pd.DataFrame(res_data)

        return df_result
