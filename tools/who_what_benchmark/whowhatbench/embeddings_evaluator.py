from typing import Any, Union

import os
import torch
import numpy as np
import pandas as pd
import datasets

from tqdm import tqdm
from torch import Tensor
from transformers import set_seed

from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import EmbedsSimilarity


DEFAULT_MAX_LENGTH = 200


def prepare_default_data(num_samples=None):
    DATASET_NAME = "microsoft/ms_marco"
    NUM_SAMPLES = num_samples if num_samples else 24
    set_seed(42)
    default_dataset = datasets.load_dataset(
        DATASET_NAME, 'v2.1', split="test", streaming=True
    ).shuffle(42).take(NUM_SAMPLES)
    return default_dataset.map(
        lambda x: {'passages': x['passages']['passage_text']}, remove_columns=default_dataset.column_names
    )


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        batch_dim = torch.arange(batch_size, device=last_hidden_states.device)
        result = last_hidden_states[batch_dim, sequence_lengths]
        return result


def mean_pooling(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).to(last_hidden_states.dtype)
    )
    sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    return sum_embeddings / sum_mask


@register_evaluator(
    "text-embedding"
)
class EmbeddingsEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        num_samples=None,
        gen_embeds_fn=None,
        pooling_type=None,
        normalize=None,
        padding_side=None
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.generation_fn = gen_embeds_fn
        self.pooling_type = pooling_type or 'cls'
        self.normalize = normalize or False
        self.padding_side = padding_side or 'right'
        self.gt_dir = os.path.dirname(gt_data)

        if base_model:
            self.gt_data = self._generate_data(
                base_model, gen_embeds_fn, os.path.join(self.gt_dir, "reference")
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        self.similarity = EmbedsSimilarity()
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

        all_metrics_per_prompt = {}
        all_metrics = {}
        all_metrics, all_metrics_per_prompt = self.similarity.evaluate(
            self.gt_data, predictions
        )

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["passages"] = predictions["passages"].values
        self.last_cmp["source_model"] = self.gt_data["embeds_path"].values
        self.last_cmp["optimized_model"] = predictions["embeds_path"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None
        res = self.last_cmp.nsmallest(top_k, metric)
        return list(row for idx, row in res.iterrows())

    def _generate_data(self, model, gen_answer_fn=None, result_dir="reference"):
        def default_gen_answer(model, tokenizer, passages, **kwargs):
            device = "cpu"
            if hasattr(model, "device"):
                device = model.device
            tokenizer_kwargs = {'padding': 'max_length', 'max_length': DEFAULT_MAX_LENGTH,
                                'truncation': True, 'padding_side': kwargs.get('padding_side', 'right')}
            inputs = self.tokenizer(passages, return_tensors="pt", **tokenizer_kwargs).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            if kwargs.get("pooling_type") == "last_token":
                embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            elif kwargs.get("pooling_type") == "mean":
                embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            else:
                embeddings = outputs.last_hidden_state[:, 0]

            if kwargs.get("normalize", False):
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

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
            data = pd.DataFrame.from_dict(prepare_default_data(self.num_samples))

        embeds_paths = []
        passages = []
        inputs = (
            data.values
            if self.num_samples is None
            else data.values[: self.num_samples]
        )

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for i, data in tqdm(enumerate(inputs), desc="Evaluate pipeline"):
            kwargs = {'padding_side': self.padding_side,
                      'pooling_type': self.pooling_type,
                      'normalize': self.normalize}
            result = gen_answer_fn(model, self.tokenizer, data[0], **kwargs)
            passages.append(data[0])
            result_path = os.path.join(result_dir, f"embeds_{i}.npy")
            with open(result_path, 'wb') as f:
                np.save(f, result)
            embeds_paths.append(result_path)

        res_data = {"passages": passages, "embeds_path": embeds_paths}
        df = pd.DataFrame(res_data)

        return df
