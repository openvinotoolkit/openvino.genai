# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import datasets
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch import Tensor
from transformers import set_seed
from typing import Literal, Any, Union
from transformers.image_utils import load_image

from .whowhat_metrics import EmbedsSimilarity
from .registry import register_evaluator, BaseEvaluator
from .utils import prepare_default_data_video as prepare_video_dataset


DEFAULT_MAX_LENGTH = 200


def prepare_default_text_data(num_samples=None):
    DATASET_NAME = "microsoft/ms_marco"
    NUM_SAMPLES = num_samples if num_samples else 24
    set_seed(42)
    default_dataset = datasets.load_dataset(
        DATASET_NAME, 'v2.1', split="test", streaming=True
    ).shuffle(42).take(NUM_SAMPLES)
    return default_dataset.map(
        lambda x: {'passages': x['passages']['passage_text']}, remove_columns=default_dataset.column_names
    )


def prepare_default_image_data(num_samples=None):
    DATASET_NAME = "yerevann/coco-karpathy"
    NUM_SAMPLES = num_samples if num_samples else 24
    set_seed(42)
    default_dataset = datasets.load_dataset(DATASET_NAME, split="test", streaming=True).shuffle(42).take(NUM_SAMPLES)
    return default_dataset.map(
        lambda x: {"images": [load_image(x["url"])] * len(x["sentences"]), "passages": x["sentences"]},
        remove_columns=default_dataset.column_names,
    )


def prepare_default_data_video(num_samples=None, num_frames=10):
    items = prepare_video_dataset(num_samples=num_samples, num_frames=num_frames)
    data = []
    for item in items:
        data.append(
            {
                "passages": [item["prompts"]],
                "videos": [item["videos"]],
            }
        )

    return data


class Qwen3VLEmbeddingWrapper:
    system_prompt = "Represent the user's input."

    def __init__(self, model, **kwargs):
        self.model = model
        self.config = model.config

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        return getattr(self.model, attr)

    @property
    def device(self):
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def eval(self):
        self.model.eval()
        return self

    def to(self, *args, **kwargs):
        if hasattr(self.model, "to"):
            self.model.to(*args, **kwargs)
        return self

    def preprocess_inputs(self, tokenizer=None, processor=None, passages=None, images=None, videos=None, **kwargs):
        passages = [] if passages is None else passages
        images = [] if images is None else images
        videos = [] if videos is None else videos
        messages = []
        for text, image, video in itertools.zip_longest(passages, images, videos):
            content = [{"type": "text", "text": text}]
            if image is not None:
                content.append({"type": "image", "image": image})
            if video is not None:
                content.append({"type": "video", "video": video})

            messages.append([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}])

        input_kwargs = {}
        if images:
            input_kwargs["images"] = images
        if videos:
            input_kwargs["videos"] = videos

        tokenizer_kwargs = {
            "padding": True,
            "padding_side": kwargs.get("padding_side", "right"),
        }

        inputs = {}
        if input_kwargs:
            templated_text_input = processor.apply_chat_template(
                messages, tokenize=False, enable_thinking=False, add_generation_prompt=False
            )
            inputs = processor(text=templated_text_input, **input_kwargs, return_tensors="pt", **tokenizer_kwargs)
        else:
            templated_text_input = tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False, add_generation_prompt=False
            )
            inputs = tokenizer(text=templated_text_input, **input_kwargs, return_tensors="pt", **tokenizer_kwargs)

        return inputs

    def do_custom_pooling(self, attention_mask, outputs):
        return last_token_pool(outputs.last_hidden_state, attention_mask)

    @staticmethod
    def is_qwen3_vl_model(model):
        is_qwen3_vl = False
        if isinstance(model, str) or isinstance(model, Path):
            from transformers import AutoConfig

            config = None
            try:
                config = AutoConfig.from_pretrained(model)
            except Exception:
                config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            if config and hasattr(config, "model_type") and config.model_type == "qwen3_vl":
                is_qwen3_vl = True
        elif hasattr(model, "config") and hasattr(model.config, "model_type") and model.config.model_type == "qwen3_vl":
            is_qwen3_vl = True

        return is_qwen3_vl


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


@register_evaluator("text-embedding", "image-embedding", "video-embedding")
class EmbeddingsEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        processor: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        num_samples=None,
        gen_embeds_fn=None,
        pooling_type=None,
        normalize=None,
        padding_side=None,
        batch_size=None,
        pipeline_type: Literal["text-embedding", "image-embedding", "video-embedding"] = "text-embedding",
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.pipeline_type = pipeline_type
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_samples = num_samples
        self.generation_fn = gen_embeds_fn
        self.pooling_type = pooling_type
        self.normalize = normalize or False
        self.padding_side = padding_side or 'right'
        self.gt_dir = os.path.dirname(gt_data)
        self.batch_size = batch_size

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
            self.gt_data[: self.num_samples], predictions[: self.num_samples]
        )

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["passages"] = predictions["passages"].values[: self.num_samples]
        self.last_cmp["source_model"] = self.gt_data["embeds_path"].values[: self.num_samples]
        self.last_cmp["optimized_model"] = predictions["embeds_path"].values[: self.num_samples]
        self.last_cmp = pd.DataFrame(self.last_cmp)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None
        res = self.last_cmp.nsmallest(top_k, metric)
        return list(row for idx, row in res.iterrows())

    def _generate_data(self, model, gen_answer_fn=None, result_dir="reference"):
        def default_gen_answer(
            model, tokenizer=None, processor=None, passages=None, images=None, videos=None, **kwargs
        ):
            device = "cpu"
            if hasattr(model, "device"):
                device = model.device

            inputs = {}
            if Qwen3VLEmbeddingWrapper.is_qwen3_vl_model(model):
                inputs = model.preprocess_inputs(
                    tokenizer=tokenizer,
                    processor=processor,
                    passages=passages,
                    images=images,
                    videos=videos,
                    **kwargs,
                )
            else:
                tokenizer_kwargs = {
                    "padding": "max_length",
                    "max_length": DEFAULT_MAX_LENGTH,
                    "truncation": True,
                    "padding_side": kwargs.get("padding_side", "right"),
                }
                inputs = self.tokenizer(passages, return_tensors="pt", **tokenizer_kwargs).to(device)

            if hasattr(inputs, "to"):
                inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            if kwargs.get("pooling_type") == "last_token" and "attention_mask" in inputs:
                embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            elif kwargs.get("pooling_type") == "mean" and "attention_mask" in inputs:
                embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            else:
                if not kwargs.get("pooling_type") and Qwen3VLEmbeddingWrapper.is_qwen3_vl_model(model):
                    embeddings = model.do_custom_pooling(inputs["attention_mask"], outputs)
                else:
                    embeddings = outputs.last_hidden_state[:, 0]

            if kwargs.get("normalize", False):
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.to(torch.float32).cpu().numpy()

        gen_answer_fn = gen_answer_fn or default_gen_answer

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    if "passages" not in self.test_data:
                        raise RuntimeError("Test data must contain 'passages' keys")
                    data = dict(self.test_data)
                else:
                    data = {"passages": list(self.test_data)}
                data = pd.DataFrame.from_dict(data)
        else:
            if self.pipeline_type == "text-embedding":
                input_dataset = prepare_default_text_data(self.num_samples)
            elif self.pipeline_type == "image-embedding":
                input_dataset = prepare_default_image_data(self.num_samples)
            elif self.pipeline_type == "video-embedding":
                input_dataset = prepare_default_data_video(self.num_samples)
            else:
                raise RuntimeError("Test data must contain 'passages' or 'images' or 'videos' keys")
            data = pd.DataFrame.from_dict(input_dataset)

        texts = data["passages"].values if self.num_samples is None else data["passages"].values[: self.num_samples]
        images = []
        videos = []

        if self.pipeline_type == "image-embedding":
            images = data["images"].values if self.num_samples is None else data["images"].values[: self.num_samples]
        if self.pipeline_type == "video-embedding":
            videos = data["videos"].values if self.num_samples is None else data["videos"].values[: self.num_samples]

        embeds_paths = []
        passages = []
        prompt = Qwen3VLEmbeddingWrapper.system_prompt if Qwen3VLEmbeddingWrapper.is_qwen3_vl_model(model) else None

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for i, (texts_input, images_input, videos_input) in tqdm(
            enumerate(itertools.zip_longest(texts, images, videos)),
            total=max(len(texts), len(images), len(videos)),
            desc="Evaluate pipeline",
        ):
            kwargs = {}
            text_data_input = None
            if texts_input:
                texts_seq = [texts_input] if isinstance(texts_input, str) else texts_input

                kwargs = {
                    "padding_side": self.padding_side,
                    "pooling_type": self.pooling_type,
                    "normalize": self.normalize,
                }
                batch_size = self.batch_size or len(texts_seq)
                data_len = len(texts_seq)

                if batch_size <= data_len:
                    text_data_input = texts_seq[:batch_size]
                else:
                    # Duplicate data to reach batch_size
                    text_data_input = list(itertools.islice(itertools.cycle(texts_seq), batch_size))
                passages.append(text_data_input)

            result = gen_answer_fn(
                model,
                self.tokenizer,
                self.processor,
                text_data_input,
                images_input,
                videos_input,
                prompt=prompt,
                **kwargs,
            )

            result_path = os.path.join(result_dir, f"embeds_{i}.npy")
            with open(result_path, 'wb') as f:
                np.save(f, result)
            embeds_paths.append(result_path)

        res_data = {"passages": passages, "embeds_path": embeds_paths}
        df = pd.DataFrame(res_data)

        return df
