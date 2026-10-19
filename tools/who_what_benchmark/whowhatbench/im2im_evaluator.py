# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Union

import datasets
from huggingface_hub import snapshot_download
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import set_seed
import torch
import openvino_genai

from .utils import parquet_generate_tables
from .registry import register_evaluator
from .inpaint_evaluator import patched_parquet
from .text2image_evaluator import Text2ImageEvaluator

from .whowhat_metrics import ImageSimilarity


def preprocess_fn(example):
    return {
        "prompts": example["Instruction_VLM-LLM"],
        "images": example["source_img"],
    }


def prepare_default_data(num_samples=None):
    DATASET_NAME = "paint-by-inpaint/PIPE"
    NUM_SAMPLES = 10 if num_samples is None else num_samples
    set_seed(42)
    default_dataset = datasets.load_dataset(
        DATASET_NAME, split="test", streaming=True
    ).filter(lambda example: example["Instruction_VLM-LLM"] != "").take(NUM_SAMPLES)
    return default_dataset.map(
        lambda x: preprocess_fn(x), remove_columns=default_dataset.column_names
    )


def prepare_default_data_multi_image(num_samples=None):
    dataset_name = "OmniGen2/OmniContext"
    dataset_revision = "0f6aabbe8d3448d51184093372e980205d9980d3"
    num_samples = 10 if num_samples is None else num_samples
    dataset_path = Path(
        snapshot_download(
            dataset_name,
            repo_type="dataset",
            revision=dataset_revision,
            allow_patterns=["data-*.arrow"],
        )
    )
    default_dataset = (
        datasets.load_dataset(
            "arrow",
            data_files=sorted(str(path) for path in dataset_path.glob("data-*.arrow")),
            split="train",
            streaming=True,
        )
        .filter(lambda example: len(example["input_images"]) >= 2)
        .take(num_samples)
    )
    return default_dataset.map(
        lambda example: {
            "prompts": example["instruction"],
            "images": example["input_images"],
        },
        remove_columns=default_dataset.column_names,
    )


@register_evaluator("image-to-image")
class Image2ImageEvaluator(Text2ImageEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        similarity_model_id: str = "openai/clip-vit-large-patch14",
        num_inference_steps=None,
        crop_prompts=True,
        num_samples=None,
        gen_image_fn=None,
        seed=42,
        is_genai=False,
        image_dir=None,
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.metrics = metrics
        self.crop_prompt = crop_prompts
        self.num_samples = num_samples
        self.num_inference_steps = num_inference_steps or self.DEFAULT_NUM_INFERENCE_STEPS
        self.seed = seed
        self.similarity = None
        self.similarity = ImageSimilarity(similarity_model_id)
        self.last_cmp = None
        self.gt_dir = os.path.dirname(gt_data)
        self.generation_fn = gen_image_fn
        self.is_genai = is_genai
        self.resolution = None
        self.image_dir = image_dir

        if base_model:
            self.gt_data = self._generate_data(
                base_model, gen_image_fn, os.path.join(self.gt_dir, "reference")
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)
            if self.num_samples is not None:
                self.gt_data = self.gt_data.iloc[: self.num_samples]

    def _generate_data(self, model, gen_image_fn=None, image_dir="reference"):
        def default_gen_image_fn(model, prompt, image, num_inference_steps, generator=None):
            kwargs = {
                "prompt": prompt,
                "image": image,
                "num_inference_steps": num_inference_steps,
                "output_type": "pil",
                "strength": 0.8,
                "generator": generator,
            }
            if model.__class__.__name__ in ("QwenImage21Pipeline", "OVQwenImage21Img2ImgPipeline"):
                # Qwen Image 2.1 conditions on image tokens and does not support the img2img strength argument.
                kwargs.pop("strength")
            with torch.no_grad():
                output = model(**kwargs)
            return output.images[0]

        generation_fn = gen_image_fn or default_gen_image_fn

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "prompts" in self.test_data
                    assert "images" in self.test_data
                    data = dict(self.test_data)
                data = pd.DataFrame.from_dict(data)
        else:
            with patched_parquet(parquet_generate_tables):
                data = pd.DataFrame.from_dict(self._prepare_default_data())

        if self.num_samples is not None:
            data = data.iloc[: self.num_samples]
        prompts = data["prompts"]
        images = data["images"]
        output_images = []
        rng = torch.Generator(device="cpu")

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for i, (prompt, image) in tqdm(
            enumerate(zip(prompts, images)), total=min(len(prompts), len(images)), desc="Evaluate pipeline"
        ):
            set_seed(self.seed)
            rng = rng.manual_seed(self.seed)
            image = self._resolve_conditioning_images(image)
            output = generation_fn(
                model,
                prompt,
                image=image,
                num_inference_steps=self.num_inference_steps,
                generator=openvino_genai.TorchGenerator(self.seed) if self.is_genai else rng
            )
            image_path = os.path.join(image_dir, f"{i}.png")
            output.save(image_path)
            output_images.append(image_path)

        res_data = {"prompts": list(prompts), "images": output_images}
        df = pd.DataFrame(res_data)

        return df

    def _prepare_default_data(self):
        return prepare_default_data(self.num_samples)

    def _resolve_conditioning_images(self, image):
        if isinstance(image, str):
            if not os.path.isabs(image) and self.image_dir:
                image = os.path.join(self.image_dir, image)
            return Image.open(image).convert("RGB")
        return image


@register_evaluator("multi-image-to-image")
class MultiImage2ImageEvaluator(Image2ImageEvaluator):
    def _prepare_default_data(self):
        return prepare_default_data_multi_image(self.num_samples)

    def _resolve_conditioning_images(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) < 2:
            raise ValueError(f"multi-image-to-image requires at least 2 conditioning images, got {len(images)}")
        resolve_image = super()._resolve_conditioning_images
        return [resolve_image(image) for image in images]
