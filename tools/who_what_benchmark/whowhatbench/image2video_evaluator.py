# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
from typing import Any, Union

import datasets
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import set_seed
from diffusers.utils import export_to_video
import torch
import openvino_genai

from .registry import register_evaluator
from .text2video_evaluator import Text2VideoEvaluator


def prepare_default_data(num_samples=None):
    DATASET_NAME = "paint-by-inpaint/PIPE"
    NUM_SAMPLES = 10 if num_samples is None else num_samples
    set_seed(42)
    default_dataset = (
        datasets.load_dataset(DATASET_NAME, split="test", streaming=True)
        .filter(lambda example: example["source_img"] is not None)
        .take(NUM_SAMPLES)
    )
    return [example["source_img"] for example in default_dataset]


@register_evaluator("image-to-video")
class Image2VideoEvaluator(Text2VideoEvaluator):
    DEF_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"

    def __init__(
        self,
        base_model: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        num_inference_steps=None,
        num_frames=None,
        crop_prompts=True,
        num_samples=None,
        gen_video_fn=None,
        seed=42,
        is_genai=False,
        empty_adapters=False,
        image_dir=None,
    ) -> None:
        # Read by _generate_data(), which the base constructor calls.
        self.image_dir = image_dir
        super().__init__(
            base_model=base_model,
            gt_data=gt_data,
            test_data=test_data,
            metrics=metrics,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            crop_prompts=crop_prompts,
            num_samples=num_samples,
            gen_video_fn=gen_video_fn,
            seed=seed,
            is_genai=is_genai,
            empty_adapters=empty_adapters,
        )

    def collect_default_data(self):
        from importlib.resources import files

        data_path = files("whowhatbench.prompts").joinpath("image_to_video_prompts.json")
        with open(data_path) as input_file:
            prompts = json.load(input_file)

        num = self.num_samples if self.num_samples is not None else len(prompts)
        images = prepare_default_data(num_samples=num)
        # The dataset is streamed and filtered, so it may yield fewer images than requested.
        # Cycle the prompts to match however many images we actually got.
        selected = [prompts[i % len(prompts)] for i in range(len(images))]

        return {
            "prompt": [p["prompt"] for p in selected],
            "negative_prompt": [p["negative_prompt"] for p in selected],
            "width": [p["width"] for p in selected],
            "height": [p["height"] for p in selected],
            "guidance_scale": [p["guidance_scale"] for p in selected],
            "images": images,
        }

    def _resolve_image(self, row, index):
        """Resolve a row's conditioning image to a PIL RGB image."""
        if "images" in row:
            spec = row["images"]
        elif "image" in row:
            spec = row["image"]
        else:
            assert self.image_dir is not None, (
                "test_data has no 'images'/'image' column and --image-dir was not provided"
            )
            spec = f"{index}.png"

        if isinstance(spec, Image.Image):
            return spec.convert("RGB")
        if isinstance(spec, np.ndarray):
            return Image.fromarray(spec).convert("RGB")
        if isinstance(spec, str):
            if not os.path.isabs(spec) and self.image_dir:
                spec = os.path.join(self.image_dir, spec)
            return Image.open(spec).convert("RGB")
        raise ValueError(f"Unsupported conditioning image type at index {index}: {type(spec).__name__}")

    def _generate_data(self, model, gen_video_fn=None, videos_dir="reference"):
        def default_gen_video_fn(
            model,
            prompt,
            image,
            negative_prompt,
            num_inference_steps,
            width=self.DEF_WIDTH,
            height=self.DEF_HEIGHT,
            num_frames=self.DEF_NUM_FRAMES,
            frame_rate=self.DEF_FRAME_RATE,
            guidance_scale=self.DEF_GUIDANCE_SCALE,
            guidance_rescale=self.DEF_GUIDANCE_RESCALE,
            generator=None,
            empty_adapters=False,
        ):
            kwargs = {"negative_prompt": negative_prompt} if guidance_scale > 1 else {}
            with torch.no_grad():
                output = model(
                    image=image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    generator=generator,
                    max_sequence_length=256,
                    **kwargs,
                )
            return output.frames[0]

        generation_fn = gen_video_fn or default_gen_video_fn

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            elif isinstance(self.test_data, dict):
                assert "prompt" in self.test_data
                data = pd.DataFrame.from_dict(dict(self.test_data))
            else:
                raise ValueError(
                    "image-to-video test_data must be a CSV path or a dict of columns including 'prompt'; "
                    f"got {type(self.test_data).__name__}. A bare prompt list has no conditioning images."
                )
        else:
            data = pd.DataFrame.from_dict(self.collect_default_data())

        if self.num_samples is not None:
            data = data.iloc[: self.num_samples]

        res_keys = ["prompt", "negative_prompt", "width", "height", "guidance_scale"]
        res_data = {key: data[key].tolist() for key in res_keys if key in data.columns}
        videos = []

        rng = torch.Generator(device="cpu")

        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)

        for i, (_, row) in tqdm(enumerate(data.iterrows()), total=len(data), desc="Evaluate pipeline"):
            set_seed(self.seed)
            rng = rng.manual_seed(self.seed)
            frames = generation_fn(
                model,
                prompt=row["prompt"],
                image=self._resolve_image(row, i),
                negative_prompt=row.get("negative_prompt", self.DEF_NEGATIVE_PROMPT),
                num_inference_steps=self.num_inference_steps,
                width=int(row.get("width", self.DEF_WIDTH)),
                height=int(row.get("height", self.DEF_HEIGHT)),
                num_frames=self.num_frames,
                frame_rate=self.frame_rate,
                guidance_scale=row.get("guidance_scale", self.DEF_GUIDANCE_SCALE),
                guidance_rescale=row.get("guidance_rescale", self.DEF_GUIDANCE_RESCALE),
                generator=openvino_genai.TorchGenerator(self.seed) if self.is_genai else rng,
                empty_adapters=self.empty_adapters,
            )
            video_path = os.path.join(videos_dir, f"video_{i}.mp4")
            export_to_video(frames, video_path, self.frame_rate)
            videos.append(video_path)

        res_data["videos"] = videos
        df = pd.DataFrame(res_data)

        return df
