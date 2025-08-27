import os
from typing import Any, Union

import pandas as pd
from tqdm import tqdm
from transformers import set_seed
import torch
import openvino_genai

from .registry import register_evaluator, BaseEvaluator

from .whowhat_metrics import ImageSimilarity

default_data = {
    "prompts": [
        "Cinematic, a vibrant Mid-century modern dining area, colorful chairs and a sideboard, ultra realistic, many detail",
        "colibri flying near a flower, side view, forest background, natural light, photorealistic, 4k",
        "Illustration of an astronaut sitting in outer space, moon behind him",
        "A vintage illustration of a retro computer, vaporwave aesthetic, light pink and light blue",
        "A view from beautiful alien planet, very beautiful, surealism, retro astronaut on the first plane, 8k photo",
        "red car in snowy forest, epic vista, beautiful landscape, 4k, 8k",
        "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
        "cute cat 4k, high-res, masterpiece, best quality, soft lighting, dynamic angle",
        "A cat holding a sign that says hello OpenVINO",
        "A small cactus with a happy face in the Sahara desert.",
    ],
}


@register_evaluator("text-to-image")
class Text2ImageEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        similarity_model_id: str = "openai/clip-vit-large-patch14",
        resolution=(512, 512),
        num_inference_steps=4,
        crop_prompts=True,
        num_samples=None,
        gen_image_fn=None,
        seed=42,
        is_genai=False,
        empty_adapters=False,
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.metrics = metrics
        self.resolution = resolution
        self.crop_prompt = crop_prompts
        self.num_samples = num_samples
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.similarity = None
        self.similarity = ImageSimilarity(similarity_model_id)
        self.last_cmp = None
        self.gt_dir = os.path.dirname(gt_data)
        self.generation_fn = gen_image_fn
        self.is_genai = is_genai
        self.empty_adapters = empty_adapters

        if base_model:
            base_model.resolution = self.resolution
            self.gt_data = self._generate_data(
                base_model, gen_image_fn, os.path.join(self.gt_dir, "reference")
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_image_fn=None, output_dir=None, **kwargs):
        if output_dir is None:
            image_folder = os.path.join(self.gt_dir, "target")
        else:
            image_folder = os.path.join(output_dir, "target")

        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            model_or_data.resolution = self.resolution
            predictions = self._generate_data(
                model_or_data, gen_image_fn, image_folder
            )
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompts"].values
        self.last_cmp["source_model"] = self.gt_data["images"].values
        self.last_cmp["optimized_model"] = predictions["images"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        res = self.last_cmp.nsmallest(top_k, metric)
        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(self, model, gen_image_fn=None, image_dir="reference"):
        def default_gen_image_fn(model, prompt, num_inference_steps, generator=None, empty_adapters=False):
            with torch.no_grad():
                output = model(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    output_type="pil",
                    width=self.resolution[1],
                    height=self.resolution[0],
                    generator=generator,
                )
            return output.images[0]

        generation_fn = gen_image_fn or default_gen_image_fn

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
            data = pd.DataFrame.from_dict(default_data)

        prompts = data["prompts"]
        prompts = (
            prompts.values
            if self.num_samples is None
            else prompts.values[: self.num_samples]
        )
        images = []
        rng = torch.Generator(device="cpu")

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for i, prompt in tqdm(enumerate(prompts), desc="Evaluate pipeline"):
            set_seed(self.seed)
            rng = rng.manual_seed(self.seed)
            image = generation_fn(
                model,
                prompt,
                self.num_inference_steps,
                generator=openvino_genai.TorchGenerator(self.seed) if self.is_genai else rng,
                empty_adapters=self.empty_adapters,
            )
            image_path = os.path.join(image_dir, f"{i}.png")
            image.save(image_path)
            images.append(image_path)

        res_data = {"prompts": list(prompts), "images": images}
        df = pd.DataFrame(res_data)

        return df
