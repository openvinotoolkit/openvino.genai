import os
from typing import Any, Union

import pandas as pd
from tqdm import tqdm
from transformers import set_seed
from diffusers.utils import export_to_video
import torch
import openvino_genai

from .registry import register_evaluator, BaseEvaluator

from .whowhat_metrics import VideoSimilarity


default_data = [
    # small resolution
    {
        "prompt": "octopus figure skating, cartoon  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 256,
        "height": 128,
        "guidance_scale": 3,
        "guidance_rescale": 0.3,
    },
    # small resolution
    {
        "prompt": "slow motion, hydrogen bond energy, atom, 4k, cinematic -gs 24 -motion 2 -ar 16:9 -fps 24  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 256,
        "height": 256,
        "guidance_scale": 3,
        "guidance_rescale": 0.3,
    },
    # middle/common resolution
    {
        "prompt": "cowboy running in slow motion in a field  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 704,
        "height": 480,
        "guidance_scale": 3,
        "guidance_rescale": 0.3,
    },
    # big resolution
    {
        "prompt": "House in front of a lake and the wind blowing through the trees  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 1216,
        "height": 704,
        "guidance_scale": 3,
        "guidance_rescale": 0.3,
    },
    # guidance_rescale 0
    {
        "prompt": "fight naruto vs saske  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 704,
        "height": 480,
        "guidance_scale": 3,
        "guidance_rescale": 0,
    },
    # guidance_scale 1
    {
        "prompt": "reporter in front of the TV cameras talking about the joker  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 704,
        "height": 480,
        "guidance_scale": 1,
        "guidance_rescale": 0.3,
    },
    # guidance_scale 1 guidance_rescale 0
    {
        "prompt": "Realistic night silhouette of a white Lwxux LX III 2008 with headlights on driving on in the fog in the dark  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 704,
        "height": 480,
        "guidance_scale": 1,
        "guidance_rescale": 0,
    },
    # guidance_scale 1 guidance_rescale 0
    {
        "prompt": "indian women washing clothes at river side  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 256,
        "height": 128,
        "guidance_scale": 3,
        "guidance_rescale": 0.3,
    },
    # big prompt
    {
        "prompt": "Levitating woman uses magic and fairy dusty spews forth from her fingers.  cinematic shot  photos taken by ARRI, photos taken "
        + "by sony, photos taken by canon, photos taken by nikon, photos taken by sony, photos taken by hasselblad  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 704,
        "height": 480,
        "guidance_scale": 3,
        "guidance_rescale": 0.3,
    },
    # big prompt, small resolution
    {
        "prompt": "A mythical river adventure in the Yellow River basin during ancient times, where majestic dragons soar through the turbulent waters, "
        + "casting a vibrant glow on the submerged landscapes, blending a sense of awe and fantasy, Sculpture, intricate clay model with luminescent "
        + "elements, --ar 16:9 --v 5  ",
        "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "width": 256,
        "height": 128,
        "guidance_scale": 3,
        "guidance_rescale": 0.3,
    },
]


@register_evaluator("text-to-video")
class Text2VideoEvaluator(BaseEvaluator):
    DEF_NUM_FRAMES = 25
    DEF_NUM_INF_STEPS = 25
    DEF_FRAME_RATE = 25
    DEF_WIDTH = 704
    DEF_HEIGHT = 480
    DEF_GUIDANCE_SCALE = 3
    DEF_GUIDANCE_RESCALE = 0.3

    def __init__(
        self,
        base_model: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        num_inference_steps=25,
        num_frames=25,
        crop_prompts=True,
        num_samples=None,
        gen_video_fn=None,
        seed=42,
        is_genai=False,
    ) -> None:
        assert base_model is not None or gt_data is not None, (
            "Text generation pipeline for evaluation or ground truth data must be defined"
        )

        self.test_data = test_data
        self.metrics = metrics
        self.crop_prompt = crop_prompts
        self.num_samples = num_samples
        self.num_inference_steps = num_inference_steps or self.DEF_NUM_INF_STEPS
        self.seed = seed
        self.similarity = VideoSimilarity()
        self.last_cmp = None
        self.gt_dir = os.path.dirname(gt_data)
        self.generation_fn = gen_video_fn
        self.is_genai = is_genai
        self.num_frames = num_frames or self.DEF_NUM_FRAMES
        self.frame_rate = self.DEF_FRAME_RATE

        if base_model:
            self.gt_data = self._generate_data(base_model, gen_video_fn, os.path.join(self.gt_dir, "reference"))
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_video_fn=None, output_dir=None, **kwargs):
        if output_dir is None:
            video_folder = os.path.join(self.gt_dir, "target")
        else:
            video_folder = os.path.join(output_dir, "target")

        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_video_fn, video_folder)
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_frame = self.similarity.evaluate(self.gt_data, predictions)
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_frame)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompt"].values
        self.last_cmp["source_model"] = self.gt_data["videos"].values
        self.last_cmp["optimized_model"] = predictions["videos"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        res = self.last_cmp.nsmallest(top_k, metric)
        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(self, model, gen_video_fn=None, videos_dir="reference"):
        def default_gen_video_fn(
            model,
            prompt,
            negative_prompt,
            num_inference_steps,
            width=self.DEF_WIDTH,
            height=self.DEF_HEIGHT,
            num_frames=self.DEF_NUM_FRAMES,
            frame_rate=self.DEF_FRAME_RATE,
            guidance_scale=self.DEF_GUIDANCE_SCALE,
            guidance_rescale=self.DEF_GUIDANCE_RESCALE,
            generator=None,
        ):
            with torch.no_grad():
                output = model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    generator=generator,
                )
            return output.frames[0]

        generation_fn = gen_video_fn or default_gen_video_fn

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "prompt" in self.test_data
                    data = dict(self.test_data)
                else:
                    data = {"prompt": list(self.test_data)}
                data = pd.DataFrame.from_dict(data)
        else:
            data = pd.DataFrame.from_dict(default_data)

        inputs = data.values if self.num_samples is None else data.values[: self.num_samples]
        res_data = dict(zip(list(default_data[0].keys()), map(list, zip(*inputs))))
        videos = []

        rng = torch.Generator(device="cpu")

        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)

        for i, input in tqdm(enumerate(inputs), desc="Evaluate pipeline"):
            set_seed(self.seed)
            rng = rng.manual_seed(self.seed)
            frames = generation_fn(
                model,
                prompt=input[0],
                negative_prompt=input[1],
                num_inference_steps=self.num_inference_steps,
                width=input[2],
                height=input[3],
                num_frames=self.num_frames,
                frame_rate=self.frame_rate,
                guidance_scale=input[4],
                guidance_rescale=input[5],
                generator=openvino_genai.TorchGenerator(self.seed) if self.is_genai else rng,
            )
            video_path = os.path.join(videos_dir, f"video_{i}.mp4")
            export_to_video(frames, video_path, self.frame_rate)
            videos.append(video_path)

        res_data["videos"] = videos
        df = pd.DataFrame(res_data)

        return df
