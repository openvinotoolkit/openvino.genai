"""Reference-free metrics for generated videos.

`VideoSimilarity` measures how far a target video is from the base model's video. It cannot say whether
a changed video is still good, or whether it still shows what the prompt asked for. The metrics here
score every video on its own, for the base and the target, and report both plus the difference.

VideoQuality
    imaging_quality   MUSIQ (SPAQ weights), mean over frames, 0..100. Needs the optional `pyiqa` package.
    aesthetic_quality LAION aesthetic linear head on CLIP ViT-L/14 image features, divided by 10 (about 0..1).
    temporal_flicker  (255 - mean |frame[t+1] - frame[t]|) / 255 over all frames, 0..1. Mixes motion with
                      flicker on moving scenes, so compare it per prompt rather than across prompts.
    The formulas follow VBench (Huang et al., CVPR 2024), so per-video values are comparable with VBench's.

ObjectPresence
    For the objects listed with a prompt (column "objects", "; "-separated), the fraction of sampled frames
    in which an open-vocabulary detector (OWLv2) finds the object. "object_presence" averages over the
    objects; "objects_lost" names the objects the base model shows and the target does not.
"""

import json
import logging
import os
import urllib.request

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get("WWB_METRICS_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "whowhatbench"))
MUSIQ_URL = "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth"
AESTHETIC_URL = "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DETECTOR_MODEL_ID = "google/owlv2-base-patch16-ensemble"


def _fetch(url):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, os.path.basename(url))
    if not os.path.isfile(path):
        logger.info("Downloading %s", url)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def _pooled(features):
    # transformers < 5 returns a tensor, >= 5 an output object whose pooler_output holds the projected features
    return features if isinstance(features, torch.Tensor) else features.pooler_output


def read_frames(video_path, max_frames=None):
    """All frames of a video as uint8 RGB (T, H, W, 3); with `max_frames`, evenly spaced frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    frames = np.stack(frames)
    if max_frames and len(frames) > max_frames:
        frames = frames[np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    return frames


def temporal_flicker(frames):
    diffs = [
        np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32)).mean() for i in range(len(frames) - 1)
    ]
    return (255.0 - float(np.mean(diffs))) / 255.0 if diffs else float("nan")


def _paired_results(name, base_values, target_values):
    """Per-video and aggregate entries for one metric scored on both models."""
    deltas = [t - b for b, t in zip(base_values, target_values)]
    per_video = {name: target_values, f"{name}_base": base_values, f"{name}_delta": deltas}
    aggregate = {
        name: float(np.nanmean(target_values)),
        f"{name}_base": float(np.nanmean(base_values)),
        f"{name}_delta": float(np.nanmean(deltas)),
    }
    return aggregate, per_video


class VideoQuality:
    METRICS = ("imaging_quality", "aesthetic_quality", "temporal_flicker")

    def __init__(self, metrics=METRICS, max_frames=16, device="cpu"):
        unknown = set(metrics) - set(self.METRICS)
        if unknown:
            raise ValueError(f"Unknown video quality metrics: {sorted(unknown)}")
        self.metrics = list(metrics)
        self.max_frames = max_frames
        self.device = device
        self._musiq = None
        self._clip = None
        self._aesthetic_head = None

    def _musiq_model(self):
        if self._musiq is None:
            try:
                from pyiqa.archs.musiq_arch import MUSIQ
            except ImportError as exc:
                raise ImportError("imaging_quality needs the pyiqa package: pip install pyiqa") from exc
            state = torch.load(_fetch(MUSIQ_URL), map_location="cpu")
            self._musiq = MUSIQ(pretrained=False).eval().to(self.device)
            self._musiq.load_state_dict(state.get("params", state))
        return self._musiq

    def _clip_model(self):
        if self._clip is None:
            from transformers import CLIPImageProcessor, CLIPModel

            self._clip = (
                CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID),
                CLIPModel.from_pretrained(CLIP_MODEL_ID).eval().to(self.device),
            )
            self._aesthetic_head = torch.nn.Linear(768, 1)
            self._aesthetic_head.load_state_dict(torch.load(_fetch(AESTHETIC_URL), map_location="cpu"))
            self._aesthetic_head.eval().to(self.device)
        return self._clip

    @torch.no_grad()
    def imaging_quality(self, frames):
        from torchvision.transforms import Resize

        model = self._musiq_model()
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        _, _, h, w = x.shape
        if max(h, w) > 512:  # VBench 'longer' preprocessing
            scale = 512.0 / max(h, w)
            x = Resize((int(scale * h), int(scale * w)), antialias=False)(x)
        x = (x / 255.0).to(self.device)
        return float(np.mean([float(model(frame.unsqueeze(0))) for frame in x]))

    @torch.no_grad()
    def aesthetic_quality(self, frames):
        processor, model = self._clip_model()
        scores = []
        for start in range(0, len(frames), 32):
            inputs = processor(images=list(frames[start : start + 32]), return_tensors="pt").to(self.device)
            features = _pooled(model.get_image_features(**inputs)).float()
            features = features / features.norm(dim=-1, keepdim=True)
            scores.append(self._aesthetic_head(features).squeeze(-1))
        return float(torch.cat(scores).mean()) / 10.0

    def score_video(self, video_path):
        frames = read_frames(video_path)
        sampled = frames[np.linspace(0, len(frames) - 1, min(self.max_frames, len(frames))).round().astype(int)]
        scores = {}
        if "imaging_quality" in self.metrics:
            scores["imaging_quality"] = self.imaging_quality(sampled)
        if "aesthetic_quality" in self.metrics:
            scores["aesthetic_quality"] = self.aesthetic_quality(sampled)
        if "temporal_flicker" in self.metrics:
            scores["temporal_flicker"] = temporal_flicker(frames)
        return scores

    def evaluate(self, data_gold, data_prediction):
        n = min(len(data_gold), len(data_prediction))
        base = [self.score_video(path) for path in data_gold["videos"].values[:n]]
        target = [self.score_video(path) for path in data_prediction["videos"].values[:n]]
        aggregate, per_video = {}, {}
        for name in self.metrics:
            agg, per = _paired_results(name, [s[name] for s in base], [s[name] for s in target])
            aggregate.update(agg)
            per_video.update(per)
        return aggregate, per_video


class ObjectPresence:
    PRESENT = 0.5  # an object counts as shown when it is found in at least this fraction of the frames

    def __init__(self, model_id=DETECTOR_MODEL_ID, threshold=0.2, max_frames=8, device="cpu"):
        self.model_id = model_id
        self.threshold = threshold
        self.max_frames = max_frames
        self.device = device
        self._detector = None

    @staticmethod
    def parse_objects(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        return [name.strip() for name in str(value).split(";") if name.strip()]

    @staticmethod
    def default_objects():
        """Objects of the built-in prompts, by prompt text."""
        from importlib.resources import files

        prompts = json.loads(files("whowhatbench.prompts").joinpath("text_to_video_prompts.json").read_text())
        return {item["prompt"]: item.get("objects", "") for item in prompts}

    def objects_per_row(self, data_gold, data_prediction, n):
        """Object lists from the "objects" column, or for data collected before that column existed,
        from the built-in prompts; empty lists where nothing is known."""
        for data in (data_prediction, data_gold):
            if "objects" in data.columns:
                return [self.parse_objects(value) for value in data["objects"].values[:n]]
        defaults = self.default_objects()
        for data in (data_prediction, data_gold):
            if "prompt" in data.columns:
                return [self.parse_objects(defaults.get(prompt)) for prompt in data["prompt"].values[:n]]
        return [[] for _ in range(n)]

    def _detector_model(self):
        if self._detector is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._detector = (
                AutoProcessor.from_pretrained(self.model_id),
                AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).eval().to(self.device),
            )
        return self._detector

    @torch.no_grad()
    def frame_scores(self, frames, objects):
        """Highest detection score of every object in every frame, shape (frames, objects)."""
        from PIL import Image

        processor, model = self._detector_model()
        queries = [f"a photo of a {name}" for name in objects]
        scores = []
        for frame in frames:
            inputs = processor(text=[queries], images=Image.fromarray(frame), return_tensors="pt").to(self.device)
            logits = model(**inputs).logits[0]  # (patches, queries)
            scores.append(torch.sigmoid(logits).max(dim=0).values[: len(objects)].cpu().numpy())
        return np.stack(scores)

    def score_video(self, video_path, objects):
        """Fraction of frames in which each object is detected, in the order of `objects`."""
        frames = read_frames(video_path, self.max_frames)
        return (self.frame_scores(frames, objects) >= self.threshold).mean(axis=0)

    def evaluate(self, data_gold, data_prediction):
        n = min(len(data_gold), len(data_prediction))
        objects_per_row = self.objects_per_row(data_gold, data_prediction, n)
        if not any(objects_per_row):
            logger.warning("object_presence: no objects known for these prompts (no 'objects' column), metric skipped")
            return {}, {}
        base, target, lost = [], [], []
        for i, objects in enumerate(objects_per_row):
            if not objects:
                base.append(float("nan"))
                target.append(float("nan"))
                lost.append("")
                continue
            b = self.score_video(data_gold["videos"].values[i], objects)
            t = self.score_video(data_prediction["videos"].values[i], objects)
            base.append(float(b.mean()))
            target.append(float(t.mean()))
            lost.append("; ".join(o for o, pb, pt in zip(objects, b, t) if pb >= self.PRESENT and pt < self.PRESENT))
        aggregate, per_video = _paired_results("object_presence", base, target)
        per_video["objects_lost"] = lost
        return aggregate, per_video
