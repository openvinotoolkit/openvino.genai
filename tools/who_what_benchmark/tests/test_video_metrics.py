"""Offline tests for the reference-free video metrics: no model download, tiny synthetic videos."""

import numpy as np
import pandas as pd
import pytest
from diffusers.utils import export_to_video
from whowhatbench.text2video_evaluator import Text2VideoEvaluator
from whowhatbench.video_metrics import ObjectPresence, VideoQuality, read_frames, temporal_flicker


def _write_video(path, frames, shift=0):
    """A moving gradient; `shift` changes the content so two videos differ."""
    yy, xx = np.mgrid[0:32, 0:48]
    video = [
        np.stack([(xx * 5 + 7 * t + shift) % 256, (yy * 8) % 256, np.full_like(xx, 90)], -1).astype(np.uint8)
        for t in range(frames)
    ]
    export_to_video(video, str(path), fps=8)
    return str(path)


@pytest.fixture
def two_videos(tmp_path):
    return _write_video(tmp_path / "base.mp4", 9), _write_video(tmp_path / "target.mp4", 9, shift=40)


def test_read_frames_and_flicker(two_videos):
    frames = read_frames(two_videos[0])
    assert frames.shape == (9, 32, 48, 3) and frames.dtype == np.uint8
    assert read_frames(two_videos[0], max_frames=4).shape[0] == 4
    still = np.repeat(frames[:1], 3, axis=0)
    assert temporal_flicker(still) == pytest.approx(1.0)
    assert 0.0 < temporal_flicker(frames) < 1.0


def test_video_quality_reports_base_target_delta(two_videos, monkeypatch):
    metric = VideoQuality(metrics=["imaging_quality", "temporal_flicker"])
    monkeypatch.setattr(metric, "imaging_quality", lambda frames: float(frames.mean()))  # no MUSIQ download
    gold = pd.DataFrame({"videos": [two_videos[0]]})
    pred = pd.DataFrame({"videos": [two_videos[1]]})

    aggregate, per_video = metric.evaluate(gold, pred)

    assert set(per_video) == {
        "imaging_quality",
        "imaging_quality_base",
        "imaging_quality_delta",
        "temporal_flicker",
        "temporal_flicker_base",
        "temporal_flicker_delta",
    }
    assert per_video["imaging_quality_delta"][0] == pytest.approx(
        per_video["imaging_quality"][0] - per_video["imaging_quality_base"][0]
    )
    assert aggregate["temporal_flicker"] == pytest.approx(per_video["temporal_flicker"][0])


def test_video_quality_rejects_unknown_metric():
    with pytest.raises(ValueError):
        VideoQuality(metrics=["sharpness"])


def test_object_presence_scores_and_lost_objects(two_videos, monkeypatch):
    metric = ObjectPresence(threshold=0.5, max_frames=4)
    # fake detector: the base video shows both objects in every frame, the target lost the second one
    scores = {two_videos[0]: np.array([[0.9, 0.8]] * 4), two_videos[1]: np.array([[0.9, 0.1]] * 4)}
    monkeypatch.setattr(metric, "frame_scores", lambda frames, objects: scores[metric._current])
    original = metric.score_video

    def score_video(path, objects):
        metric._current = path
        return original(path, objects)

    monkeypatch.setattr(metric, "score_video", score_video)
    gold = pd.DataFrame({"videos": [two_videos[0]], "objects": ["couch; tie"]})
    pred = pd.DataFrame({"videos": [two_videos[1]], "objects": ["couch; tie"]})

    aggregate, per_video = metric.evaluate(gold, pred)

    assert per_video["object_presence_base"] == [1.0]
    assert per_video["object_presence"] == [0.5]
    assert per_video["objects_lost"] == ["tie"]
    assert aggregate["object_presence_delta"] == pytest.approx(-0.5)


def test_object_presence_without_objects_column_is_skipped(two_videos):
    gold = pd.DataFrame({"videos": [two_videos[0]]})
    aggregate, per_video = ObjectPresence().evaluate(gold, gold)
    assert aggregate == {} and per_video == {}


def test_parse_objects():
    assert ObjectPresence.parse_objects("couch; tie ;; chair") == ["couch", "tie", "chair"]
    assert ObjectPresence.parse_objects(float("nan")) == []
    assert ObjectPresence.parse_objects(None) == []


def test_evaluator_reads_prompt_columns_by_name(tmp_path):
    seen = []

    def gen_video_fn(
        model,
        prompt,
        negative_prompt,
        num_inference_steps,
        width,
        height,
        num_frames,
        frame_rate,
        guidance_scale,
        guidance_rescale,
        generator,
        empty_adapters,
    ):
        seen.append((prompt, negative_prompt, width, height, guidance_scale, guidance_rescale))
        return [np.zeros((height, width, 3), dtype=np.uint8)] * 3

    evaluator = Text2VideoEvaluator(
        base_model=object(),
        gt_data=str(tmp_path / "gt.csv"),
        metrics=[],
        num_samples=2,
        num_frames=3,
        gen_video_fn=gen_video_fn,
    )

    assert list(evaluator.gt_data.columns) == [
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "guidance_scale",
        "objects",
        "videos",
    ]
    assert seen[0][2:] == (256, 128, 3, Text2VideoEvaluator.DEF_GUIDANCE_RESCALE)
    assert evaluator.gt_data["objects"][0] == "woman; blue jacket; black hat"
    assert evaluator.similarity is None and evaluator.extra_metrics == []

    # a dataset with only prompts falls back to the defaults for the other columns
    evaluator = Text2VideoEvaluator(
        base_model=object(),
        gt_data=str(tmp_path / "gt2.csv"),
        metrics=[],
        test_data={"prompt": ["a cat"]},
        num_frames=3,
        gen_video_fn=gen_video_fn,
    )
    assert seen[-1] == (
        "a cat",
        "",
        Text2VideoEvaluator.DEF_WIDTH,
        Text2VideoEvaluator.DEF_HEIGHT,
        Text2VideoEvaluator.DEF_GUIDANCE_SCALE,
        Text2VideoEvaluator.DEF_GUIDANCE_RESCALE,
    )
    assert "objects" not in evaluator.gt_data.columns
