#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = lhs.reshape(-1)
    rhs = rhs.reshape(-1)
    lhs_norm = np.linalg.norm(lhs)
    rhs_norm = np.linalg.norm(rhs)
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))


def read_video(path: str, num_frames: int) -> tuple[ov.Tensor, openvino_genai.VideoMetadata]:
    cap = cv2.VideoCapture(path)
    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_num_frames <= 0:
        cap.release()
        raise RuntimeError(f"Failed to read frames from video: {path}")

    sampled_indices = np.linspace(0, total_num_frames - 1, num=min(num_frames, total_num_frames), dtype=int)
    sampled_set = set(sampled_indices.tolist())

    frames: list[np.ndarray] = []
    actual_indices: list[int] = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sampled_set:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            actual_indices.append(frame_idx)
        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError(f"No sampled frames were collected from video: {path}")

    video_metadata = openvino_genai.VideoMetadata()
    video_metadata.fps = fps
    video_metadata.frames_indices = actual_indices

    return ov.Tensor(np.stack(frames, axis=0)), video_metadata


def embed_image(pipeline: openvino_genai.EmbeddingPipeline, path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image_tensor = ov.Tensor(np.array(image))
    return np.asarray(pipeline.embed(images=[image_tensor]).embeddings.data, dtype=np.float32)


def embed_video(pipeline: openvino_genai.EmbeddingPipeline, path: str, num_video_frames: int) -> np.ndarray:
    video_tensor, video_metadata = read_video(path, num_video_frames)
    return np.asarray(
        pipeline.embed(videos=[video_tensor], videos_metadata=[video_metadata]).embeddings.data,
        dtype=np.float32,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to multimodal embedding model directory")
    parser.add_argument(
        "--query",
        required=True,
        help="Text query used to find the most similar image or video",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Image paths to compare with the query",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=[],
        help="Video paths to compare with the query",
    )
    parser.add_argument(
        "--num-video-frames",
        type=int,
        default=8,
        help="Number of video frames to sample",
    )
    parser.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    if not args.images and not args.videos:
        parser.error("At least one input must be provided via --images or --videos")

    pipeline = openvino_genai.EmbeddingPipeline(args.model_dir, args.device)
    query_embedding = np.asarray(pipeline.embed(args.query).embeddings.data, dtype=np.float32)

    results: list[tuple[float, str, str]] = []
    for image_path in args.images:
        image_embedding = embed_image(pipeline, image_path)
        results.append((cosine_similarity(query_embedding, image_embedding), "image", image_path))

    for video_path in args.videos:
        video_embedding = embed_video(pipeline, video_path, args.num_video_frames)
        results.append((cosine_similarity(query_embedding, video_embedding), "video", video_path))

    results.sort(reverse=True, key=lambda item: item[0])

    print("Query:", args.query)
    print("Ranked inputs by cosine similarity:")
    for rank, (score, input_type, input_path) in enumerate(results, start=1):
        print(f"{rank}. {input_type}: {Path(input_path).resolve()} similarity={score:.6f}")

    best_score, best_type, best_path = results[0]
    print("Most similar input:", best_type, Path(best_path).resolve(), f"similarity={best_score:.6f}")


if __name__ == "__main__":
    main()
