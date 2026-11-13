#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import cv2
import numpy as np
import openvino as ov
import openvino_genai


def main():
    parser = argparse.ArgumentParser(description="Video super-resolution (FlashVSR, x4) using OpenVINO GenAI")
    parser.add_argument("model_dir", help="Path to the model directory exported with genai-cli")
    parser.add_argument("input_video", help="Path to the low-resolution input video")
    parser.add_argument("output_video", nargs="?", default="genai_video2video.avi", help="Output video path")
    parser.add_argument("device", nargs="?", default="CPU", help="Device (GPU can be used as well)")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input_video)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.input_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25

    pipe = openvino_genai.VideoToVideoPipeline(args.model_dir, args.device)
    # The frames are streamed: every chunk is processed as soon as enough input frames are available.
    pipe.start_stream(sparse_attention_ratio=2.0, generator=openvino_genai.CppStdGenerator(42))

    writer = None

    def write(frames: ov.Tensor):
        nonlocal writer
        for frame in frames.data:
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        write(pipe.push_frames(ov.Tensor(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))))
    write(pipe.end_stream())
    if writer is not None:
        writer.release()

    metrics = pipe.get_performance_metrics()
    print(f"Wrote {args.output_video}")
    print(f"  Load time: {metrics.get_load_time():.2f} ms")
    print(f"  Processing time: {metrics.get_generate_duration():.2f} ms")


if __name__ == "__main__":
    main()
