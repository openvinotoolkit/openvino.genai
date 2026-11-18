#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import cv2
import numpy as np
import openvino as ov
import openvino_genai
from video_utils import save_audio, save_video


def read_video(path: str):
    capture = cv2.VideoCapture(path)
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames), frame_rate


def main():
    parser = argparse.ArgumentParser(description="Edit or prolong a video with LTX-2")
    parser.add_argument("model_dir", help="Path to the LTX-2 model directory (with vae_encoder/)")
    parser.add_argument("video", help="Source video")
    parser.add_argument("prompt", help="Description of the video to generate, including its sound")
    parser.add_argument(
        "--edit-steps",
        type=int,
        help="Edit the whole video: noise it and run this many last steps of the schedule (e.g. 4 of LTX-2.5's 8)",
    )
    parser.add_argument(
        "--prolong-to",
        type=int,
        help="Prolong the video to this many frames (8 * k + 1), continuing from its last frames",
    )
    parser.add_argument("--context-frames", type=int, default=33, help="Last frames that the continuation keeps")
    # LTX-2.5 ships a prompt enhancer that rewrites a short prompt into the detailed description the model was trained
    # on, looking at the conditioning frames as well. It is opt-in: generate(..., enable_prompt_enhancement=True)
    parser.add_argument(
        "--enhance-prompt", action="store_true", help="Rewrite the prompt with the model's prompt enhancer (LTX-2.5)"
    )
    args = parser.parse_args()
    if not args.edit_steps and not args.prolong_to:
        parser.error("set --edit-steps, --prolong-to or both")

    frames, frame_rate = read_video(args.video)
    video = ov.Tensor(frames)  # [F, H, W, 3] uint8
    height, width = frames.shape[1] // 32 * 32, frames.shape[2] // 32 * 32

    pipe = openvino_genai.Video2VideoPipeline(args.model_dir, "CPU")  # GPU can be used as well

    def callback(step, num_steps, latent):
        print(f"Generation step {step + 1} / {num_steps}")
        return False

    kwargs = dict(height=height, width=width, frame_rate=frame_rate, callback=callback,
                  enable_prompt_enhancement=args.enhance_prompt, generator=openvino_genai.CppStdGenerator(42))
    if args.edit_steps:
        kwargs["edit_num_steps"] = args.edit_steps
    if args.prolong_to:
        # The output starts with the last frames of the source and continues them
        context = ov.Tensor(np.ascontiguousarray(frames[-args.context_frames:]))
        kwargs["conditions"] = [openvino_genai.VideoCondition(context, index=0)]
        kwargs["num_frames"] = args.prolong_to

    output = pipe.generate(video, args.prompt, **kwargs)

    save_video("genai_video.avi", output.video, frame_rate)
    # The soundtrack is generated, the source audio is not used. To combine both into one file:
    #   ffmpeg -i genai_video.avi -i genai_audio.wav genai_video.mp4
    save_audio("genai_audio.wav", output.audio, output.audio_sample_rate)


if __name__ == "__main__":
    main()
