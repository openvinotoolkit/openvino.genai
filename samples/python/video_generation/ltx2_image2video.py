#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image
from video_utils import save_audio, save_video


def default_image_crf(model_dir: str) -> int:
    """The H.264 CRF LTX-2 image conditions were trained with: 18 for LTX-2.5 (Gemma 4), 33 before."""
    text_encoder_config = json.loads(Path(model_dir, "text_encoder", "config.json").read_text())
    return 18 if text_encoder_config.get("model_type", "").startswith("gemma4") else 33


def apply_image_crf(image: np.ndarray, crf: int) -> np.ndarray:
    """Re-encodes an image as one H.264 frame, as diffusers' LTX-2 pipelines do before encoding it."""
    if crf == 0:
        return image
    try:
        from diffusers.pipelines.ltx2.utils import apply_image_conditioning_crf

        return apply_image_conditioning_crf(image, crf)
    except ImportError:
        pass
    import av  # pip install av

    height, width = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    image = image[:height, :width]
    with BytesIO() as output_file:
        container = av.open(output_file, "w", format="mp4")
        try:
            stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
            stream.height, stream.width = height, width
            frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
            container.mux(stream.encode(frame))
            container.mux(stream.encode())
        finally:
            container.close()
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        container = av.open(video_file)
        try:
            stream = next(s for s in container.streams if s.type == "video")
            return next(container.decode(stream)).to_ndarray(format="rgb24")
        finally:
            container.close()


def main():
    parser = argparse.ArgumentParser(description="Generate a video through one or more images with LTX-2")
    parser.add_argument("model_dir", help="Path to the LTX-2 model directory (with vae_encoder/)")
    parser.add_argument("prompt", help="Description of the video, including its sound")
    parser.add_argument("first_image", help="Image the video starts with")
    parser.add_argument("--last-image", help="Image the video ends with")
    parser.add_argument("--num-frames", type=int, default=121, help="Number of frames, 8 * k + 1")
    parser.add_argument("--crf", type=int, help="H.264 CRF applied to the images, 0 to skip (default: the model's)")
    # LTX-2.5 ships a prompt enhancer that rewrites a short prompt into the detailed description the model was trained
    # on, looking at the conditioning frames as well. It is opt-in: generate(..., enable_prompt_enhancement=True)
    parser.add_argument(
        "--enhance-prompt", action="store_true", help="Rewrite the prompt with the model's prompt enhancer (LTX-2.5)"
    )
    args = parser.parse_args()

    crf = default_image_crf(args.model_dir) if args.crf is None else args.crf

    def read_image(path):
        return ov.Tensor(apply_image_crf(np.array(Image.open(path).convert("RGB")), crf))

    pipe = openvino_genai.Image2VideoPipeline(args.model_dir, "CPU")  # GPU can be used as well

    def callback(step, num_steps, latent):
        print(f"Generation step {step + 1} / {num_steps}")
        return False

    # The model's defaults are used for the rest: LTX-2.5 runs its distilled 8-step schedule without guidance
    kwargs = dict(
        num_frames=args.num_frames,
        enable_prompt_enhancement=args.enhance_prompt,
        callback=callback,
        generator=openvino_genai.CppStdGenerator(42),
    )
    if args.last_image:
        images = [
            openvino_genai.VideoCondition(read_image(args.first_image), index=0),
            openvino_genai.VideoCondition(read_image(args.last_image), index=-1),
        ]
        output = pipe.generate(images, args.prompt, **kwargs)
    else:
        output = pipe.generate(read_image(args.first_image), args.prompt, **kwargs)

    frame_rate = pipe.get_generation_config().frame_rate
    save_video("genai_video.avi", output.video, frame_rate)
    # To combine both into one file: ffmpeg -i genai_video.avi -i genai_audio.wav genai_video.mp4
    save_audio("genai_audio.wav", output.audio, output.audio_sample_rate)


if __name__ == "__main__":
    main()
