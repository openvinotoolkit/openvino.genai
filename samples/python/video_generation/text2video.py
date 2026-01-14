#!/usr/bin/env python3

import argparse
import numpy as np
import cv2
import openvino_genai


def save_video(filename: str, video_tensor, fps: int = 25):
    shape = video_tensor.shape
    if len(shape) != 5:
        raise ValueError(f"Expected video tensor with shape [B, F, H, W, C], got {shape}")

    batch_size, num_frames, height, width, channels = shape

    if channels not in [1, 3, 4]:
        raise ValueError(f"Channels must be 1, 3, or 4, got {channels}")

    video_data = video_tensor.data

    for b in range(batch_size):
        if batch_size == 1:
            output_path = filename
        else:
            base, ext = filename.rsplit(".", 1) if "." in filename else (filename, "avi")
            output_path = f"{base}_b{b}.{ext}"

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        for f in range(num_frames):
            frame = video_data[b, f]

            if channels == 1:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif channels == 3:
                frame_rgb = frame
            else:
                frame_rgb = frame[:, :, :3]

            writer.write(frame_rgb)

        writer.release()
        print(f"Wrote {output_path} ({num_frames} frames, {width}x{height} @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Generate video from text prompt using OpenVINO GenAI")
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("prompt", help="Text prompt for video generation")
    parser.add_argument(
        "--device", default="CPU", choices=["CPU", "GPU"], help="Device to run inference on (default: CPU)"
    )
    parser.add_argument("--output", default="genai_video.avi", help="Output video filename (default: genai_video.avi)")
    args = parser.parse_args()

    pipe = openvino_genai.Text2VideoPipeline(args.model_dir, args.device)

    frame_rate = 25

    def callback(step, num_steps, latent):
        print(f"Generation step {step + 1} / {num_steps}")
        return False

    output = pipe.generate(
        args.prompt,
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        height=480,
        width=704,
        num_frames=161,
        num_inference_steps=25,
        num_videos_per_prompt=1,
        callback=callback,
        frame_rate=frame_rate,
        guidance_scale=3,
    )

    save_video(args.output, output.video, frame_rate)

    print(f"\nPerformance metrics:")
    print(f"  Load time: {output.perf_metrics.load_time:.2f} ms")
    print(f"  Generate duration: {output.perf_metrics.get_generate_duration().mean:.2f} ms")
    print(f"  Text encoder duration: {output.perf_metrics.get_text_encoder_duration().mean:.2f} ms")
    print(f"  Denoising duration: {output.perf_metrics.get_denoising_duration().mean:.2f} ms")
    print(f"  Decoding duration: {output.perf_metrics.get_decoding_duration().mean:.2f} ms")


if __name__ == "__main__":
    main()
