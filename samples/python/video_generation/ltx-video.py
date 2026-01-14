#!/usr/bin/env python3

import argparse
import torch
import numpy as np
import openvino_genai
from diffusers.utils import export_to_video

try:
    from diffusers import LTXPipeline
except ImportError:
    LTXPipeline = None


def generate_with_genai(pipe, prompt: str, negative_prompt: str, frame_rate: int):
    output = pipe.generate(
        prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=704,
        num_frames=161,
        frame_rate=frame_rate,
        num_inference_steps=25,
        generator=openvino_genai.TorchGenerator(42),
        guidance_scale=3,
    )

    video_data = output.video.data
    frames = [video_data[0, f] for f in range(video_data.shape[1])]
    return frames


def generate_with_diffusers(pipeline, prompt: str, negative_prompt: str, frame_rate: int):
    ltx_pipeline_output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=704,
        num_frames=161,
        frame_rate=frame_rate,
        num_inference_steps=25,
        generator=torch.Generator(device="cpu").manual_seed(42),
        guidance_scale=3,
    )
    return ltx_pipeline_output.frames[0]


def main():
    parser = argparse.ArgumentParser(description="Compare OpenVINO GenAI and Diffusers LTX-Video outputs")
    parser.add_argument("model_dir", help="Path to the OpenVINO model directory")
    parser.add_argument("--skip-diffusers", action="store_true", help="Skip Diffusers reference generation")
    args = parser.parse_args()

    frame_rate = 25
    prompt = "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage."
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    print("Generating with OpenVINO GenAI...")
    ov_pipe = openvino_genai.Text2VideoPipeline(args.model_dir, "CPU")
    ov_video = generate_with_genai(ov_pipe, prompt, negative_prompt, frame_rate)
    export_to_video(ov_video, "genai_video.mp4", fps=frame_rate)
    print(f"Saved genai_video.mp4")

    if not args.skip_diffusers:
        if LTXPipeline is None:
            raise ImportError("diffusers package is required when not using --skip-diffusers")
        
        print("\nGenerating with Diffusers...")
        diffusers_pipeline = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float32)
        diffusers_video = generate_with_diffusers(diffusers_pipeline, prompt, negative_prompt, frame_rate)
        export_to_video(diffusers_video, "diffusers_video.mp4", fps=frame_rate)
        print(f"Saved diffusers_video.mp4")

        print("\nComparing outputs...")
        max_diff = np.abs(np.stack(ov_video, dtype=np.int16) - np.stack(diffusers_video, dtype=np.int16)).max()
        print(f"Maximum pixel difference: {max_diff}")

        if max_diff <= 9:
            print("✓ Videos match within tolerance")
        else:
            print(f"✗ Videos differ by more than expected (max_diff={max_diff} > 9)")
    else:
        print("\nSkipping Diffusers comparison")


if __name__ == "__main__":
    main()
