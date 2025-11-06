import argparse
import torch
import openvino
import numpy as np
from optimum.intel.openvino import OVLTXPipeline  # OVDiffusionPipeline
from diffusers import LTXPipeline
from diffusers.utils import export_to_video


def generate(pipeline, frame_rate):
    # prompt = "Will Smith eating spaghetti"
    prompt = "A woman with long brown hair and light skin smiles at another woman...A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage."
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    ltx_pipeline_output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=704,
        num_frames=65,
        frame_rate=frame_rate,
        num_inference_steps=15,
        generator=torch.Generator(device="cpu").manual_seed(42),
        guidance_scale=3,
    )
    return ltx_pipeline_output.frames[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to the model directory')
    args = parser.parse_args()
    frame_rate = 25

    ov_pipe = OVLTXPipeline.from_pretrained(
        args.model_dir,
        device='CPU',
        load_in_8bit=False,
        ov_config={openvino.properties.hint.inference_precision: openvino.Type.f32},
    )
    ov_video = generate(ov_pipe, frame_rate)
    print(ov_video)
    export_to_video(ov_video, "5_opt_video.mp4", fps=frame_rate)

    # diffusers_pipeline = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float32)
    # diffusers_video = generate(diffusers_pipeline, frame_rate)
    # export_to_video(diffusers_video, "diffusers_video.mp4", fps=frame_rate)

    # max_diff = np.abs(
    #     np.stack(ov_video, dtype=np.int16) - np.stack(diffusers_video, dtype=np.int16)
    # ).max()
    # print(max_diff)
    # assert max_diff <= 9


if "__main__" == __name__:
    main()
