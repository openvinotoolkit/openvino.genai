import torch
import openvino
import numpy as np
from optimum.intel.openvino import OVLTXPipeline  # OVDiffusionPipeline
from diffusers import LTXPipeline
from diffusers.utils import export_to_video


def generate(pipeline, frame_rate):
    prompt = "Will Smith eating spaghetti"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    ltx_pipeline_output = pipeline(
        prompt=prompt,
        # prompt_embeds=0,
        # prompt_attention_mask=1,
        negative_prompt=negative_prompt,
        height=512,
        width=704,
        num_frames=9,
        frame_rate=frame_rate,
        # num_inference_steps=25,
        num_inference_steps=2,
        generator=torch.Generator(device="cpu").manual_seed(42),
        guidance_scale=3,
    )
    return ltx_pipeline_output.frames[0]


def main():
    frame_rate = 25

    ov_pipe = OVLTXPipeline.from_pretrained(
        '/home/vzlobin/z/g/i/LTX-Video/',
        device='CPU',
        load_in_8bit=False,
        ov_config={openvino.properties.hint.inference_precision: openvino.Type.f32},
    )
    ov_video = generate(ov_pipe, frame_rate)
    print(ov_video)
    # export_to_video(ov_video, "ov_video.mp4", fps=frame_rate)

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
