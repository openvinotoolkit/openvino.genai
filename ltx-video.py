import torch
import numpy as np
from optimum.intel.openvino import OVDiffusionPipeline, OVLTXPipeline
from diffusers import LTXPipeline
from diffusers.utils import export_to_video


def main():
    prompt = "Will Smith eating spaghetti"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    diffusers_pipeline = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float32)
    diffusers_video = diffusers_pipeline(
        prompt=prompt,
        # prompt_embeds=0,
        # prompt_attention_mask=1,
        negative_prompt=negative_prompt,
        height=512,
        width=704,
        num_frames=9,
        frame_rate=25,
        # num_inference_steps=25,
        num_inference_steps=2,
        generator=torch.Generator(device="cpu").manual_seed(42),
        guidance_scale=3,
    ).frames[0]
    export_to_video(diffusers_video, "diffusers_video.mp4", fps=25)

    ov_pipe = OVLTXPipeline.from_pretrained('./i/LTX-Video/', device='CPU', load_in_8bit=False)
    ov_video = ov_pipe(
        prompt=prompt,
        # prompt_embeds=0,
        # prompt_attention_mask=1,
        negative_prompt=negative_prompt,
        height=512,
        width=704,
        num_frames=9,
        frame_rate=25,
        # num_inference_steps=25,
        num_inference_steps=2,
        generator=torch.Generator(device="cpu").manual_seed(42),
        guidance_scale=3,
    ).frames[0]
    export_to_video(diffusers_video, "ov_video.mp4", fps=25)

    assert np.abs(
        np.stack(ov_video, dtype=np.int16) - np.stack(diffusers_video, dtype=np.int16)
    ).max() <= 9


if "__main__" == __name__:
    main()
