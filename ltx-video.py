# from optimum.intel.openvino import OVDiffusionPipeline, OVLTXPipeline
# import torch
# from diffusers.utils import export_to_video

# ov_pipe = OVLTXPipeline.from_pretrained('./i/LTX-Video/', device='CPU', load_in_8bit=False)

# # prompt = "A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility."
# prompt = "Will Smith eating spaghetti"
# negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
# generator = torch.Generator(device="cpu").manual_seed(42)

# video = ov_pipe(
#     prompt=prompt,
#     # prompt_embeds=0,
#     # prompt_attention_mask=1,
#     negative_prompt=negative_prompt,
#     height=512,
#     width=704,
#     num_frames=9,
#     frame_rate=25,
#     # num_inference_steps=25,
#     num_inference_steps=2,
#     generator=generator,
#     guidance_scale=3,
# ).frames[0]
# export_to_video(video, "output_ov.mp4", fps=25)

############################
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_image, load_video

pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float32)
pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained("Lightricks/ltxv-spatial-upscaler-0.9.8", vae=pipe.vae, torch_dtype=torch.float32)
pipe.vae.enable_tiling()

def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png")
video = load_video(export_to_video([image])) # compress the image using video compression as the model was trained on videos
condition1 = LTXVideoCondition(video=video, frame_index=0)

prompt = "A cute little penguin takes out a book and starts reading it"
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
expected_height, expected_width = 480, 832
downscale_factor = 2 / 3
num_frames = 96

# Part 1. Generate video at smaller resolution
downscaled_height, downscaled_width = int(expected_height * downscale_factor), int(expected_width * downscale_factor)
downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(downscaled_height, downscaled_width)
latents = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=downscaled_width,
    height=downscaled_height,
    num_frames=num_frames,
    num_inference_steps=30,
    generator=torch.Generator().manual_seed(0),
    output_type="latent",
).frames

# Part 2. Upscale generated video using latent upsampler with fewer inference steps
# The available latent upsampler upscales the height/width by 2x
upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
upscaled_latents = pipe_upsample(
    latents=latents,
    output_type="latent"
).frames

# Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
video = pipe(
    conditions=[condition1],
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=upscaled_width,
    height=upscaled_height,
    num_frames=num_frames,
    denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
    num_inference_steps=10,
    latents=upscaled_latents,
    decode_timestep=0.05,
    image_cond_noise_scale=0.025,
    generator=torch.Generator().manual_seed(0),
    output_type="pil",
).frames[0]

# Part 4. Downscale the video to the expected resolution
video = [frame.resize((expected_width, expected_height)) for frame in video]

export_to_video(video, "output.mp4", fps=24)