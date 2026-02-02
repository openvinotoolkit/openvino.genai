from pathlib import Path

import sys
import argparse

# Parse arguments early to fail fast on invalid arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Wan 2.1 Pipeline Sample")
    parser.add_argument('--model_path', default="../../cpp/module_genai/ut_pipelines/Wan2.1-T2V-1.3B-Diffusers",
                        help="Path to the directory of models")
    parser.add_argument('--device', default="GPU", help="Device, default `GPU`, 'CPU' is also supported.")
    parser.add_argument('--prompt', default=None,
                        help="The prompt for generation")
    parser.add_argument('--negative_prompt', default=None,
                        help="The negative prompt for generation")
    parser.add_argument('--height', type=int, default=None, help="Image height")
    parser.add_argument('--width', type=int, default=None, help="Image width")
    parser.add_argument('--steps', type=int, default=None, help="Number of inference steps (default from JSON or 9)")
    parser.add_argument('--guidance_scale', type=float, default=None, help="Guidance scale")
    parser.add_argument('--num_frames', type=int, default=None, help="Number of frames")
    parser.add_argument('--fps', type=int, default=None, help="Frames per second")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    return parser.parse_args()

# Parse args early - will exit on error before heavy imports
args = parse_args()

import torch
# from diffusers import AutoencoderKLWan, WanPipeline, DiffusionPipeline, UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
# from transformers import AutoTokenizer
from diffusers.utils import export_to_video

import openvino as ov
from openvino import Tensor

# from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
# from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from dataclasses import dataclass
from typing import Optional, Union
from diffusers.utils import BaseOutput
# from diffusers.utils.torch_utils import randn_tensor
import ftfy
import regex as re
import html
import openvino_genai
import yaml


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


TEXT_ENCODER_PATH = "text_encoder/text_encoder.xml"
VAE_DECODER_PATH = "vae/vae_decoder.xml"
TRANSFORMER_PATH = "transformer/transformer.xml"

@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for Wan pipelines.

    Args:
        frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    frames: torch.Tensor


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


core = ov.Core()


class OVWanPipeline:
    def __init__(self, model_dir, device_map="CPU", ov_config=None):
        model_dir = Path(model_dir)
        if isinstance(device_map, str):
            device_map = {"transformer": device_map, "text_encoder": device_map, "vae": device_map}
        self.vae = core.compile_model(model_dir / VAE_DECODER_PATH, device_map["vae"], ov_config)
        # super().__init__()

        # self.register_modules(
        #     vae=vae,
        # )

        self.vae_scale_factor_spatial = 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.z_dim = 16
        self.latents_mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        self.latents_std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]
        cfg_data = {
            'global_context': {
                'model_type': 'wan2.1'
            },
            'pipeline_modules': {
                'pipeline_params': {
                    'type': 'ParameterModule',
                    'outputs': [
                        {
                            'name': 'prompt',
                            'type': 'String'
                        },
                        {
                            'name': 'negative_prompt',
                            'type': 'String'
                        },
                        {
                            'name': 'width',
                            'type': 'Int'
                        },
                        {
                            'name': 'height',
                            'type': 'Int'
                        },
                        {
                            'name': 'batch_size',
                            'type': 'Int'
                        },
                        {
                            'name': 'num_images_per_prompt',
                            'type': 'Int'
                        },
                        {
                            'name': 'seed',
                            'type': 'Int'
                        },
                        {
                            'name': 'num_frames',
                            'type': 'Int'
                        },
                        {
                            'name': 'guidance_scale',
                            'type': 'Float'
                        },
                        {
                            'name': 'num_inference_steps',
                            'type': 'Int'
                        },
                        {
                            'name': 'max_sequence_length',
                            'type': 'Int'
                        }
                    ]
                },
                'clip_text_encoder': {
                    'type': 'ClipTextEncoderModule',
                    'device': device_map['text_encoder'],
                    'inputs': [
                        {
                            'name': 'prompt',
                            'type': 'String',
                            'source': 'pipeline_params.prompt'
                        },
                        {
                            'name': 'negative_prompt',
                            'type': 'String',
                            'source': 'pipeline_params.negative_prompt'
                        },
                        {
                            'name': 'num_images_per_prompt',
                            'type': 'Int',
                            'source': 'pipeline_params.num_images_per_prompt'
                        },
                        {
                            'name': 'max_sequence_length',
                            'type': 'Int',
                            'source': 'pipeline_params.max_sequence_length'
                        },
                        {
                            'name': 'guidance_scale',
                            'type': 'Float',
                            'source': 'pipeline_params.guidance_scale'
                        }
                    ],
                    'outputs': [
                        {
                            'name': 'prompt_embeds',
                            'type': 'VecOVTensor'
                        },
                        {
                            'name': 'negative_prompt_embeds',
                            'type': 'VecOVTensor'
                        }
                    ],
                    'params': {
                        'model_path': str(model_dir),
                    }
                },
                'latent_image': {
                    'type': 'RandomLatentImageModule',
                    'device': 'CPU',
                    'inputs': [
                        {
                            'name': 'batch_size',
                            'type': 'Int',
                            'source': 'pipeline_params.batch_size'
                        },
                        {
                            'name': 'num_images_per_prompt',
                            'type': 'Int',
                            'source': 'pipeline_params.num_images_per_prompt'
                        },
                        {
                            'name': 'height',
                            'type': 'Int',
                            'source': 'pipeline_params.height'
                        },
                        {
                            'name': 'width',
                            'type': 'Int',
                            'source': 'pipeline_params.width'
                        },
                        {
                            'name': 'num_frames',
                            'type': 'Int',
                            'source': 'pipeline_params.num_frames'
                        },
                        {
                            'name': 'seed',
                            'type': 'Int',
                            'source': 'pipeline_params.seed'
                        }
                    ],
                    'outputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor'
                        }
                    ],
                    'params': {
                        'model_path': str(model_dir),
                    }
                },
                'denoiser_loop': {
                    'type': 'DenoiserLoopModule',
                    'device': device_map['transformer'],
                    'inputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor',
                            'source': 'latent_image.latents'
                        },
                        {
                            'name': 'prompt_embeds',
                            'type': 'VecOVTensor',
                            'source': 'clip_text_encoder.prompt_embeds'
                        },
                        {
                            'name': 'prompt_embeds_negative',
                            'type': 'VecOVTensor',
                            'source': 'clip_text_encoder.negative_prompt_embeds'
                        },
                        {
                            'name': 'guidance_scale',
                            'type': 'Float',
                            'source': 'pipeline_params.guidance_scale'
                        },
                        {
                            'name': 'num_inference_steps',
                            'type': 'Int',
                            'source': 'pipeline_params.num_inference_steps'
                        }
                    ],
                    'outputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor'
                        }
                    ],
                    'params': {
                        'model_path': str(model_dir),
                    }
                },
                'pipeline_result': {
                    'type': 'ResultModule',
                    'inputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor',
                            'source': 'denoiser_loop.latents'
                        }
                    ]
                }
            }
        }
        yaml_content = yaml.dump(cfg_data)
        self.pipe = openvino_genai.ModulePipeline(config_yaml_content=yaml_content)

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, list[str]] = None,
        negative_prompt: Union[str, list[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        max_sequence_length: int = 512,
    ):
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.pipe.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            batch_size=batch_size,
            num_images_per_prompt=num_videos_per_prompt,
            seed=42,
            num_frames=num_frames,
            max_sequence_length=max_sequence_length
        )

        latents = torch.from_numpy(self.pipe.get_output("latents").data)

        if not output_type == "latent":
            latents_mean = torch.tensor(self.latents_mean).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(self.latents_std).view(1, self.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            video = torch.from_numpy(self.vae(latents)[0])
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)

device_map = {
    "text_encoder": "CPU",
    "transformer": args.device,
    "vae": args.device,
}

ov_pipe = OVWanPipeline(args.model_path, device_map=device_map)

DEFAULT_PROMPT = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
DEFAULT_NEGATIVE_PROMPT = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 480
DEFAULT_NUM_FRAMES = 40
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_FPS = 10

prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
negative_prompt = args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
width = args.width if args.width is not None else DEFAULT_WIDTH
height = args.height if args.height is not None else DEFAULT_HEIGHT
num_inference_steps = args.steps if args.steps is not None else DEFAULT_NUM_INFERENCE_STEPS
guidance_scale = args.guidance_scale if args.guidance_scale is not None else DEFAULT_GUIDANCE_SCALE
num_frames = args.num_frames if args.num_frames is not None else DEFAULT_NUM_FRAMES
fps = args.fps if args.fps is not None else DEFAULT_FPS

output = ov_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps).frames[0]
export_to_video(output, "output.mp4", fps=fps)
print("Video saved to output.mp4")