from pathlib import Path

import argparse

"""
Wan 2.1 Video Generation Pipeline Sample

This sample demonstrates how to use the OpenVINO GenAI ModulePipeline
for Wan 2.1 text-to-video generation with optional VAE tiling.

Usage Examples:
    # Basic usage with default settings (tiling enabled, 256x256 tiles)
    python md_video_generation.py --model_path ./models/Wan2.1-T2V-1.3B

    # Run on GPU with custom resolution
    python md_video_generation.py --model_path ./models/Wan2.1-T2V-1.3B --device GPU --width 832 --height 480

    # Disable tiling (for smaller resolutions or comparison)
    python md_video_generation.py --model_path ./models/Wan2.1-T2V-1.3B --use_tiling 0

    # Custom tile size (512x512 tiles with 384 stride, 25% overlap)
    python md_video_generation.py --model_path ./models/Wan2.1-T2V-1.3B --tile_size 512

    # Full customization
    python md_video_generation.py \\
        --model_path ./models/Wan2.1-T2V-1.3B \\
        --device GPU \\
        --prompt "A cat walking on the beach at sunset" \\
        --width 832 --height 480 \\
        --num_frames 33 --steps 30 \\
        --use_tiling 1 --tile_size 256
"""

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
    parser.add_argument('--use_tiling', type=int, default=1, choices=[0, 1],
                        help="VAE tiling mode: 0=disable, 1=enable (default)")
    parser.add_argument('--tile_size', type=int, default=256,
                        help="VAE decoder tile size in pixels (default: 256)")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    parser.add_argument('--splitted_model', action='store_true', help="Enable splitted model inference")
    return parser.parse_args()

# Parse args early - will exit on error before heavy imports
args = parse_args()

import torch
import openvino as ov
from typing import Optional, Union
import openvino_genai
import yaml

core = ov.Core()
class OVWanPipeline:
    def __init__(self, model_dir, device_map="CPU", fps: int = 10, splitted_model: bool = False,use_tiling: bool = True, tile_size: int = 256):
        model_dir = Path(model_dir)
        if isinstance(device_map, str):
            device_map = {"transformer": device_map, "text_encoder": device_map, "vae": device_map}
        self.fps = fps
        self.use_tiling = use_tiling
        self.tile_size = tile_size

        # Calculate stride (75% of tile_size, 25% overlap)
        tile_stride = int(tile_size * 0.75)
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
                        'cache_dir': "./cache_dir_text_encoder/"  # [Optional], default is empty string.
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
                        'cache_dir': "./cache_dir_denoiser_loop/",  # [Optional], default is empty string.
                        'splitted_model': 'true' if splitted_model else 'false'
                    }
                },
                'vae_decoder': {
                    'type': 'VAEDecoderModule',
                    'device': device_map['vae'],
                    'inputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor',
                            'source': 'denoiser_loop.latents'
                        }
                    ],
                    'outputs': [
                        {
                            'name': 'image',
                            'type': 'OVTensor'
                        }
                    ],
                    'params': {
                        'model_path': str(model_dir),
                        'enable_postprocess': 'true',
                        'enable_tiling': 'true' if self.use_tiling else 'false',
                        'tile_sample_min_height': str(self.tile_size),
                        'tile_sample_min_width': str(self.tile_size),
                        'tile_sample_stride_height': str(tile_stride),
                        'tile_sample_stride_width': str(tile_stride),
                    }
                },
                'video_saver': {
                    'type': 'SaveVideoModule',
                    'inputs': [
                        {
                            'name': 'raw_data',
                            'type': 'OVTensor',
                            'source': 'vae_decoder.image'
                        }
                    ],
                    'outputs': [
                        {
                            'name': 'saved_video',
                            'type': 'String'
                        },
                        {
                            'name': 'saved_videos',
                            'type': 'VecString'
                        }
                    ],
                    'params': {
                        'fps': self.fps
                    }
                },
                'pipeline_result': {
                    'type': 'ResultModule',
                    'inputs': [
                        {
                            'name': 'saved_video',
                            'type': 'String',
                            'source': 'video_saver.saved_video'
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

        saved_video = self.pipe.get_output("saved_video")

        return saved_video

device_map = {
    "text_encoder": "CPU",
    "transformer": args.device,
    "vae": args.device,
}

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
splitted_model = args.splitted_model if args.splitted_model is not None else False

use_tiling = args.use_tiling == 1
tile_size = args.tile_size

ov_pipe = OVWanPipeline(args.model_path, device_map=device_map, fps=fps, splitted_model=splitted_model, use_tiling=use_tiling, tile_size=tile_size)

output = ov_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps)
print(f"Video saved to {output}")