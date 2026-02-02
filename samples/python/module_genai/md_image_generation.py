#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse

# Parse arguments early to fail fast on invalid arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Z-Image Pipeline Sample")
    parser.add_argument('--model_path', default="../../cpp/module_genai/ut_pipelines/Z-Image-Turbo-fp16-ov",
                        help="Path to the directory of models")
    parser.add_argument('--device', default="GPU", help="Device, default `GPU`, 'CPU' is also supported.")
    parser.add_argument('--enable_tiling', action='store_true', help="Enable tiling. default false.")
    parser.add_argument('--json', default=None, help="Optional: Path to ComfyUI JSON file to convert and use")
    parser.add_argument('--prompt', default=None,
                        help="The prompt for generation (if not provided and --json is used, will be extracted from JSON)")
    parser.add_argument('--height', type=int, default=None, help="Image height (must be divisible by 32, default from JSON or 1040)")
    parser.add_argument('--width', type=int, default=None, help="Image width (must be divisible by 32, default from JSON or 1040)")
    parser.add_argument('--steps', type=int, default=None, help="Number of inference steps (default from JSON or 9)")
    parser.add_argument('--tile_size', type=int, default=None, help="VAE decoder tile size (sample_size for tiling)")
    parser.add_argument('--debug', action='store_true', help="Enable debug output")
    return parser.parse_args()

# Parse args early - will exit on error before heavy imports
args = parse_args()

import time
import numpy as np
from openvino import Core, Tensor
from transformers import AutoTokenizer
from typing import List, Optional, Union
import torch
import openvino_genai
import yaml
import PIL
from PIL import Image
import json
import os


def validate_and_print_config(yaml_content: str, config_name: str = "config") -> bool:
    """
    Validate YAML configuration and print results.

    :param yaml_content: YAML configuration string
    :param config_name: Name for logging purposes
    :return: True if valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Validating {config_name}...")
    print(f"{'='*60}")

    result = openvino_genai.ModulePipeline.validate_config_string(yaml_content)

    print(f"Valid: {result.valid}")

    if result.errors:
        print(f"Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print(f"Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")

    print(f"{'='*60}\n")
    return result.valid


def load_from_comfyui_json(json_path: str, **kwargs) -> tuple:
    """
    Load pipeline configuration from ComfyUI JSON file.

    :param json_path: Path to ComfyUI JSON file
    :param kwargs: Optional parameters (model_path, device, tile_size, etc.)
    :return: Tuple of (yaml_content, extracted_params)
    """
    print(f"\n{'='*60}")
    print(f"Converting ComfyUI JSON: {json_path}")
    print(f"{'='*60}")

    yaml_content, params = openvino_genai.ModulePipeline.comfyui_json_to_yaml(
        json_path,
        **kwargs
    )

    print(f"Conversion successful!")
    print(f"Extracted parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    return yaml_content, params

class TransformerPipeline():
    def __init__(
        self,
        model_path: str,
        device: str,
        enable_tiling:bool,
        tile_size:int=None):
        self.model_path = model_path
        self.device = device
        core = Core()
        vae_decoder_model = core.read_model(self.model_path + "/vae_decoder/openvino_model.xml")
        self.vae_decoder_request = core.compile_model(vae_decoder_model, self.device)
        with open(self.model_path + "/vae_decoder/config.json", "r") as f:
            vae_decoder_config = json.load(f)
            self.vae_scaling_factor = vae_decoder_config.get("scaling_factor", 0.3611)
            self.vae_shift_factor = vae_decoder_config.get("shift_factor", 0.1159)
            self.vae_block_out_channels = vae_decoder_config.get("block_out_channels", [128, 256, 512, 512])

        text_encoder_model = core.read_model(self.model_path + "/text_encoder/openvino_model.xml")
        self.text_encoder_request = core.compile_model(text_encoder_model, self.device)
        self.text_encoder_hidden_states_output_names = [
            name for out in text_encoder_model.outputs for name in out.names if name.startswith("hidden_states")
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path + "/tokenizer", use_fast=False)
        self.vae_scale_factor = (
            2 ** (len(self.vae_block_out_channels) - 1)
        )
        cfg_data = {
            'global_context': {
                'model_type': 'zimage'
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
                            'name': 'guidance_scale',
                            'type': 'Float'
                        },
                        {
                            'name': 'max_sequence_length',
                            'type': 'Int'
                        },
                        {
                            'name': 'num_inference_steps',
                            'type': 'Int'
                        },
                        {
                            'name': 'width',
                            'type': 'Int'
                        },
                        {
                            'name': 'height',
                            'type': 'Int'
                        }
                    ]
                },
                'clip_text_encoder': {
                    'type': 'ClipTextEncoderModule',
                    'device': self.device,
                    'description': 'Encode positive prompt and negative prompt',
                    'inputs': [
                        {
                            'name': 'prompt',
                            'type': 'String',
                            'source': "pipeline_params.prompt"
                        },
                        {
                            'name': 'guidance_scale',
                            'type': 'Float',
                            'source': "pipeline_params.guidance_scale"
                        },
                        {
                            'name': 'max_sequence_length',
                            'type': 'Int',
                            'source': "pipeline_params.max_sequence_length"
                        },
                    ],
                    'outputs': [
                        {
                            'name': 'prompt_embeds',
                            'type': 'VecOVTensor'
                        }
                    ],
                    'params': {
                        'model_path': self.model_path,
                    }
                },
                'latent_image': {
                    'type': 'RandomLatentImageModule',
                    'device': self.device,
                    'description': 'Generate initial latent image.',
                    'inputs': [
                        {
                            'name': 'width',
                            'type': 'Int',
                            'source': "pipeline_params.width"
                        },
                        {
                            'name': 'height',
                            'type': 'Int',
                            'source': "pipeline_params.height"
                        }
                    ],
                    'outputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor'
                        }
                    ],
                    'params': {
                        'model_path': self.model_path,
                    }
                },
                'denoiser_loop': {
                    'type': 'DenoiserLoopModule',
                    'device': self.device,
                    'description': 'Z-Image denoiser loop.',
                    'inputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor',
                            'source': 'latent_image.latents',
                        },
                        {
                            'name': 'prompt_embeds',
                            'type': 'VecOVTensor',
                            'source': "clip_text_encoder.prompt_embeds"
                        },
                        {
                            'name': 'num_inference_steps',
                            'type': 'Int',
                            'source': "pipeline_params.num_inference_steps"
                        }
                    ],
                    'outputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor'
                        }
                    ],
                    'params': {
                        'model_path': self.model_path,
                    }
                },
                'vae': {
                    'type': 'VAEDecoderModule',
                    'device': self.device,
                    'description': 'Z-Image denoiser loop.',
                    'inputs': [
                        {
                            'name': 'latent' if enable_tiling else 'latents',
                            'type': 'OVTensor',
                            'source': 'denoiser_loop.latents',
                        }
                    ],
                    'outputs': [
                        {
                            'name': 'image',
                            'type': 'OVTensor'
                        }
                    ],
                    'params': {
                        'model_path': self.model_path
                    }
                },
                'pipeline_result': {
                    'type': 'ResultModule',
                    'inputs': [
                        {
                            'name': 'image',
                            'type': 'OVTensor',
                            'source': 'vae.image'
                        }
                    ]
                }
            }
        }

        if enable_tiling:
            print(f"enable_tiling = {enable_tiling}")
            if tile_size is not None:
                print(f"tile_size = {tile_size}")
            cfg_data = self._update_config_with_tiling(cfg_data, decoder_module_name="vae", tile_size=tile_size)

        # Convert to YAML string
        yaml_content = yaml.dump(cfg_data)

        # Validate configuration before creating pipeline
        if not validate_and_print_config(yaml_content, "Z-Image Pipeline Config"):
            print("Warning: Configuration validation failed, proceeding anyway...")

        print("[DEBUG] Creating ModulePipeline with YAML content...")
        print(f"[DEBUG] YAML content length: {len(yaml_content)} chars")

        # Try creating pipeline with explicit error handling
        try:
            print("[DEBUG] Calling ModulePipeline constructor...")
            sys.stdout.flush()  # Force flush before potential crash
            self.pipe = openvino_genai.ModulePipeline(config_yaml_content=yaml_content)
            print("[DEBUG] ModulePipeline constructor completed successfully")
        except Exception as e:
            print(f"[ERROR] ModulePipeline constructor failed: {e}")
            raise

    def _update_config_with_tiling(self, cfg_data:dict, decoder_module_name:str, tile_size:int=None):
        decoder_node = cfg_data["pipeline_modules"][decoder_module_name]

        vae_decoder_tiling = {
            'vae_decoder_tiling': {
                'type': "VAEDecoderTilingModule",
                'device': self.device,
                'inputs': [],
                'outputs': [],
                'params': {
                    'tile_overlap_factor': "0.25",
                    'model_path': self.model_path,
                    'sub_module_name': "vae_decoder_submodule"
                }
            }
        }
        # Map tile_size to sample_size
        if tile_size is not None:
            vae_decoder_tiling['vae_decoder_tiling']['params']['sample_size'] = str(tile_size)
        vae_decoder_tiling['vae_decoder_tiling']['inputs'] = decoder_node['inputs']
        vae_decoder_tiling['vae_decoder_tiling']['outputs'] = decoder_node['outputs']
        cfg_data["pipeline_modules"][decoder_module_name] = vae_decoder_tiling['vae_decoder_tiling']

        sub_modules = {
            'sub_modules': [
                {
                    'name': "vae_decoder_submodule",
                    'vae_decoder': {
                        'type' : "VAEDecoderModule",
                        'device': self.device,
                        'inputs': [
                            {
                               'name': 'latents',
                                'type': 'OVTensor',
                            }
                        ],
                        'outputs': [
                            {
                               'name': 'image',
                                'type': 'OVTensor',
                            }
                        ],
                        'params': {
                            'model_path': self.model_path,
                            'enable_postprocess': 'false',
                        }
                    }
                }
            ]
        }

        cfg_data["sub_modules"] = sub_modules['sub_modules']
        return cfg_data

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ):
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        self.pipe.generate(
            prompt=prompt,
            guidance_scale=0.0,
            max_sequence_length=max_sequence_length,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height
        )

        output = self.pipe.get_output("image")
        latents = torch.from_numpy(output.data).to(torch.uint8)

        image = latents.cpu().numpy()
        images = self.numpy_to_pil(image)

        return images

    def numpy_to_pil(self, images: np.ndarray) -> List[PIL.Image.Image]:
        if images.ndim == 3:
            images = images[None, ...]

        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image.astype("uint8")) for image in images]

        return pil_images

def main():
    # Use global args parsed at module load time
    global args

    # Default values when ComfyUI JSON is not provided
    DEFAULT_PROMPT = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp, bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda, blurred colorful distant lights."
    DEFAULT_HEIGHT = 1040
    DEFAULT_WIDTH = 1040
    DEFAULT_STEPS = 9

    if args.debug:
        print("\n[DEBUG] Environment Info:")
        print(f"  Python: {sys.version}")
        import openvino
        print(f"  OpenVINO: {openvino.__version__}")
        print(f"  openvino_genai location: {openvino_genai.__file__}")
        print(f"  Model path (absolute): {os.path.abspath(args.model_path)}")
        print()

    # Variables to hold final values (may come from JSON or args)
    final_prompt = args.prompt
    final_height = args.height
    final_width = args.width
    final_steps = args.steps
    comfyui_yaml_content = None  # Will hold converted YAML if --json is provided
    comfyui_params = {}  # Will hold extracted params if --json is provided

    # Demo: Using ComfyUI JSON conversion if provided
    if args.json and os.path.exists(args.json):
        print("\n" + "="*60)
        print("Demo: ComfyUI JSON to YAML Conversion")
        print("="*60)

        # Build conversion kwargs, include tile_size if provided via command line
        conversion_kwargs = {
            "model_path": args.model_path,
            "device": args.device
        }
        if args.tile_size is not None:
            conversion_kwargs["tile_size"] = args.tile_size

        yaml_content, params = load_from_comfyui_json(
            args.json,
            **conversion_kwargs
        )

        if yaml_content:
            # Validate the converted YAML
            validate_and_print_config(yaml_content, "Converted ComfyUI Config")

            # Dump the converted YAML content
            print(f"\n{'='*60}")
            print("Converted YAML content:")
            print(f"{'='*60}")
            print(yaml_content)
            print(f"{'='*60}\n")

            # Extract parameters from JSON if not provided via command line
            if final_prompt is None:
                final_prompt = params.get("prompt", DEFAULT_PROMPT)
            if final_height is None:
                final_height = params.get("height", DEFAULT_HEIGHT)
            if final_width is None:
                final_width = params.get("width", DEFAULT_WIDTH)
            if final_steps is None:
                final_steps = params.get("num_inference_steps", DEFAULT_STEPS)

            # tile_size: command line overrides JSON value
            final_tile_size = args.tile_size if args.tile_size is not None else params.get("tile_size")

            print(f"\nUsing parameters from ComfyUI JSON:")
            print(f"  prompt: {final_prompt[:50]}..." if len(final_prompt) > 50 else f"  prompt: {final_prompt}")
            negative_prompt = params.get("negative_prompt", "")
            print(f"  negative_prompt: {negative_prompt[:50]}..." if len(negative_prompt) > 50 else f"  negative_prompt: {negative_prompt}")
            print(f"  width: {final_width}")
            print(f"  height: {final_height}")
            print(f"  steps: {final_steps}")
            print(f"  seed: {params.get('seed', 0)}")
            print(f"  tile_size: {final_tile_size}")

            # Save yaml_content and params to use for pipeline creation
            comfyui_yaml_content = yaml_content
            comfyui_params = params

            print("\nComfyUI JSON successfully converted to YAML!")
        else:
            print("ComfyUI JSON conversion returned empty YAML")

    # Apply defaults if still None (no ComfyUI JSON or params not found)
    if final_prompt is None:
        final_prompt = DEFAULT_PROMPT
    if final_height is None:
        final_height = DEFAULT_HEIGHT
    if final_width is None:
        final_width = DEFAULT_WIDTH
    if final_steps is None:
        final_steps = DEFAULT_STEPS

    try:
        if comfyui_yaml_content:
            # Use the converted YAML from ComfyUI JSON
            print("[INFO] Creating ModulePipeline from ComfyUI YAML...")
            pipe = openvino_genai.ModulePipeline(config_yaml_content=comfyui_yaml_content)
            print("[INFO] ModulePipeline created successfully")

            print("[INFO] Starting generation...")
            pipe.generate(
                prompt=final_prompt,
                negative_prompt=comfyui_params.get("negative_prompt", ""),
                guidance_scale=comfyui_params.get("guidance_scale", 1.0),
                max_sequence_length=comfyui_params.get("max_sequence_length", 512),
                num_inference_steps=final_steps,
                width=final_width,
                height=final_height,
                batch_size=comfyui_params.get("batch_size", 1),
                seed=comfyui_params.get("seed", 0)
            )
            print("[INFO] Generation completed")

            # Get output - check for saved_image first (SaveImageModule), then image
            try:
                output_path = pipe.get_output("saved_image")
                print(f"Image saved by pipeline to: {output_path}")
            except:
                output = pipe.get_output("image")
                latents = torch.from_numpy(output.data).to(torch.uint8)
                image = latents.cpu().numpy()
                if image.ndim == 3:
                    image = image[None, ...]
                pil_image = Image.fromarray(image[0].astype("uint8"))
                out_name = "output_zimage_comfyui.png"
                pil_image.save(out_name)
                print(f"Image saved to: {out_name}")
        else:
            # Create pipeline using standard TransformerPipeline config
            print("[INFO] Creating TransformerPipeline...")
            pipeline = TransformerPipeline(
                args.model_path,
                args.device,
                args.enable_tiling,
                args.tile_size)
            print("[INFO] Pipeline created successfully")

            print("[INFO] Starting generation...")
            images = pipeline(
                prompt=final_prompt,
                height=final_height,
                width=final_width,
                num_inference_steps=final_steps
            )
            print("[INFO] Generation completed")

            out_name = "output_zimage_tiling.png" if args.enable_tiling else "output_zimage.png"
            images[0].save(out_name)
            print(f"Image saved to: {out_name}")

    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
