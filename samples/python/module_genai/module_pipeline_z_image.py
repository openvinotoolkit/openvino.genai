#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
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

class TransformerPipeline():
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "CPU"
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
                            'name': 'prompt_embed',
                            'type': 'OVTensor'
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
                'denoiser_loop': {
                    'type': 'ZImageDenoiserLoopModule',
                    'device': self.device,
                    'description': 'Z-Image denoiser loop.',
                    'inputs': [
                        {
                            'name': 'prompt_embed',
                            'type': 'OVTensor',
                            'source': "pipeline_params.prompt_embed"
                        },
                        {
                            'name': 'num_inference_steps',
                            'type': 'Int',
                            'source': "pipeline_params.num_inference_steps"
                        },
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
                        'model_path': self.model_path
                    }
                },
                'vae': {
                    'type': 'VAEDecoderModule',
                    'device': self.device,
                    'description': 'Z-Image denoiser loop.',
                    'inputs': [
                        {
                            'name': 'latents',
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

        self.pipe = openvino_genai.ModulePipeline(config_yaml_content=yaml.dump(cfg_data))

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = False,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds = self._encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = []
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        device = "cpu"

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        text_encoder_inputs = {
            "input_ids": text_input_ids,
            "attention_mask": prompt_masks,
        }

        ov_outputs = self.text_encoder_request(text_encoder_inputs, share_inputs=True)
        hidden_states = [torch.from_numpy(ov_outputs[out_name]) for out_name in self.text_encoder_hidden_states_output_names]
        prompt_embeds = hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

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

        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        self.pipe.generate(
            prompt_embed=Tensor(prompt_embeds[0].detach().cpu().contiguous().numpy()),
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
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', default="./ut_pipelines/Z-Image-Turbo-fp16-ov/",  help="Path to the directory of models")
    parser.add_argument('prompt', default="", help="The prompt for generation")
    args = parser.parse_args()

    pipeline = TransformerPipeline(args.model_path)
    images = pipeline(
        prompt=args.prompt,
        height=512,
        width=512,
        num_inference_steps=9
    )
    images[0].save("zimage_denoiser_loop_output-vae.png")

if __name__ == "__main__":
    main()
