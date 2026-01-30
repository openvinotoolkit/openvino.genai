from pathlib import Path

import sys
import argparse
from unittest.mock import DEFAULT

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
from diffusers import AutoencoderKLWan, WanPipeline, DiffusionPipeline, UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer
from diffusers.utils import export_to_video

import openvino as ov
from openvino import Tensor

from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from dataclasses import dataclass
from typing import Optional, Union
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
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


class OVWanPipeline(DiffusionPipeline):
    def __init__(self, model_dir, device_map="CPU", ov_config=None):
        model_dir = Path(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        scheduler = UniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")
        if isinstance(device_map, str):
            device_map = {"transformer": device_map, "text_encoder": device_map, "vae": device_map}
        transformer_model = core.read_model(model_dir / TRANSFORMER_PATH)
        transformer = core.compile_model(transformer_model, device_map["transformer"], ov_config)
        text_encoder_model = core.read_model(model_dir / TEXT_ENCODER_PATH)
        text_encoder = core.compile_model(text_encoder_model, device_map["text_encoder"], ov_config)
        vae = core.compile_model(model_dir / VAE_DECODER_PATH, device_map["vae"], ov_config)
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 4
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
                            'name': 'latents',
                            'type': 'OVTensor'
                        },
                        {
                            'name': 'prompt_embed',
                            'type': 'OVTensor'
                        },
                        {
                            'name': 'prompt_embed_negative',
                            'type': 'OVTensor'
                        },
                        {
                            'name': 'guidance_scale',
                            'type': 'Float'
                        },
                        {
                            'name': 'num_inference_steps',
                            'type': 'Int'
                        }
                    ]
                },
                'denoiser_loop': {
                    'type': 'DenoiserLoopModule',
                    'device': device_map['transformer'],
                    'inputs': [
                        {
                            'name': 'latents',
                            'type': 'OVTensor',
                            'source': 'pipeline_params.latents'
                        },
                        {
                            'name': 'prompt_embed',
                            'type': 'OVTensor',
                            'source': 'pipeline_params.prompt_embed'
                        },
                        {
                            'name': 'prompt_embed_negative',
                            'type': 'OVTensor',
                            'source': 'pipeline_params.prompt_embed_negative'
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

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, list[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = torch.from_numpy(self.text_encoder(text_input_ids)[0])
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, list[str]],
        negative_prompt: Optional[Union[str, list[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to" " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

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
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = 16
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            generator,
            latents,
        )

        self.pipe.generate(
            latents=Tensor(latents.detach().cpu().contiguous().numpy()),
            prompt_embed=Tensor(prompt_embeds.squeeze(0).detach().cpu().contiguous().numpy()),
            prompt_embed_negative=Tensor(negative_prompt_embeds.squeeze(0).detach().cpu().contiguous().numpy()),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
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
DEFAULT_HEIGHT = 240
DEFAULT_WIDTH = 240
DEFAULT_NUM_FRAMES = 20
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_FPS = 10

prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
negative_prompt = args.negative_prompt if args.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

output = ov_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=args.height if args.height is not None else DEFAULT_HEIGHT,
    width=args.width if args.width is not None else DEFAULT_WIDTH,
    num_frames=args.num_frames if args.num_frames is not None else DEFAULT_NUM_FRAMES,
    guidance_scale=args.guidance_scale if args.guidance_scale is not None else DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=args.steps if args.steps is not None else DEFAULT_NUM_INFERENCE_STEPS,
    generator=torch.Generator("cpu").manual_seed(42)).frames[0]
export_to_video(output, "output.mp4", fps=DEFAULT_FPS)
print("Video saved to output.mp4")