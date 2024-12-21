import os
# set proxy if needed
# os.environ['http_proxy'] = "http://127.0.0.1:20171" 
# os.environ['https_proxy'] = "http://127.0.0.1:20171" 
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderTiny
from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np
from pathlib import Path
import openvino as ov
from collections import namedtuple
import gc
from functools import partial
from typing import Tuple



def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()




def convert():
    pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)
    tae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32)
    
    
    # convert openpose body estimator(detector)
    OPENPOSE_OV_PATH = Path("models/openpose.xml")
    if not OPENPOSE_OV_PATH.exists():
        with torch.no_grad():
            ov_model = ov.convert_model(
                pose_estimator.body_estimation.model,
                example_input=torch.zeros([1, 3, 184, 136]),
                input=[[1, 3, 184, 136]],
            )
            ov.save_model(ov_model, OPENPOSE_OV_PATH)
            del ov_model
            cleanup_torchscript_cache()
        print("OpenPose successfully converted to IR")
    else:
        print(f"OpenPose will be loaded from {OPENPOSE_OV_PATH}")

    # convert controlnet
    CONTROLNET_OV_PATH = Path("models/controlnet-pose.xml")
    inputs = {
        "sample": torch.randn((2, 4, 64, 64)),
        "timestep": torch.tensor([1]),
        "encoder_hidden_states": torch.randn((2, 77, 768)),
        "controlnet_cond": torch.randn((2, 3, 512, 512)),
    }
    input_info = [(name, ov.PartialShape(inp.shape)) for name, inp in inputs.items()]

    controlnet.eval()
    with torch.no_grad():
        down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)

    if not CONTROLNET_OV_PATH.exists():
        with torch.no_grad():
            controlnet.forward = partial(controlnet.forward, return_dict=False)
            ov_model = ov.convert_model(controlnet, example_input=inputs, input=input_info)
            ov.save_model(ov_model, CONTROLNET_OV_PATH)
            del ov_model
            cleanup_torchscript_cache()
        print("ControlNet successfully converted to IR")
    else:
        print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")

    del controlnet
    gc.collect()

    # convert unet
    UNET_OV_PATH = Path("models/unet_controlnet.xml")

    dtype_mapping = {
        torch.float32: ov.Type.f32,
        torch.float64: ov.Type.f64,
        torch.int32: ov.Type.i32,
        torch.int64: ov.Type.i64,
    }

    class UnetWrapper(torch.nn.Module):
        def __init__(
            self,
            unet,
            sample_dtype=torch.float32,
            timestep_dtype=torch.int64,
            encoder_hidden_states=torch.float32,
            down_block_additional_residuals=torch.float32,
            mid_block_additional_residual=torch.float32,
        ):
            super().__init__()
            self.unet = unet
            self.sample_dtype = sample_dtype
            self.timestep_dtype = timestep_dtype
            self.encoder_hidden_states_dtype = encoder_hidden_states
            self.down_block_additional_residuals_dtype = down_block_additional_residuals
            self.mid_block_additional_residual_dtype = mid_block_additional_residual

        def forward(
            self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            down_block_additional_residuals: Tuple[torch.Tensor],
            mid_block_additional_residual: torch.Tensor,
        ):
            sample.to(self.sample_dtype)
            timestep.to(self.timestep_dtype)
            encoder_hidden_states.to(self.encoder_hidden_states_dtype)
            down_block_additional_residuals = [res.to(self.down_block_additional_residuals_dtype) for res in down_block_additional_residuals]
            mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
            )


    def flattenize_inputs(inputs):
        flatten_inputs = []
        for input_data in inputs:
            if input_data is None:
                continue
            if isinstance(input_data, (list, tuple)):
                flatten_inputs.extend(flattenize_inputs(input_data))
            else:
                flatten_inputs.append(input_data)
        return flatten_inputs


    if not UNET_OV_PATH.exists():
        inputs.pop("controlnet_cond", None)
        inputs["down_block_additional_residuals"] = down_block_res_samples
        inputs["mid_block_additional_residual"] = mid_block_res_sample

        unet = UnetWrapper(pipe.unet)
        unet.eval()

        with torch.no_grad():
            ov_model = ov.convert_model(unet, example_input=inputs)

        flatten_inputs = flattenize_inputs(inputs.values())
        for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):
            input_tensor.get_node().set_partial_shape(ov.PartialShape(input_data.shape))
            input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])
        ov_model.validate_nodes_and_infer_types()
        ov.save_model(ov_model, UNET_OV_PATH)
        del ov_model
        cleanup_torchscript_cache()
        del unet
        del pipe.unet
        gc.collect()
        print("Unet successfully converted to IR")
    else:
        del pipe.unet
        print(f"Unet will be loaded from {UNET_OV_PATH}")
    gc.collect()

    # convert text encoder
    TEXT_ENCODER_OV_PATH = Path("models/text_encoder.xml")
    def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
        """
        Convert Text Encoder model to OpenVINO IR.
        Function accepts text encoder model, prepares example inputs for conversion, and convert it to OpenVINO Model
        Parameters:
            text_encoder (torch.nn.Module): text_encoder model
            ir_path (Path): File for storing model
        Returns:
            None
        """
        if not ir_path.exists():
            input_ids = torch.ones((1, 77), dtype=torch.long)
            # switch model to inference mode
            text_encoder.eval()

            # disable gradients calculation for reducing memory consumption
            with torch.no_grad():
                ov_model = ov.convert_model(
                    text_encoder,  # model instance
                    example_input=input_ids,  # inputs for model tracing
                    input=([1, 77],),
                )
                ov.save_model(ov_model, ir_path)
                del ov_model
            cleanup_torchscript_cache()
            print("Text Encoder successfully converted to IR")


    if not TEXT_ENCODER_OV_PATH.exists():
        convert_encoder(pipe.text_encoder, TEXT_ENCODER_OV_PATH)
    else:
        print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")
    del pipe.text_encoder
    gc.collect()

    # convert vae
    VAE_DECODER_OV_PATH = Path("models/vae_decoder.xml")
    TAE_DECODER_OV_PATH = Path("models/tae_decoder.xml")

    def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
        """
        Convert VAE model to IR format.
        Function accepts pipeline, creates wrapper class for export only necessary for inference part,
        prepares example inputs for convert,
        Parameters:
            vae (torch.nn.Module): VAE model
            ir_path (Path): File for storing model
        Returns:
            None
        """

        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, latents):
                return self.vae.decode(latents)

        if not ir_path.exists():
            vae_decoder = VAEDecoderWrapper(vae)
            latents = torch.zeros((1, 4, 64, 64))

            vae_decoder.eval()
            with torch.no_grad():
                ov_model = ov.convert_model(
                    vae_decoder,
                    example_input=latents,
                    input=[
                        (1, 4, 64, 64),
                    ],
                )
                ov.save_model(ov_model, ir_path)
            del ov_model
            cleanup_torchscript_cache()
            print("VAE decoder successfully converted to IR")


    if not VAE_DECODER_OV_PATH.exists():
        convert_vae_decoder(pipe.vae, VAE_DECODER_OV_PATH)
    else:
        print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")

    if not TAE_DECODER_OV_PATH.exists():
        convert_vae_decoder(tae, TAE_DECODER_OV_PATH)
    else:
        print(f"TAE decoder will be loaded from {TAE_DECODER_OV_PATH}")

if __name__ == "__main__":
    convert()