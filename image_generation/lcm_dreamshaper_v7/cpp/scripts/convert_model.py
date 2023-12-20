import torch
import openvino as ov
import gc
import warnings
from pathlib import Path
from diffusers import DiffusionPipeline
import gc
import warnings
from pathlib import Path
from diffusers import DiffusionPipeline


warnings.filterwarnings("ignore")

TEXT_ENCODER_OV_PATH = Path("./lcm_dreamshaper_v7/FP32/text_encoder/openvino_model.xml")
UNET_OV_PATH = Path("./lcm_dreamshaper_v7/FP32/unet/openvino_model.xml")
VAE_DECODER_OV_PATH = Path("./lcm_dreamshaper_v7/FP32/vae_decoder/openvino_model.xml")
VAE_ENCODER_OV_PATH = Path("./lcm_dreamshaper_v7/FP32/vae_encoder/openvino_model.xml")

def load_orginal_pytorch_pipeline_componets(skip_models=False, skip_safety_checker=True):
    pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    feature_extractor = pipe.feature_extractor if not skip_safety_checker else None
    safety_checker = pipe.safety_checker if not skip_safety_checker else None
    text_encoder, unet, vae = None, None, None
    if not skip_models:
        text_encoder = pipe.text_encoder
        text_encoder.eval()
        unet = pipe.unet
        unet.eval()
        vae = pipe.vae
        vae.eval()
    del pipe
    gc.collect()
    return (
        scheduler,
        tokenizer,
        feature_extractor,
        safety_checker,
        text_encoder,
        unet,
        vae,
    )
skip_conversion = (
    TEXT_ENCODER_OV_PATH.exists()
    and UNET_OV_PATH.exists()
    and VAE_DECODER_OV_PATH.exists()
)

(
    scheduler,
    tokenizer,
    feature_extractor,
    safety_checker,
    text_encoder,
    unet,
    vae,
) = load_orginal_pytorch_pipeline_componets(skip_conversion)

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
    """
    Convert Text Encoder mode.
    Function accepts text encoder model, and prepares example inputs for conversion,
    Parameters:
        text_encoder (torch.nn.Module): text_encoder model from Stable Diffusion pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """
    input_ids = torch.ones((1, 77), dtype=torch.long)
    # switch model to inference mode
    text_encoder.eval()

    # disable gradients calculation for reducing memory consumption
    with torch.no_grad():
        # Export model to IR format
        ov_model = ov.convert_model(
            text_encoder,
            example_input=input_ids,
            input=[
                (-1, 77),
            ],
        )
    ov.save_model(ov_model, ir_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"Text Encoder successfully converted to IR and saved to {ir_path}")


if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder(text_encoder, TEXT_ENCODER_OV_PATH)
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")

del text_encoder
gc.collect()


def convert_unet(unet: torch.nn.Module, ir_path: Path):
    """
    Convert U-net model to IR format.
    Function accepts unet model, prepares example inputs for conversion,
    Parameters:
        unet (StableDiffusionPipeline): unet from Stable Diffusion pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """
    # prepare inputs
    dummy_inputs = {
        "sample": torch.randn((1, 4, 64, 64)),
        "timestep": torch.ones([1]).to(torch.float32),
        "encoder_hidden_states": torch.randn((1, 77, 768)),
        "timestep_cond": torch.randn((1, 256)),
    }
    unet.eval()
    with torch.no_grad():
        ov_model = ov.convert_model(unet, example_input=dummy_inputs)

    ov.save_model(ov_model, ir_path)
    del ov_model
    cleanup_torchscript_cache()
    gc.collect()
    print(f"Unet successfully converted to IR and saved to {ir_path}")


if not UNET_OV_PATH.exists():
    convert_unet(unet, UNET_OV_PATH)
else:
    print(f"Unet will be loaded from {UNET_OV_PATH}")
del unet
gc.collect()


def convert_vae_encoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model for encoding to IR format. 
    Function accepts vae model, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for conversion, 
    Parameters: 
        vae (torch.nn.Module): VAE model from StableDiffusio pipeline 
        ir_path (Path): File for storing model
    Returns:
        None
    """
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            return self.vae.encode(x=image)["latent_dist"].sample()
    vae_encoder = VAEEncoderWrapper(vae)
    vae_encoder.eval()
    image = torch.zeros((1, 3, 512, 512))
    with torch.no_grad():
        ov_model = ov.convert_model(vae_encoder, example_input=image, input=[((1,3,512,512),)])
    ov.save_model(ov_model, ir_path)
    del ov_model
    cleanup_torchscript_cache()
    print(f'VAE encoder successfully converted to IR and saved to {ir_path}')


if not VAE_ENCODER_OV_PATH.exists():
    convert_vae_encoder(vae, VAE_ENCODER_OV_PATH)
else:
    print(f"VAE encoder will be loaded from {VAE_ENCODER_OV_PATH}")


def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model for decoding to IR format. 
    Function accepts vae model, creates wrapper class for export only necessary for inference part, 
    prepares example inputs for conversion, 
    Parameters: 
        vae (torch.nn.Module): VAE model frm StableDiffusion pipeline
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

    vae_decoder = VAEDecoderWrapper(vae)
    latents = torch.zeros((1, 4, 64, 64))

    vae_decoder.eval()
    with torch.no_grad():
        ov_model = ov.convert_model(vae_decoder, example_input=latents, input=[((1,4,64,64),)])
    ov.save_model(ov_model, ir_path)
    del ov_model
    cleanup_torchscript_cache()
    print(f'VAE decoder successfully converted to IR and saved to {ir_path}')


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder(vae, VAE_DECODER_OV_PATH)
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")

del vae
gc.collect()
