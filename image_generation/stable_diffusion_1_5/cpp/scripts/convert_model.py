from pathlib import Path
import argparse
from optimum.intel.openvino import OVStableDiffusionPipeline
from openvino import Type, save_model
from transformers import AutoTokenizer
from openvino_tokenizers import convert_tokenizer
import torch


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-b', '--batch', type = int, default = 1, required = True,
                      help='Required. batch_size for solving single/multiple prompt->image generation.')
    args.add_argument('-t', '--type', type = str, default = "FP32", required = True,
                      help='Required. data type, FP32, FP16, and compressed type INT8.')
    args.add_argument('-dyn', '--dynamic', type = bool, default = False, required = False,
                      help='Specify the model input shape to use dynamic shape.')
    args.add_argument('-sd','--sd_weights', type = str, default="", required = True,
                      help='Specify the path of stable diffusion model')
    # fmt: on
    return parser.parse_args()

args = parse_args()

output_path = Path(args.sd_weights) / (args.type + ("_dyn" if args.dynamic else "_static"))

# convert SD model to IR

if args.type == "INT8":
    from diffusers import StableDiffusionPipeline
    from optimum.exporters.openvino import export_models
    from optimum.exporters.onnx import __main__ as optimum_main
    from optimum.intel.openvino import OV_XML_FILE_NAME
    from nncf import compress_weights

    pt_model = StableDiffusionPipeline.from_pretrained(args.sd_weights, trust_remote_code=True)

    pt_model.text_encoder = compress_weights(pt_model.text_encoder)
    pt_model.unet = compress_weights(pt_model.unet)
    pt_model.vae = compress_weights(pt_model.vae)

    _, models_and_onnx_configs = optimum_main._get_submodels_and_onnx_configs(
        model=pt_model,
        task="stable-diffusion",
        monolith=False,
        custom_onnx_configs={},
        custom_architecture=False,
        _variant="default"
    )

    for model_name in models_and_onnx_configs:
        subcomponent = models_and_onnx_configs[model_name][0]
        if hasattr(subcomponent, "save_config"):
            subcomponent.save_config(output_path / model_name)
        elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
            subcomponent.config.save_pretrained(output_path / model_name)
    files_subpaths = [Path(name_dir) / OV_XML_FILE_NAME for name_dir in models_and_onnx_configs]

    # Saving the additional components needed to perform inference.
    pt_model.scheduler.save_pretrained(output_path / "scheduler")

    feature_extractor = getattr(pt_model, "feature_extractor", None)
    if feature_extractor is not None:
        feature_extractor.save_pretrained(output_path / "feature_extractor")

    tokenizer = getattr(pt_model, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path / "tokenizer")

    tokenizer_2 = getattr(pt_model, "tokenizer_2", None)
    if tokenizer_2 is not None:
        tokenizer_2.save_pretrained(output_path / "tokenizer_2")

    pt_model.save_config(output_path)

    export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        output_dir=output_path,
        output_names=files_subpaths
    )
    del pt_model

    if not args.dynamic:
        model = OVStableDiffusionPipeline.from_pretrained(output_path, export=False, compile=False)
        model.reshape(args.batch, 512, 512, 1)
        model.save_pretrained(output_path)

else:
    model = OVStableDiffusionPipeline.from_pretrained(args.sd_weights, trust_remote_code=True, export=True, compile=False)
    if args.type == "FP16":
        model.half()
    if not args.dynamic:
        model.reshape(args.batch, 512, 512, 1)

    model.save_pretrained(output_path)

# convert tokenizer

tokenizer_path = output_path / "tokenizer"
hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
ov_tokenizer_encoder = convert_tokenizer(hf_tokenizer, tokenizer_output_type=Type.i32)

save_model(ov_tokenizer_encoder, tokenizer_path / "openvino_tokenizer.xml", compress_to_fp16=False)
