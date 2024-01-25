# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys
import gc
import time
import copy
import logging as log
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
from typing import Tuple, Union, Dict, TYPE_CHECKING
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    LDMSuperResolutionPipeline,
)
from diffusers import UNet2DConditionModel, AutoencoderTiny, LCMScheduler
from nncf import compress_weights
from openvino import Type as OVType, PartialShape, save_model, convert_model
from openvino.runtime import Core, get_version
from optimum.exporters import TasksManager
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.exporters.onnx import get_encoder_decoder_models_for_export
from optimum.exporters.openvino import export_models
from optimum.intel.openvino import (
    OVModelForSeq2SeqLM,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLPipeline,
    OVLatentConsistencyModelPipeline,
    OV_XML_FILE_NAME,
    OV_DECODER_NAME,
    OV_DECODER_WITH_PAST_NAME,
    OV_ENCODER_NAME,
)
from optimum.utils.import_utils import is_torch_available, is_diffusers_available

try:
    from optimum.exporters.openvino.__main__ import _get_submodels_and_export_configs
except ImportError:
    from optimum.exporters.onnx.__main__ import (
        _get_submodels_and_onnx_configs as _get_submodels_and_export_configs,
    )

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
from utils.nncf_utils import get_compressed_path
from utils.model_utils import add_stateful_model_arguments
from optimum.exporters.openvino.utils import flattenize_inputs
from utils.conversion_utils.convert_patch import patch_model_for_optimum_export
from utils.conversion_utils.better_transformer_patch import (
    register_bettertransformer_config,
)
from utils.conversion_utils.export_configs import *  # noqa: F401,F403
from utils.ov_model_classes import register_normalized_configs
from utils.conversion_utils.helpers import (
    PYTORCH_DIR,
    OV_DIR,
    GPTQ_DIR,
    PYTORCH_COMPRESS_WEIGHTS_DIR,
    is_torch_compression,
    is_ov_compression,
    is_gptq,
    is_fp16,
    patch_gptq,
    unpatch_gptq,
    save_tokenizer,
    compress_ov_model_weights_helper,
    save_ov_model_helper,
    get_fp_path,
    is_ov_model_provided,
    BackendType,
)

if TYPE_CHECKING:
    from optimum.onnx.configuration import OnnxConfig

    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin

register_normalized_configs()
register_bettertransformer_config()


def convert_optimum_causallm_base(model, args, model_config=None, compress_only=False):
    tokenizer_id = args.tokenizer_id or args.model_id
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    precision = args.precision
    gptq_applied = is_gptq(model_config)
    pt_compress_weights = is_torch_compression(args)
    if not compress_only:
        model_config = model.config
        model = patch_model_for_optimum_export(model)
        precision = precision if not gptq_applied else GPTQ_DIR.format(precision=args.precision)
        ov_out_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / precision
        if gptq_applied and args.compress_weights:
            log.info("Weights compression will be skipped for GPTQ models")
        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / PYTORCH_DIR
            model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)
        dummy_shapes = DEFAULT_DUMMY_SHAPES
        onnx_config, models_and_onnx_configs = _get_submodels_and_export_configs(
            model=model,
            task="text-generation-with-past",
            custom_onnx_configs={},
            custom_architecture=None,
            fn_get_submodels=None,
            preprocessors=None,
            _variant="default",
            monolith=False,
        )
        if "decoder_with_past_model" in models_and_onnx_configs:
            models_and_onnx_configs = {"model": models_and_onnx_configs["decoder_with_past_model"]}
        model.config.save_pretrained(ov_out_dir)
        files_subpaths = ["openvino_" + model_name + ".xml" for model_name in models_and_onnx_configs.keys()]
        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=ov_out_dir,
            output_names=files_subpaths,
            input_shapes=dummy_shapes,
            device="cpu",
            compression_option="fp16" if args.precision == "FP16" else None,
            model_kwargs={},
            stateful=args.stateful,
        )
        save_tokenizer(tok, ov_out_dir)

    if is_ov_compression(args) and not gptq_applied:
        if compress_only:
            fp_path = get_fp_path(args, "openvino_model.xml")
            log.info(
                f"Model conversion to {args.precision} will be skipped as found converted model {fp_path}."
                "If it is not expected behaviour, please remove previously converted model or use --force_convert option"
            )
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            optimized_dir = get_compressed_path(args.output_dir, args.precision, compress_option)
            model_config.save_pretrained(optimized_dir)
            fp_path = get_fp_path(args, "openvino_model.xml")
            ir_model = Core().read_model(fp_path)

            compress_ov_model_weights_helper(
                ir_model,
                tok,
                model_config,
                optimized_dir,
                compress_option,
                is_fp16(args),
                args,
            )

    if pt_compress_weights and not gptq_applied:
        assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
        compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
        compressed_model = compress_weights(model)
        onnx_config, models_and_onnx_configs = _get_submodels_and_export_configs(
            model=compressed_model,
            task="text-generation-with-past",
            custom_onnx_configs={},
            custom_architecture=None,
            fn_get_submodels=None,
            preprocessors=None,
            _variant="default",
            monolith=False,
        )
        pt_out_dir = (
            Path(args.output_dir)
            / PYTORCH_DIR
            / OV_DIR
            / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=precision, compression=compression)
        )
        model.config.save_pretrained(pt_out_dir)
        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=pt_out_dir,
            output_names=files_subpaths,
            input_shapes=dummy_shapes,
            device="cpu",
            compression_option="fp16" if args.precision == "FP16" else None,
            model_kwargs={},
            stateful=args.stateful,
        )
        save_tokenizer(tok, pt_out_dir)
    return


def convert_causal_lm(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    ov_out_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR
    precision = args.precision
    compression_only = is_ov_compression(args) and not is_torch_compression(args) and is_ov_model_provided(args.model_id, ov_out_dir, precision)
    model_kwargs = {}
    if post_init is not None:
        model_kwargs["torch_dtype"] = torch.float32
    model = None
    if not compression_only:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, config=config, **model_kwargs)
        try:
            model.to(torch.float32)
        except Exception:
            pass
    convert_optimum_causallm_base(model, args, config, compression_only)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_seq2seq(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer_id = args.model_id if "blenderbot-9B" not in args.model_id else "facebook/blenderbot-3B"
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    pt_compress_weights = is_torch_compression(args)
    if args.save_orig or pt_compress_weights:
        pt_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            config=config,
        )
        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / PYTORCH_DIR
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)
        if pt_compress_weights:
            assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
            compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
            compressed_pt_model = compress_weights(pt_model)
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=pt_model, exporter="onnx", task="text2text-generation")
            onnx_config = onnx_config_constructor(pt_model.config, use_past=True)
            models_and_onnx_configs = get_encoder_decoder_models_for_export(compressed_pt_model, onnx_config)
            encoder_file_name = Path("encoder") / OV_ENCODER_NAME
            decoder_file_name = Path("decoder") / OV_DECODER_NAME
            decoder_with_past_file_name = Path("decoder_with_past") / OV_DECODER_WITH_PAST_NAME

            output_names = [
                encoder_file_name,
                decoder_file_name,
                decoder_with_past_file_name,
            ]
            save_dir_path = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compression)
            )
            try:
                export_models(
                    models_and_onnx_configs=models_and_onnx_configs,
                    opset=onnx_config.DEFAULT_ONNX_OPSET,
                    output_dir=save_dir_path,
                    output_names=output_names,
                    compression_option="fp16" if args.precision == "FP16" else None,
                )
                save_tokenizer(tok, save_dir_path)
            except Exception as ex:
                log.warning(f"PT weights compression failed with {ex}, please use OpenVINO backend instead")

        del pt_model
        gc.collect()

    ov_compression = is_ov_compression(args)
    ov_encoder = is_ov_model_provided(args.model_id, args.output_dir, args.precision, "openvino_encoder_model.xml")
    ov_decoder = is_ov_model_provided(args.model_id, args.output_dir, args.precision, "openvino_decoder_model.xml")
    compress_only = ov_compression and not args.force_convert and ov_encoder and ov_decoder
    if not compress_only:
        start = time.perf_counter()
        model = OVModelForSeq2SeqLM.from_pretrained(
            args.model_id,
            export=True,
            compile=False,
            trust_remote_code=True,
            config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
        )
        end = time.perf_counter()
        log.info(f"Conversion total time {end - start}s")

        start1 = time.perf_counter()
        ov_out_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
        model.save_pretrained(ov_out_dir)
        end1 = time.perf_counter()
        log.info(f"Serialization total time {end1 - start1}s")
        save_tokenizer(tok, ov_out_dir)
        del model
        gc.collect()

    if ov_compression:
        if compress_only:
            log.info(
                f"Model conversion to {args.precision} will be skipped as found converted model. "
                "If it is not expected behaviour, please remove previously converted model or use --force_convert option"
            )
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            optimized_dir = get_compressed_path(args.output_dir, args.precision, compress_option)
            fp_enc_path = get_fp_path(args, "openvino_encoder_model.xml")
            enc_model = Core().read_model(fp_enc_path)
            compress_ov_model_weights_helper(
                enc_model,
                tok,
                config,
                optimized_dir,
                compress_option,
                is_fp16(args),
                args,
                "openvino_encoder_model",
            )
            fp_dec_path = get_fp_path(args, "openvino_decoder_model.xml")
            dec_model = Core().read_model(fp_dec_path)
            compress_ov_model_weights_helper(
                dec_model,
                tok,
                config,
                optimized_dir,
                compress_option,
                is_fp16(args),
                args,
                "openvino_decoder_model",
            )
            fp_dec_path = get_fp_path(args, "openvino_decoder_with_past_model.xml")
            if fp_dec_path is not None:
                dec_model = Core().read_model(fp_dec_path)
                compress_ov_model_weights_helper(
                    dec_model,
                    tok,
                    config,
                    optimized_dir,
                    compress_option,
                    is_fp16(args),
                    args,
                    "openvino_decoder_with_past_model",
                )


def _get_submodels_for_export_stable_diffusion(
    pipeline: "StableDiffusionPipeline",
) -> Dict[str, Union["PreTrainedModel", "ModelMixin"]]:
    """
    Returns the components of a Stable Diffusion model.
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline
    from diffusers.models.attention_processor import AttnProcessor

    models_for_export = {}
    if isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
        projection_dim = pipeline.text_encoder_2.config.projection_dim
    else:
        projection_dim = pipeline.text_encoder.config.projection_dim

    # Text encoder
    if pipeline.text_encoder is not None:
        if isinstance(pipeline, StableDiffusionXLImg2ImgPipeline):
            pipeline.text_encoder.config.output_hidden_states = True
        models_for_export["text_encoder"] = pipeline.text_encoder

    # U-NET
    pipeline.unet.config.text_encoder_projection_dim = projection_dim
    # The U-NET time_ids inputs shapes depends on the value of `requires_aesthetics_score`
    # https://github.com/huggingface/diffusers/blob/v0.18.2/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L571
    pipeline.unet.config.requires_aesthetics_score = getattr(pipeline.config, "requires_aesthetics_score", False)
    pipeline.unet.set_attn_processor(AttnProcessor())
    models_for_export["unet"] = pipeline.unet

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = copy.deepcopy(pipeline.vae)
    vae_encoder.forward = lambda sample: {"latent_sample": vae_encoder.encode(x=sample)["latent_dist"].sample()}
    models_for_export["vae_encoder"] = vae_encoder

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    vae_decoder.forward = lambda latent_sample: vae_decoder.decode(z=latent_sample)
    models_for_export["vae_decoder"] = vae_decoder

    text_encoder_2 = getattr(pipeline, "text_encoder_2", None)
    if text_encoder_2 is not None:
        text_encoder_2.config.output_hidden_states = True
        models_for_export["text_encoder_2"] = text_encoder_2

    return models_for_export


def get_stable_diffusion_models_for_export(
    pipeline: "StableDiffusionPipeline",
    int_dtype: str = "int64",
    float_dtype: str = "fp32",
) -> Dict[str, Tuple[Union["PreTrainedModel", "ModelMixin"], "OnnxConfig"]]:
    """
    Returns the components of a Stable Diffusion model and their subsequent onnx configs.

    Args:
        pipeline ([`StableDiffusionPipeline`]):
            The model to export.
        int_dtype (`str`, defaults to `"int64"`):
            The data type of integer tensors, could be ["int64", "int32", "int8"], default to "int64".
        float_dtype (`str`, defaults to `"fp32"`):
            The data type of float tensors, could be ["fp32", "fp16", "bf16"], default to "fp32".

    Returns:
        `Dict[str, Tuple[Union[`PreTrainedModel`, `TFPreTrainedModel`], `OnnxConfig`]: A Dict containing the model and
        onnx configs for the different components of the model.
    """
    models_for_export = _get_submodels_for_export_stable_diffusion(pipeline)

    # Text encoder
    if "text_encoder" in models_for_export:
        text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
            model=pipeline.text_encoder,
            exporter="onnx",
            task="feature-extraction",
        )
        text_encoder_onnx_config = text_encoder_config_constructor(pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
        models_for_export["text_encoder"] = (models_for_export["text_encoder"], text_encoder_onnx_config)

    # U-NET
    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.unet,
        exporter="onnx",
        task="semantic-segmentation",
        model_type="unet",
    )
    unet_onnx_config = onnx_config_constructor(pipeline.unet.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["unet"] = (models_for_export["unet"], unet_onnx_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = models_for_export["vae_encoder"]
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter="onnx",
        task="semantic-segmentation",
        model_type="vae-encoder",
    )
    vae_onnx_config = vae_config_constructor(vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["vae_encoder"] = (vae_encoder, vae_onnx_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = models_for_export["vae_decoder"]
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter="onnx",
        task="semantic-segmentation",
        model_type="vae-decoder",
    )
    vae_onnx_config = vae_config_constructor(vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["vae_decoder"] = (vae_decoder, vae_onnx_config)

    if "text_encoder_2" in models_for_export:
        onnx_config_constructor = TasksManager.get_exporter_config_constructor(
            model=pipeline.text_encoder_2,
            exporter="onnx",
            task="feature-extraction",
            model_type="clip-text-with-projection",
        )
        onnx_config = onnx_config_constructor(pipeline.text_encoder_2.config, int_dtype=int_dtype, float_dtype=float_dtype)
        models_for_export["text_encoder_2"] = (models_for_export["text_encoder_2"], onnx_config)

    return models_for_export


def convert_sd_common(pipeline, output_dir, args, tiny_vae=False):
    models_and_onnx_configs = get_stable_diffusion_models_for_export(pipeline)
    if tiny_vae:
        models_and_onnx_configs["vae_encoder"][0].forward = lambda sample: {
            "latent_sample": models_and_onnx_configs["vae_encoder"][0].encode(x=sample)["latents"]
        }
        models_and_onnx_configs["vae_decoder"][0].forward = lambda latent_sample: models_and_onnx_configs["vae_decoder"][0].decode(latent_sample)
    for model_name in models_and_onnx_configs:
        subcomponent = models_and_onnx_configs[model_name][0]
        if hasattr(subcomponent, "save_config"):
            subcomponent.save_config(output_dir / model_name)
        elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
            subcomponent.config.save_pretrained(output_dir / model_name)

            files_subpaths = [Path(name_dir) / OV_XML_FILE_NAME for name_dir in models_and_onnx_configs]

            # Saving the additional components needed to perform inference.
        pipeline.scheduler.save_pretrained(output_dir.joinpath("scheduler"))

        feature_extractor = getattr(pipeline, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output_dir.joinpath("feature_extractor"))

        tokenizer = getattr(pipeline, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir.joinpath("tokenizer"))

        tokenizer_2 = getattr(pipeline, "tokenizer_2", None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output_dir.joinpath("tokenizer_2"))

        pipeline.save_config(output_dir)

    export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        output_dir=output_dir,
        output_names=files_subpaths,
        compression_option="fp16" if args.precision == "FP16" else None
    )


def convert_sd(args):
    pt_compress_weights = is_torch_compression(args)
    pt_model = StableDiffusionPipeline.from_pretrained(args.model_id)
    if args.save_orig:
        pt_model.save_pretrained(Path(args.output_dir) / PYTORCH_DIR)

    output_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
    convert_sd_common(pt_model, output_dir, args)

    if pt_compress_weights:
        assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
        compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
        wc_text_encoder = compress_weights(pt_model.text_encoder)
        wc_unet = compress_weights(pt_model.unet)
        wc_vae = compress_weights(pt_model.vae)
        pt_model.text_encoder = wc_text_encoder
        pt_model.unet = wc_unet
        pt_model.vae = wc_vae
        output = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compression)
        convert_sd_common(pt_model, output, args)
    del pt_model
    gc.collect()

    if is_ov_compression(args):
        for weigths_compression_option in args.compress_weights:
            if weigths_compression_option not in ["INT8", "INT8_ASYM"]:
                log.warning(
                    "Weights compression {weigths_compression_option} does not supported for SD, will be ignored"
                )
                continue
            model = OVStableDiffusionPipeline.from_pretrained(output_dir, compile=False)
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            model.text_encoder.model = compress_weights(model.text_encoder.model)
            model.unet.model = compress_weights(model.unet.model)
            model.vae_decoder.model = compress_weights(model.vae_decoder.model)
            model.save_pretrained(ov_int8_dir)

            del model
            gc.collect()


def convert_lcm(args):
    pt_compress_weights = is_torch_compression(args)
    pt_model = StableDiffusionPipeline.from_pretrained(args.model_id)
    if args.save_orig:
        pt_model.save_pretrained(Path(args.output_dir) / PYTORCH_DIR)

    output_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
    convert_sd_common(pt_model, output_dir, args)

    if pt_compress_weights:
        assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
        compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
        wc_text_encoder = compress_weights(pt_model.text_encoder)
        wc_unet = compress_weights(pt_model.unet)
        wc_vae = compress_weights(pt_model.vae)
        pt_model.text_encoder = wc_text_encoder
        pt_model.unet = wc_unet
        pt_model.vae = wc_vae
        output = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compression)
        convert_sd_common(pt_model, output, args)
    del pt_model
    gc.collect()

    if is_ov_compression(args):
        for weigths_compression_option in args.compress_weights:
            if weigths_compression_option not in ["INT8", "INT8_ASYM"]:
                log.warning(
                    "Weights compression {weigths_compression_option} does not supported for LCM, will be ignored"
                )
                continue
            model = OVLatentConsistencyModelPipeline.from_pretrained(output_dir, compile=False)
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            model.text_encoder.model = compress_weights(model.text_encoder.model)
            model.unet.model = compress_weights(model.unet.model)
            model.vae_decoder.model = compress_weights(model.vae_decoder.model)
            model.save_pretrained(ov_int8_dir)

            del model
            gc.collect()


def convert_sdxl(args):
    pt_compress_weights = is_torch_compression(args)

    def build_pt_model(model_id):
        model_ids = [idx.replace(" ", "") for idx in model_id.split(",")]
        pt_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_ids[0])
        tiny_vae = False
        if len(model_ids) > 1:
            for additional_model in model_ids[1:]:
                if "lora" in additional_model:
                    pt_model.load_lora_weights(additional_model)
                    pt_model.fuse_lora()
                    if "lcm" in additional_model:
                        pt_model.scheduler = LCMScheduler.from_config(pt_model.scheduler.config)
                    continue

                if "lcm" in additional_model and "lora" not in additional_model:
                    unet = UNet2DConditionModel.from_pretrained(additional_model)
                    pt_model.unet = unet
                    pt_model.scheduler = LCMScheduler.from_config(pt_model.scheduler.config)
                    continue

                if "tae" in additional_model:
                    tiny_vae = True
                    vae = AutoencoderTiny.from_pretrained(additional_model)
                    pt_model.vae = vae
                    continue

        return pt_model, tiny_vae

    pt_model, tiny_vae = build_pt_model(args.model_id)
    if args.save_orig:
        pt_model.save_pretrained(Path(args.output_dir) / PYTORCH_DIR)

        del pt_model
        gc.collect()
        pt_model, tiny_vae = build_pt_model(args.model_id)

    fp_out_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
    convert_sd_common(pt_model, fp_out_dir, args, tiny_vae)
    if pt_compress_weights:
        assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
        compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
        output = (
            Path(args.output_dir)
            / PYTORCH_DIR
            / OV_DIR
            / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compression)
        )
        pt_model.text_encoder = compress_weights(pt_model.text_encoder)
        pt_model.unet = compress_weights(pt_model.unet)
        pt_model.vae = compress_weights(pt_model.vae)
        if getattr(pt_model, "text_encoder_2", None) is not None:
            pt_model.text_encoder_2 = compress_weights(pt_model.text_encoder_2)
        convert_sd_common(pt_model, output, args, tiny_vae)

    del pt_model
    gc.collect()

    if is_ov_compression(args):
        for weigths_compression_option in args.compress_weights:
            if weigths_compression_option not in ["INT8", "INT8_ASYM"]:
                log.warning(
                    "Weights compression {weigths_compression_option} does not supported for SDXL, will be ignored"
                )
                continue
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            model = OVStableDiffusionXLPipeline.from_pretrained(fp_out_dir, compile=False)
            model.text_encoder.model = compress_weights(model.text_encoder.model)
            if getattr(model, "text_encoder_2", None) is not None:
                model.text_encoder_2.model = compress_weights(model.text_encoder_2.model)
            model.unet.model = compress_weights(model.unet.model)
            model.vae_decoder.model = compress_weights(model.vae_decoder.model)
            if getattr(model, "vae_encoder", None) is not None:
                model.vae_encoder.model = compress_weights(model.vae_encoder.model)
            model.save_pretrained(ov_int8_dir)

            del model
            gc.collect()


def convert_ldm_super_res(args):
    pipeline = LDMSuperResolutionPipeline.from_pretrained(args.model_id)
    if args.save_orig:
        pipeline.save_pretrained(Path(args.output_dir) / PYTORCH_DIR)
    unet_example_input = [
        torch.zeros((1, 6, 128, 128)),
        torch.tensor(1, dtype=torch.int32),
    ]

    class Decoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, latents):
            return self.model.decode(latents)

    decoder = Decoder(pipeline.vqvae)

    pt_compress_weights = is_torch_compression(args)
    compress_to_fp16 = is_fp16(args)
    if pt_compress_weights:
        assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
        compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
        compressed_unet = compress_weights(pipeline.unet)
        ov_compressed_unet = convert_model(compressed_unet, example_input=unet_example_input)
        ov_compressed_unet.inputs[1].get_node().set_element_type(OVType.i32)
        ov_compressed_unet.inputs[1].get_node().set_partial_shape(PartialShape([]))
        ov_compressed_unet.validate_nodes_and_infer_types()
        pt_out_dir = (
            Path(args.output_dir)
            / PYTORCH_DIR
            / OV_DIR
            / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compression)
        )
        save_model(
            ov_compressed_unet,
            pt_out_dir / "unet.xml",
            compress_to_fp16=compress_to_fp16,
        )
        pipeline.scheduler.save_config(pt_out_dir)
        # Couldn't compress decoder weights (RuntimeError: cdist only supports floating-point dtypes, X2 got: Byte)
        ov_decoder = convert_model(decoder, example_input=torch.zeros((1, 3, 128, 128)))
        save_model(ov_decoder, pt_out_dir / "vqvae.xml", compress_to_fp16=compress_to_fp16)

    # convert model to OpenVINO IR
    ov_unet = convert_model(pipeline.unet, example_input=unet_example_input)
    ov_unet.inputs[1].get_node().set_element_type(OVType.i32)
    ov_unet.inputs[1].get_node().set_partial_shape(PartialShape([]))
    ov_unet.validate_nodes_and_infer_types()
    save_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
    save_model(ov_unet, save_dir / "unet.xml", compress_to_fp16=compress_to_fp16)
    ov_decoder = convert_model(decoder, example_input=torch.zeros((1, 3, 128, 128)))
    save_model(ov_decoder, save_dir / "vqvae.xml", compress_to_fp16=compress_to_fp16)
    pipeline.scheduler.save_config(save_dir)

    if is_ov_compression(args):
        for weigths_compression_option in args.compress_weights:
            if weigths_compression_option not in ["INT8", "INT8_ASYM"]:
                log.warning(
                    "Weights compression {weigths_compression_option} does not supported for LDM, will be ignored"
                )
                continue
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            compressed_ov_unet = compress_weights(ov_unet)
            save_model(
                compressed_ov_unet,
                ov_int8_dir / "unet.xml",
                compress_to_fp16=compress_to_fp16,
            )
            compressed_ov_decoder = compress_weights(ov_decoder)
            save_model(
                compressed_ov_decoder,
                ov_int8_dir / "vqvae.xml",
                compress_to_fp16=compress_to_fp16,
            )
            pipeline.scheduler.save_config(ov_int8_dir)


def convert_mpt(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.use_cache = True
        outs = pt_model(
            input_ids=torch.ones((1, 10), dtype=torch.long),
            attention_mask=torch.ones((1, 10), dtype=torch.long),
        )
        old = outs.past_key_values[0][0].ndim == 3
        inputs = ["input_ids"]
        outputs = ["logits"]

        dynamic_shapes = {"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}}

        for idx in range(len(outs.past_key_values)):
            inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
            dynamic_shapes[inputs[-1]] = {2: "past_sequence + sequence"}
            dynamic_shapes[inputs[-2]] = {3 if not old else 2: "past_sequence + sequence"}
            outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

        inputs.append("attention_mask")
        dummy_inputs = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "past_key_values": outs.past_key_values,
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
        }
        pt_model.config.torchscript = True
        orig_forward = pt_model.forward

        @wraps(orig_forward)
        def ts_patched_forward(
            input_ids: torch.Tensor,
            past_key_values: Tuple[Tuple[torch.Tensor]],
            attention_mask: torch.Tensor,
        ):
            pkv_list = list(past_key_values)
            outs = orig_forward(
                input_ids=input_ids,
                past_key_values=pkv_list,
                attention_mask=attention_mask,
            )
            return (outs.logits, tuple(outs.past_key_values))

        pt_model.forward = ts_patched_forward
        ov_model = convert_model(pt_model, example_input=dummy_inputs)
        pt_model.forward = orig_forward

        for inp_name, m_input, input_data in zip(inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
            input_node = m_input.get_node()
            if input_node.element_type == OVType.dynamic:
                m_input.get_node().set_element_type(OVType.f32)
            shape = list(input_data.shape)
            if inp_name in dynamic_shapes:
                for k in dynamic_shapes[inp_name]:
                    shape[k] = -1
            input_node.set_partial_shape(PartialShape(shape))
            m_input.get_tensor().set_names({inp_name})

        for out, out_name in zip(ov_model.outputs, outputs):
            out.get_tensor().set_names({out_name})

        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    model_kwargs = {}
    precision = args.precision
    compression_only = (
        args.compress_weights
        and not args.force_convert
        and not is_torch_compression(args)
        and is_ov_model_provided(args.model_id, args.output_dir, args.precision)
    )
    gptq_applied = is_gptq(config)
    precision = precision if not gptq_applied else GPTQ_DIR.format(precision=args.precision)
    if post_init is not None:
        model_kwargs = {"torch_dtype": torch.float32}
    pt_model = None
    tokenizer_id = args.tokenizer_id or args.model_id
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    compress_to_fp16 = is_fp16(args)
    if not compression_only:
        pt_model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, config=config, **model_kwargs)
        pt_model.config.use_cache = True
        pt_model.eval()

        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / PYTORCH_DIR
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)

        ov_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / precision
        compress_to_fp16 = is_fp16(args)

        convert_to_ov(pt_model, tok, ov_dir, compress_to_fp16)
        if is_torch_compression(args):
            assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
            compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
            compressed_pt_model = compress_weights(pt_model)
            pt_path = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=precision, compression=compression)
            )
            convert_to_ov(compressed_pt_model, tok, pt_path, compress_to_fp16)

    if is_ov_compression(args):
        ov_path = get_fp_path(args, "openvino_model.xml")
        if compression_only:
            log.info(
                f"Model conversion to {args.precision} will be skipped as found converted model {ov_path}. "
                "If it is not expected behaviour, please remove previously converted model or use --force_convert option"
            )
        ov_model = Core().read_model(ov_path)
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            ov_compressed_path = get_compressed_path(args.output_dir, args.precision, compress_option)
            compress_ov_model_weights_helper(
                ov_model,
                tok,
                config,
                ov_compressed_path,
                compress_option,
                compress_to_fp16,
                args,
            )

    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_chatglm(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.torchscript = True
        last_token = torch.tensor([[130328]])
        past = torch.zeros(28, 2, 5, 1, 32, 128)
        position_ids = torch.tensor([[[2], [4]]])
        dummy_input = {
            "input_ids": last_token,
            "past_key_values": past,
            "position_ids": position_ids,
        }
        ov_model = convert_model(pt_model, example_input=dummy_input)
        ov_model.outputs[0].get_tensor().set_names({"logits"})
        for i in range(1, len(ov_model.outputs), 2):
            idx = (i - 1) // 2
            ov_model.outputs[i].get_tensor().set_names({f"present.{int(idx)}.key"})
            ov_model.outputs[i + 1].get_tensor().set_names({f"present.{int(idx)}.value"})
        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    model_kwargs = {}
    precision = args.precision
    compression_only = (
        args.compress_weights and not args.force_convert and not is_torch_compression(args) and is_ov_model_provided(args.model_id, args.output_dir, precision)
    )
    compress_to_fp16 = is_fp16(args)
    gptq_applied = is_gptq(config)
    precision = precision if not gptq_applied else GPTQ_DIR.format(precision=args.precision)
    tokenizer_id = args.tokenizer_id or args.model_id
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    ov_out_path = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / precision
    if post_init is not None:
        model_kwargs = {"torch_dtype": torch.float32}
    if not compression_only:
        pt_model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True, config=config, **model_kwargs)
        pt_model.config.use_cache = True
        pt_model.to(torch.float32)
        pt_model.eval()

        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / PYTORCH_DIR
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)
        convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16=compress_to_fp16)

        pt_compress_weights = is_torch_compression(args)
        if pt_compress_weights:
            assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
            compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
            compressed_pt_model = compress_weights(pt_model)
            pt_out_path = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=precision, compression=compression)
            )
            convert_to_ov(compressed_pt_model, tok, pt_out_path)

    if is_ov_compression(args):
        ov_model_path = get_fp_path(args, "openvino_model.xml")
        if compression_only:
            log.info(
                f"Model conversion to {args.precision} will be skipped as found converted model {ov_model_path}. "
                "If it is not expected behaviour, please remove previously converted model or use --force_convert option"
            )
        ov_model = Core().read_model(ov_model_path)
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            ov_compressed_path = get_compressed_path(args.output_dir, args.precision, args.compress_weights)
            compress_ov_model_weights_helper(
                ov_model,
                tok,
                config,
                ov_compressed_path,
                compress_to_fp16,
                compress_option,
                args,
            )

    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_falcon(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long))
        inputs = ["input_ids"]
        outputs = ["logits"]

        dynamic_shapes = {"input_ids": {1: "seq_len"}}

        for idx in range(len(outs.past_key_values)):
            inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
            dynamic_shapes[inputs[-1]] = {1: "past_sequence + sequence"}
            dynamic_shapes[inputs[-2]] = {1: "past_sequence + sequence"}
            outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

        dummy_inputs = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "past_key_values": outs.past_key_values,
        }
        flatten_inputs = flattenize_inputs(dummy_inputs.values())
        pt_model.config.torchscript = True
        ov_model = convert_model(pt_model, example_input=dummy_inputs)
        for port, input_data, input_name in zip(ov_model.inputs[1:], flatten_inputs[1:], inputs[1:]):
            port.get_node().set_element_type(OVType.f32)
            shape = list(input_data.shape)
            shape[2] = -1
            port.get_node().set_partial_shape(PartialShape(shape))
            port.get_tensor().set_names({input_name})
        for idx, out_name in enumerate(outputs):
            ov_model.outputs[idx].get_tensor().set_names({out_name})
        ov_model.validate_nodes_and_infer_types()
        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    model_kwargs = {}
    precision = args.precision
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    model_kwargs = {}
    compression_only = (
        args.compress_weights
        and not args.force_convert
        and not is_torch_compression(args)
        and is_ov_model_provided(args.model_id, args.output_dir, args.precision)
    )
    gptq_applied = is_gptq(config)
    if post_init is not None:
        model_kwargs = {"torch_dtype": torch.float32}
    pt_model = None
    tokenizer_id = args.tokenizer_id or args.model_id
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    gptq_applied = is_gptq(config)
    precision = precision if not gptq_applied else GPTQ_DIR.format(precision=args.precision)
    if post_init is not None:
        model_kwargs = {"torch_dtype": torch.float32}
    pt_model = None
    compress_to_fp16 = is_fp16(args)
    ov_out_path = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
    if not compression_only:
        pt_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
            trust_remote_code=True,
            **model_kwargs,
        )
        pt_model.config.use_cache = True
        pt_model.eval()

        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / PYTORCH_DIR
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)

        convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16)

        if is_torch_compression(args):
            assert "INT8" in args.compress_weights or "INT8_ASYM" in args.compress_weights, "Only INT8 compression supported for PyTorch backend"
            compression = "INT8" if "INT8" in args.compress_weights else "INT8_ASYM"
            pt_compressed_model = compress_weights(pt_model)
            pt_comp_path = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compression)
            )
            convert_to_ov(pt_compressed_model, tok, pt_comp_path, compress_to_fp16)

    if is_ov_compression(args):
        fp_path = get_fp_path(args, "openvino_model.xml")
        if compression_only:
            log.info(
                f"Model conversion to {args.precision} will be skipped as found converted model {fp_path}. "
                "If it is not expected behaviour, please remove previously converted model or use --force_convert option"
            )
        ov_model = Core().read_model(fp_path)
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            ov_compressed_path = get_compressed_path(args.output_dir, args.precision, compress_option)
            compress_ov_model_weights_helper(
                ov_model,
                tok,
                pt_model.config,
                ov_compressed_path,
                compress_to_fp16,
                compress_option,
                args,
            )

    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_baichaun(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    model_kwargs = {}
    compression_only = (
        args.compress_weights
        and not args.force_convert
        and not is_torch_compression(args)
        and is_ov_model_provided(args.model_id, args.output_dir, args.precision)
    )
    if post_init is not None:
        model_kwargs = {"torch_dtype": torch.float32}
    model = None
    if not compression_only:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, **model_kwargs)
        try:
            model.to(torch.float32)
            if post_init is None:
                model(torch.ones([1, 10], dtype=torch.long))
        except Exception:
            pass
    try:
        # workaround issue with initialization
        convert_optimum_causallm_base(model, args, config, compression_only)
    except Exception:
        convert_optimum_causallm_base(model, args, config, compression_only)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_qwen(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    model_kwargs = {"revision": "2abd8e5777bb4ce9c8ab4be7dbbd0fe4526db78d"}
    precision = args.precision
    compression_only = (
        args.compress_weights and not args.force_convert and not is_torch_compression(args) and is_ov_model_provided(args.model_id, args.output_dir, precision)
    )
    if post_init is not None:
        model_kwargs = {
            "torch_dtype": torch.float32,
            "revision": "c02ede58c0ab0045f5e4788c35842bec6a7baa0a",
        }
    model = None
    if not compression_only:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, **model_kwargs)
        try:
            model.to(torch.float32)
        except Exception:
            pass

    convert_optimum_causallm_base(model, args, config, compression_only)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_codegen2(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if config.model_type == "codegen":
        config.model_type = "codegen2"
    cuda, post_init = patch_gptq(config)
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    pt_model.config = config
    convert_optimum_causallm_base(pt_model, args, model_config=config)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_aquilachat(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if config.model_type == "llama":
        config.model_type = "aquila"
    cuda, post_init = patch_gptq(config)
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    pt_model.config = config
    convert_optimum_causallm_base(pt_model, args, model_config=config)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


converters = {
    "decoder": convert_causal_lm,
    "blenderbot": convert_seq2seq,
    "t5": convert_seq2seq,
    "stable-diffusion-xl": convert_sdxl,
    "ssd-1b": convert_sdxl,
    "sdxl": convert_sdxl,
    "stable-diffusion": convert_sd,
    "tiny-sd": convert_sd,
    "small-sd": convert_sd,
    "lcm": convert_lcm,
    "ldm": convert_ldm_super_res,
    "mpt": convert_mpt,
    "replit": convert_mpt,
    "chatglm2": convert_causal_lm,
    "chatglm3": convert_causal_lm,
    "chatglm": convert_chatglm,
    "chatglm2": convert_causal_lm,
    "chatglm3": convert_causal_lm,
    "falcon": convert_falcon,
    "baichuan": convert_baichaun,
    "qwen": convert_qwen,
    "codegen2": convert_codegen2,
    "aquilachat": convert_aquilachat,
}


def get_convert_model_type(model_id):
    default = "decoder"
    for key in converters:
        if key in model_id:
            return key

    return default


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_id", required=True, help="model_id or directory for loading")
    parser.add_argument(
        "--tokenizer_id",
        required=False,
        help="tokenizer id or directory for loading. If not provided, model_id will be used by default",
    )
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")
    parser.add_argument("--save_orig", action="store_true", help="save pytorch model on disk")
    parser.add_argument(
        "-p",
        "--precision",
        choices=["FP32", "FP16"],
        default="FP32",
        help="base conversion precision",
    )
    parser.add_argument("--force_convert", action="store_true", help="Force model conversion")

    compression_group = parser.add_argument_group("Weights compression parameters")
    compression_group.add_argument(
        "-c",
        "--compress_weights",
        type=str,
        choices=["INT8", "INT8_ASYM", "4BIT_DEFAULT", "INT4_SYM", "INT4_ASYM"],
        nargs="+",
        help=(
            "The weight compression option, e.g. INT8 - INT8 weights (deprecated, please use INT8_ASYM instead), "
            "4BIT_DEFAULT - for 4-bit compression with predefined configs, "
            "INT4_* - for INT4 compressed weights."
        ),
    )
    compression_group.add_argument(
        "--compress_weights_backends",
        help="Backend names used to compress the input model weights separated by space.",
        choices=[BackendType.PYTORCH.value, BackendType.OPENVINO.value],
        default=BackendType.OPENVINO.value,
        type=str.lower,
        nargs="+",
    )
    compression_group.add_argument(
        "--ratio",
        help="Compression ratio between primary and backup precision, e.g. INT4/INT8",
        default=None,
        type=float,
    )
    compression_group.add_argument(
        "--group_size",
        help="Size of the group of weights that share the same quantization parameters",
        default=None,
        type=int,
    )
    compression_group.add_argument(
        "--all_layers",
        action="store_true",
        help="Compress all layers including embeddings and prediction head",
    )
    add_stateful_model_arguments(parser)

    args = parser.parse_args()
    log.info(f"openvino runtime version: {get_version()}")
    model_type = get_convert_model_type(args.model_id.lower())
    converter = converters[model_type]
    converter(args)


main()
