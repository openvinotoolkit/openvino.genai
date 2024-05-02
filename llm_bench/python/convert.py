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
from typing import Tuple, Union, Dict, Optional, TYPE_CHECKING
import nncf
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    LDMSuperResolutionPipeline,
)
from diffusers import UNet2DConditionModel, AutoencoderTiny, LCMScheduler
from nncf.torch.model_creation import is_wrapped_model
from openvino import Type as OVType, PartialShape, save_model, convert_model
from openvino.runtime import Core, get_version
from optimum.exporters import TasksManager
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.intel.openvino.configuration import OVConfig
from optimum.exporters.utils import get_encoder_decoder_models_for_export
from optimum.exporters.openvino import export_models
from optimum.exporters.openvino.model_patcher import patch_model_with_bettertransformer
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

from optimum.exporters.utils import _get_submodels_and_export_configs

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
import utils.conversion_utils.export_configs  # noqa: F401,F403
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
    is_int8_compression,
    BackendType,
)
from utils.nncf_utils import COMPRESSION_OPTIONS

if TYPE_CHECKING:
    from optimum.onnx.configuration import OnnxConfig

    if is_torch_available():
        from transformers.modeling_utils import PreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin

register_bettertransformer_config()


def compress_torchmodels(
    models_and_export_configs,
    stateful: bool = True,
    dummy_shapes: Optional[Dict] = None,
    compression_options: Optional[Dict] = None,
):
    if dummy_shapes is None:
        dummy_shapes = {}

    if compression_options is None:
        compression_options = {}

    for model_name in models_and_export_configs.keys():
        submodel, sub_export_config = models_and_export_configs[model_name]
        if stateful:
            submodel = patch_model_with_bettertransformer(submodel)
        if is_wrapped_model(submodel):
            dataset = None
        else:
            dummy_inputs = sub_export_config.generate_dummy_inputs(framework="pt", **dummy_shapes)
            dataset = nncf.Dataset([dummy_inputs])
        compressed_submodel = nncf.compress_weights(submodel, dataset=dataset, **compression_options)
        models_and_export_configs[model_name] = (compressed_submodel, sub_export_config)
    return models_and_export_configs


def convert_optimum_causallm_base(model, args, model_config=None, compress_only=False):
    tokenizer_id = args.tokenizer_id or args.model_id
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    precision = args.precision
    gptq_applied = is_gptq(model_config)
    pt_compress_weights = is_torch_compression(args)
    if args.stateful:
        log.warning(
            "usage --stateful flag is deprecated and will be removed in future, default behaviour is export stateful model"
            " please use --disable_stateful if you need model without state"
        )
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
        export_config, models_and_export_configs = _get_submodels_and_export_configs(
            model=model,
            task="text-generation-with-past",
            exporter="openvino",
            custom_export_configs={},
            custom_architecture=None,
            fn_get_submodels=None,
            preprocessors=None,
            _variant="default",
            monolith=False,
            library_name="transformers"
        )
        if "decoder_with_past_model" in models_and_export_configs:
            models_and_export_configs = {"model": models_and_export_configs["decoder_with_past_model"]}
        model.config.save_pretrained(ov_out_dir)
        files_subpaths = ["openvino_" + model_name + ".xml" for model_name in models_and_export_configs.keys()]
        export_models(
            models_and_export_configs=models_and_export_configs,
            output_dir=ov_out_dir,
            output_names=files_subpaths,
            input_shapes=dummy_shapes,
            device="cpu",
            ov_config=OVConfig(dtype="fp16") if args.precision == "FP16" else None,
            model_kwargs={},
            stateful=not args.disable_stateful,
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
        compression_modes = []
        for cw in args.compress_weights:
            if is_int8_compression(cw):
                compression_modes.append(cw)
        assert compression_modes, "Only INT8 compression supported for PyTorch backend"
        number_compression_modes = len(compression_modes)
        original_model = model
        for idx, compress_mode in enumerate(compression_modes):
            if number_compression_modes - idx > 1:
                model = copy.deepcopy(original_model)
            else:
                model = original_model

            _, models_and_export_configs = _get_submodels_and_export_configs(
                model=model,
                exporter="openvino",
                task="text-generation-with-past",
                custom_export_configs={},
                custom_architecture=None,
                fn_get_submodels=None,
                preprocessors=None,
                _variant="default",
                monolith=False,
                library_name="transformers"
            )

            compression_options = COMPRESSION_OPTIONS[compress_mode]
            models_and_export_configs = compress_torchmodels(
                models_and_export_configs,
                stateful=not args.disable_stateful,
                dummy_shapes=dummy_shapes,
                compression_options=compression_options,
            )

            pt_out_dir = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=precision, compression=compress_mode)
            )
            model.config.save_pretrained(pt_out_dir)
            export_models(
                models_and_export_configs=models_and_export_configs,
                output_dir=pt_out_dir,
                output_names=files_subpaths,
                input_shapes=dummy_shapes,
                device="cpu",
                ov_config=OVConfig(dtype="fp16") if args.precision == "FP16" else None,
                model_kwargs={},
                stateful=not args.disable_stateful,
            )
            save_tokenizer(tok, pt_out_dir)
    return


def convert_causal_lm(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    ov_out_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR
    precision = args.precision
    compression_only = (
        is_ov_compression(args)
        and not is_torch_compression(args)
        and is_ov_model_provided(args.model_id, ov_out_dir, precision)
    )
    model_kwargs = {}
    if post_init is not None:
        model_kwargs["torch_dtype"] = torch.float32
    model = None
    if not compression_only:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, trust_remote_code=True, config=config, **model_kwargs
        )
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
            compression_modes = []
            for cw in args.compress_weights:
                if is_int8_compression(cw):
                    compression_modes.append(cw)
            assert compression_modes, "Only INT8 compression supported for PyTorch backend"
            for idx, compress_mode in enumerate(compression_modes):
                if idx > 0:
                    pt_model = AutoModelForSeq2SeqLM.from_pretrained(
                        args.model_id,
                        trust_remote_code=True,
                        config=config,
                    )

                export_config_constructor = TasksManager.get_exporter_config_constructor(
                    model=pt_model, exporter="openvino", task="text2text-generation"
                )
                export_config = export_config_constructor(pt_model.config, use_past=True)
                models_and_export_configs = get_encoder_decoder_models_for_export(pt_model, export_config)

                compression_options = COMPRESSION_OPTIONS[compress_mode]
                models_and_export_configs = compress_torchmodels(
                    models_and_export_configs, compression_options=compression_options
                )

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
                    / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compress_mode)
                )
                try:
                    export_models(
                        models_and_export_configs=models_and_export_configs,
                        opset=export_config.DEFAULT_ONNX_OPSET,
                        output_dir=save_dir_path,
                        output_names=output_names,
                        ov_config=OVConfig(dtype="fp16") if args.precision == "FP16" else None,
                        stateful=False
                    )
                    save_tokenizer(tok, save_dir_path)
                except Exception as ex:
                    log.warning(f"PT weights compression failed with {ex}, please use OpenVINO backend instead")

        del pt_model
        gc.collect()

    # skip openvino compression pipeline if pytorch compression pipeline was used
    if pt_compress_weights:
        return

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
            load_in_8bit=False
        )
        if is_fp16(args):
            model.half()
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
    if isinstance(vae_encoder, AutoencoderTiny):
        vae_encoder.forward = lambda sample: {"latent_sample": vae_encoder.encode(x=sample)["latents"]}
    else:
        vae_encoder.forward = lambda sample: {"latent_sample": vae_encoder.encode(x=sample)["latent_dist"].sample()}
    models_for_export["vae_encoder"] = vae_encoder

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = copy.deepcopy(pipeline.vae)
    if isinstance(vae_encoder, AutoencoderTiny):
        vae_decoder.forward = lambda latent_sample: vae_decoder.decode(latent_sample)
    else:
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
            exporter="openvino",
            task="feature-extraction",
            library_name="diffusers",
        )
        text_encoder_export_config = text_encoder_config_constructor(
            pipeline.text_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype
        )
        models_for_export["text_encoder"] = (models_for_export["text_encoder"], text_encoder_export_config)

    # U-NET
    export_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.unet,
        exporter="openvino",
        task="semantic-segmentation",
        model_type="unet",
        library_name="diffusers",
    )
    unet_export_config = export_config_constructor(pipeline.unet.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["unet"] = (models_for_export["unet"], unet_export_config)

    # VAE Encoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L565
    vae_encoder = models_for_export["vae_encoder"]
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_encoder,
        exporter="openvino",
        task="semantic-segmentation",
        model_type="vae-encoder",
        library_name="diffusers",
    )
    vae_export_config = vae_config_constructor(vae_encoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["vae_encoder"] = (vae_encoder, vae_export_config)

    # VAE Decoder https://github.com/huggingface/diffusers/blob/v0.11.1/src/diffusers/models/vae.py#L600
    vae_decoder = models_for_export["vae_decoder"]
    vae_config_constructor = TasksManager.get_exporter_config_constructor(
        model=vae_decoder,
        exporter="openvino",
        task="semantic-segmentation",
        model_type="vae-decoder",
        library_name="diffusers",
    )
    vae_export_config = vae_config_constructor(vae_decoder.config, int_dtype=int_dtype, float_dtype=float_dtype)
    models_for_export["vae_decoder"] = (vae_decoder, vae_export_config)

    if "text_encoder_2" in models_for_export:
        export_config_constructor = TasksManager.get_exporter_config_constructor(
            model=pipeline.text_encoder_2,
            exporter="openvino",
            task="feature-extraction",
            model_type="clip-text-with-projection",
            library_name="diffusers",
        )
        export_config = export_config_constructor(
            pipeline.text_encoder_2.config, int_dtype=int_dtype, float_dtype=float_dtype
        )
        models_for_export["text_encoder_2"] = (models_for_export["text_encoder_2"], export_config)

    return models_for_export


def convert_sd_prepared_for_export_common(pipeline, models_and_export_configs, output_dir, args):
    for model_name in models_and_export_configs:
        subcomponent = models_and_export_configs[model_name][0]
        if hasattr(subcomponent, "save_config"):
            subcomponent.save_config(output_dir / model_name)
        elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
            subcomponent.config.save_pretrained(output_dir / model_name)

            files_subpaths = [Path(name_dir) / OV_XML_FILE_NAME for name_dir in models_and_export_configs]

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
        models_and_export_configs=models_and_export_configs,
        output_dir=output_dir,
        output_names=files_subpaths,
        ov_config=OVConfig(dtype="fp16") if args.precision == "FP16" else None,
        stateful=False
    )


def convert_sd_common(pipeline, output_dir, args):
    models_and_export_configs = get_stable_diffusion_models_for_export(pipeline)
    convert_sd_prepared_for_export_common(pipeline, models_and_export_configs, output_dir, args)


def convert_sd(args):
    pt_compress_weights = is_torch_compression(args)
    pt_model = StableDiffusionPipeline.from_pretrained(args.model_id)
    if args.save_orig:
        pt_model.save_pretrained(Path(args.output_dir) / PYTORCH_DIR)

    output_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
    models_and_export_configs = get_stable_diffusion_models_for_export(pt_model)
    convert_sd_prepared_for_export_common(pt_model, models_and_export_configs, output_dir, args)

    if pt_compress_weights:
        compression_modes = []
        for cw in args.compress_weights:
            if is_int8_compression(cw):
                compression_modes.append(cw)
        assert compression_modes, "Only INT8 compression supported for PyTorch backend"
        for idx, compress_mode in enumerate(compression_modes):
            if idx > 0:
                pt_model = StableDiffusionPipeline.from_pretrained(args.model_id)
                models_and_export_configs = get_stable_diffusion_models_for_export(pt_model)

            target_models_and_export_configs = {
                k: models_and_export_configs[k] for k in ("text_encoder", "unet", "vae_decoder")
            }
            compression_options = COMPRESSION_OPTIONS[compress_mode]
            models_and_export_configs.update(
                compress_torchmodels(target_models_and_export_configs, compression_options=compression_options)
            )

            output = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compress_mode)
            )
            convert_sd_prepared_for_export_common(pt_model, models_and_export_configs, output, args)
    del pt_model
    gc.collect()

    if is_ov_compression(args):
        for weigths_compression_option in args.compress_weights:
            if not is_int8_compression(weigths_compression_option):
                log.warning(
                    f"Weights compression {weigths_compression_option} is not supported for SD, will be ignored"
                )
                continue
            model = OVStableDiffusionPipeline.from_pretrained(output_dir, compile=False)
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            model.text_encoder.model = nncf.compress_weights(model.text_encoder.model)
            model.unet.model = nncf.compress_weights(model.unet.model)
            model.vae_decoder.model = nncf.compress_weights(model.vae_decoder.model)
            model.save_pretrained(ov_int8_dir)

            del model
            gc.collect()


def convert_lcm(args):
    pt_compress_weights = is_torch_compression(args)
    pt_model = StableDiffusionPipeline.from_pretrained(args.model_id)
    if args.save_orig:
        pt_model.save_pretrained(Path(args.output_dir) / PYTORCH_DIR)

    output_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
    models_and_export_configs = get_stable_diffusion_models_for_export(pt_model)
    convert_sd_prepared_for_export_common(pt_model, models_and_export_configs, output_dir, args)

    if pt_compress_weights:
        compression_modes = []
        for cw in args.compress_weights:
            if is_int8_compression(cw):
                compression_modes.append(cw)
        assert compression_modes, "Only INT8 compression supported for PyTorch backend"
        for idx, compress_mode in enumerate(compression_modes):
            if idx > 0:
                pt_model = StableDiffusionPipeline.from_pretrained(args.model_id)
                models_and_export_configs = get_stable_diffusion_models_for_export(pt_model)

            target_models_and_export_configs = {
                k: models_and_export_configs[k] for k in ("text_encoder", "unet", "vae_decoder")
            }
            compression_options = COMPRESSION_OPTIONS[compress_mode]
            models_and_export_configs.update(
                compress_torchmodels(target_models_and_export_configs, compression_options=compression_options)
            )

            output = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compress_mode)
            )
            convert_sd_prepared_for_export_common(pt_model, models_and_export_configs, output, args)
    del pt_model
    gc.collect()

    if is_ov_compression(args):
        for weigths_compression_option in args.compress_weights:
            if not is_int8_compression(weigths_compression_option):
                log.warning(
                    f"Weights compression {weigths_compression_option} is not supported for LCM, will be ignored"
                )
                continue
            model = OVLatentConsistencyModelPipeline.from_pretrained(output_dir, compile=False)
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            model.text_encoder.model = nncf.compress_weights(model.text_encoder.model)
            model.unet.model = nncf.compress_weights(model.unet.model)
            model.vae_decoder.model = nncf.compress_weights(model.vae_decoder.model)
            model.save_pretrained(ov_int8_dir)

            del model
            gc.collect()


def convert_sdxl(args):
    pt_compress_weights = is_torch_compression(args)

    def build_pt_model(model_id):
        model_ids = [idx.replace(" ", "") for idx in model_id.split(",")]
        pt_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_ids[0])
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
                    vae = AutoencoderTiny.from_pretrained(additional_model)
                    pt_model.vae = vae
                    continue

        return pt_model

    pt_model = build_pt_model(args.model_id)
    if args.save_orig:
        pt_model.save_pretrained(Path(args.output_dir) / PYTORCH_DIR)

        del pt_model
        gc.collect()
        pt_model = build_pt_model(args.model_id)

    fp_out_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / args.precision
    models_and_export_configs = get_stable_diffusion_models_for_export(pt_model)
    convert_sd_prepared_for_export_common(pt_model, models_and_export_configs, fp_out_dir, args)

    if pt_compress_weights:
        compression_modes = []
        for cw in args.compress_weights:
            if is_int8_compression(cw):
                compression_modes.append(cw)
        assert compression_modes, "Only INT8 compression supported for PyTorch backend"
        for idx, compress_mode in enumerate(compression_modes):
            if idx > 0:
                pt_model = build_pt_model(args.model_id)
                models_and_export_configs = get_stable_diffusion_models_for_export(pt_model)

            compression_options = COMPRESSION_OPTIONS[compress_mode]
            models_and_export_configs = compress_torchmodels(
                models_and_export_configs, compression_options=compression_options
            )

            output = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compress_mode)
            )

            convert_sd_prepared_for_export_common(pt_model, models_and_export_configs, output, args)

    del pt_model
    gc.collect()

    if is_ov_compression(args):
        for weigths_compression_option in args.compress_weights:
            if not is_int8_compression(weigths_compression_option):
                log.warning(
                    f"Weights compression {weigths_compression_option} is not supported for SDXL, will be ignored"
                )
                continue
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            compression_options = COMPRESSION_OPTIONS[weigths_compression_option]
            model = OVStableDiffusionXLPipeline.from_pretrained(fp_out_dir, compile=False)
            model.text_encoder.model = nncf.compress_weights(model.text_encoder.model, **compression_options)
            if getattr(model, "text_encoder_2", None) is not None:
                model.text_encoder_2.model = nncf.compress_weights(model.text_encoder_2.model, **compression_options)
            model.unet.model = nncf.compress_weights(model.unet.model)
            model.vae_decoder.model = nncf.compress_weights(model.vae_decoder.model, **compression_options)
            if getattr(model, "vae_encoder", None) is not None:
                model.vae_encoder.model = nncf.compress_weights(model.vae_encoder.model, **compression_options)
            model.save_pretrained(ov_int8_dir)

            del model
            gc.collect()


def convert_ldm_super_res(args):
    pipeline = LDMSuperResolutionPipeline.from_pretrained(args.model_id)
    if args.save_orig:
        pipeline.save_pretrained(Path(args.output_dir) / PYTORCH_DIR)
    unet_example_input = (
        torch.zeros((1, 6, 128, 128)),
        torch.tensor(1, dtype=torch.int32),
    )

    class Decoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, latents):
            return self.model.decode(latents)

    decoder = Decoder(pipeline.vqvae)

    compress_to_fp16 = is_fp16(args)
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
    del ov_unet, ov_decoder
    gc.collect()

    pt_compress_weights = is_torch_compression(args)
    if pt_compress_weights:
        compression_modes = []
        for cw in args.compress_weights:
            if is_int8_compression(cw):
                compression_modes.append(cw)
        assert compression_modes, "Only INT8 compression supported for PyTorch backend"
        for idx, compress_mode in enumerate(compression_modes):
            if idx > 0:
                pipeline = LDMSuperResolutionPipeline.from_pretrained(args.model_id)
                decoder = Decoder(pipeline.vqvae)

            compression_options = COMPRESSION_OPTIONS[compress_mode]
            compressed_unet = nncf.compress_weights(
                pipeline.unet, dataset=nncf.Dataset([unet_example_input]), **compression_options
            )
            ov_compressed_unet = convert_model(compressed_unet, example_input=unet_example_input)
            ov_compressed_unet.inputs[1].get_node().set_element_type(OVType.i32)
            ov_compressed_unet.inputs[1].get_node().set_partial_shape(PartialShape([]))
            ov_compressed_unet.validate_nodes_and_infer_types()
            pt_out_dir = (
                Path(args.output_dir)
                / PYTORCH_DIR
                / OV_DIR
                / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=args.precision, compression=compress_mode)
            )
            save_model(
                ov_compressed_unet,
                pt_out_dir / "unet.xml",
                compress_to_fp16=compress_to_fp16,
            )
            pipeline.scheduler.save_config(pt_out_dir)
            decoder_example_input = torch.zeros(1, 3, 128, 128)
            compressed_decoder = nncf.compress_weights(
                decoder, dataset=nncf.Dataset([decoder_example_input]), **compression_options
            )
            ov_compressed_decoder = convert_model(compressed_decoder, example_input=decoder_example_input)
            save_model(ov_compressed_decoder, pt_out_dir / "vqvae.xml", compress_to_fp16=compress_to_fp16)

    if is_ov_compression(args):
        for weigths_compression_option in args.compress_weights:
            if not is_int8_compression(weigths_compression_option):
                log.warning(
                    f"Weights compression {weigths_compression_option} is not supported for LDM, will be ignored"
                )
                continue
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            ov_unet = Core().read_model(save_dir / "unet.xml")
            compressed_ov_unet = nncf.compress_weights(ov_unet)
            save_model(
                compressed_ov_unet,
                ov_int8_dir / "unet.xml",
                compress_to_fp16=compress_to_fp16,
            )
            ov_decoder = Core().read_model(save_dir / "vqvae.xml")
            compressed_ov_decoder = nncf.compress_weights(ov_decoder)
            save_model(
                compressed_ov_decoder,
                ov_int8_dir / "vqvae.xml",
                compress_to_fp16=compress_to_fp16,
            )
            pipeline.scheduler.save_config(ov_int8_dir)


def convert_mpt(args):
    @torch.no_grad
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

    remote_code = False
    pt_model = None
    try:
        config = AutoConfig.from_pretrained(args.model_id)
    except Exception:
        config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
        remote_code = True
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

        def create_model(model_id, config, model_kwargs):
            pt_model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=remote_code, config=config, **model_kwargs
            )
            pt_model.config.use_cache = True
            pt_model.eval()
            return pt_model

        pt_model = create_model(args.model_id, config, model_kwargs)

        if not remote_code:
            return convert_optimum_causallm_base(pt_model, args, config, compression_only)

        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / PYTORCH_DIR
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)

        ov_dir = Path(args.output_dir) / PYTORCH_DIR / OV_DIR / precision
        compress_to_fp16 = is_fp16(args)

        convert_to_ov(pt_model, tok, ov_dir, compress_to_fp16)
        if is_torch_compression(args):
            compression_modes = []
            for cw in args.compress_weights:
                if is_int8_compression(cw):
                    compression_modes.append(cw)
            assert compression_modes, "Only INT8 compression supported for PyTorch backend"

            dummy_inputs = {
                "input_ids": torch.ones((1, 10), dtype=torch.long),
                "attention_mask": torch.ones((1, 10), dtype=torch.long),
            }

            for idx, compress_mode in enumerate(compression_modes):
                if idx > 0:
                    pt_model = create_model(args.model_id, config, model_kwargs)

                compression_options = COMPRESSION_OPTIONS[compress_mode]
                compressed_pt_model = nncf.compress_weights(
                    pt_model, dataset=nncf.Dataset([dummy_inputs]), **compression_options
                )

                pt_path = (
                    Path(args.output_dir)
                    / PYTORCH_DIR
                    / OV_DIR
                    / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=precision, compression=compress_mode)
                )
                convert_to_ov(compressed_pt_model, tok, pt_path, compress_to_fp16)

    if is_ov_compression(args):
        if not remote_code:
            return convert_optimum_causallm_base(pt_model, args, config, compression_only)
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
    def make_dummy_input():
        last_token = torch.tensor([[130328]])
        past = torch.zeros(28, 2, 5, 1, 32, 128)
        position_ids = torch.tensor([[[2], [4]]])
        return {
            "input_ids": last_token,
            "past_key_values": past,
            "position_ids": position_ids,
        }

    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.torchscript = True
        dummy_input = make_dummy_input()
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
        args.compress_weights
        and not args.force_convert
        and not is_torch_compression(args)
        and is_ov_model_provided(args.model_id, args.output_dir, precision)
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

        def create_model(model_id, config, model_kwargs):
            pt_model = AutoModel.from_pretrained(model_id, trust_remote_code=True, config=config, **model_kwargs)
            pt_model.config.use_cache = True
            pt_model.to(torch.float32)
            pt_model.eval()
            return pt_model

        pt_model = create_model(args.model_id, config, model_kwargs)

        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / PYTORCH_DIR
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)
        convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16=compress_to_fp16)

        pt_compress_weights = is_torch_compression(args)
        if pt_compress_weights:
            compression_modes = []
            for cw in args.compress_weights:
                if is_int8_compression(cw):
                    compression_modes.append(cw)
            assert compression_modes, "Only INT8 compression supported for PyTorch backend"

            dummy_input = make_dummy_input()
            for idx, compress_mode in enumerate(compression_modes):
                if idx > 0:
                    pt_model = create_model(args.model_id, config, model_kwargs)

                compression_options = COMPRESSION_OPTIONS[compress_mode]
                compressed_pt_model = nncf.compress_weights(
                    pt_model, dataset=nncf.Dataset([dummy_input]), **compression_options
                )

                pt_out_path = (
                    Path(args.output_dir)
                    / PYTORCH_DIR
                    / OV_DIR
                    / PYTORCH_COMPRESS_WEIGHTS_DIR.format(precision=precision, compression=compress_mode)
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
    config = AutoConfig.from_pretrained(args.model_id)
    cuda, post_init = patch_gptq(config)
    model_kwargs = {}
    precision = args.precision
    compression_only = (
        args.compress_weights
        and not args.force_convert
        and not is_torch_compression(args)
        and is_ov_model_provided(args.model_id, args.output_dir, args.precision)
    )
    if post_init is not None:
        model_kwargs = {"torch_dtype": torch.float32}
    pt_model = None
    gptq_applied = is_gptq(config)
    precision = precision if not gptq_applied else GPTQ_DIR.format(precision=args.precision)
    if not compression_only:
        pt_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=AutoConfig.from_pretrained(args.model_id),
            **model_kwargs,
        )
        pt_model.config.use_cache = True
        pt_model.eval()

    convert_optimum_causallm_base(pt_model, args, config, compression_only)

    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_phi(args):
    trust_remote_code = False
    try:
        config = AutoConfig.from_pretrained(args.model_id)
    except Exception:
        config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
        trust_remote_code = True
    cuda, post_init = patch_gptq(config)
    model_kwargs = {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = trust_remote_code
    precision = args.precision
    compression_only = (
        args.compress_weights
        and not args.force_convert
        and not is_torch_compression(args)
        and is_ov_model_provided(args.model_id, args.output_dir, args.precision)
    )
    if post_init is not None:
        model_kwargs["torch_dtype"] = torch.float32
    pt_model = None
    gptq_applied = is_gptq(config)
    precision = precision if not gptq_applied else GPTQ_DIR.format(precision=args.precision)
    if not compression_only:
        pt_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=AutoConfig.from_pretrained(args.model_id),
            **model_kwargs,
        )
        pt_model.config.use_cache = True
        pt_model.eval()

    convert_optimum_causallm_base(pt_model, args, config, compression_only)

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
    model_kwargs = {}
    precision = args.precision
    compression_only = (
        args.compress_weights
        and not args.force_convert
        and not is_torch_compression(args)
        and is_ov_model_provided(args.model_id, args.output_dir, precision)
    )
    if post_init is not None:
        model_kwargs = {
            "torch_dtype": torch.float32,
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
    "phi-": convert_phi,
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
        choices=["INT8", "INT8_ASYM", "INT8_SYM", "4BIT_DEFAULT", "4BIT_MAXIMUM", "INT4_SYM", "INT4_ASYM"],
        nargs="+",
        help=(
            "The weight compression option, e.g. INT8 - INT8 weights (deprecated, please use INT8_ASYM instead), "
            "4BIT_DEFAULT - for 4-bit compression with predefined configs with performance-accuracy trade-off, "
            "4BIT_MAXIMUM - for 4-bit compression with predefined configs for the best performance, "
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
    compression_group.add_argument(
        "--dataset",
        help=(
            "Dataset parameters for data-aware compression in format path,name,split,item_name "
            "(for example \"wikitext,wikitext-2-v1,train[:1000],text\") "
            "path,name,split - parameters for load_dataset from datasets "
            "and item_name is field name in dataset with text."
        ),
        default=None,
        type=str,
    )
    compression_group.add_argument(
        "--awq",
        action="store_true",
        help="Apply AWQ algorithm during compression",
    )
    add_stateful_model_arguments(parser)

    args = parser.parse_args()
    log.info(f"openvino runtime version: {get_version()}")
    model_type = get_convert_model_type(args.model_id.lower())
    converter = converters[model_type]
    converter(args)


main()
