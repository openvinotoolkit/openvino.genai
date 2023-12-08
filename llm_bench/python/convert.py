# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys
import gc
import time
import logging as log
from argparse import ArgumentParser
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Tuple, Dict, Optional
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline, LDMSuperResolutionPipeline, DiffusionPipeline
from diffusers import UNet2DConditionModel, AutoencoderTiny, LCMScheduler
from nncf import compress_weights
from openvino import Type, PartialShape, save_model, convert_model
from openvino.runtime import Core
from optimum.exporters import TasksManager
from optimum.exporters.tasks import make_backend_config_constructor_for_task
from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import (
    NormalizedTextConfig, NormalizedConfigManager, DEFAULT_DUMMY_SHAPES,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyInputGenerator
)
from optimum.exporters.onnx import get_encoder_decoder_models_for_export
from optimum.exporters.openvino import export_models
from optimum.utils.save_utils import maybe_load_preprocessors
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
from optimum.exporters.onnx import __main__ as optimum_main
try:
    from optimum.exporters.openvino.__main__ import _get_submodels_and_export_configs
except ImportError:
    from optimum.exporters.onnx.__main__ import _get_submodels_and_onnx_configs as _get_submodels_and_export_configs

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from utils.nncf_utils import COMPRESSION_OPTIONS, INT4_MODEL_CONFIGURATION, get_compressed_path
from utils.conversion_utils.convert_patch import patch_model_for_optimum_export
from utils.conversion_utils.better_transformer_patch import patch_model_with_bettertransformer


class BackendType(Enum):
    PYTORCH = 'pytorch'
    OPENVINO = 'openvino'


def save_tokenizer(tokenizer, out_dir):
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        log.error(f'tokenizer loading failed with {e}')


def compress_ov_model_weights_helper(ov_model, tok, config, out_path, compress_weights_format="INT8", fp16=False, args={}, model_name="openvino_model"):
    compression_args = None
    if "4BIT_DEFAULT" in compress_weights_format:
        model_id = out_path.parents[3].name
        if model_id in INT4_MODEL_CONFIGURATION:
            compression_args = INT4_MODEL_CONFIGURATION[model_id]
        else:
            compression_args = COMPRESSION_OPTIONS["INT4_SYM"]

    if compression_args is None:
        compression_args = COMPRESSION_OPTIONS[compress_weights_format]
        if args.ratio is not None:
            compression_args["ratio"] = args.ratio
        if args.group_size is not None:
            compression_args["group_size"] = args.group_size
    log.info("Compression options:")
    log.info(compression_args)
    compressed_ov_model = compress_weights(ov_model, **compression_args)
    save_ov_model_helper(compressed_ov_model, out_path, model_name, fp16=fp16, tok=tok, config=config)


def save_ov_model_helper(ov_model, out_path, model_name='openvino_model', fp16=False, tok=None, config=None):
    model_name = model_name or "openvino_model"
    save_model(ov_model, Path(out_path) / f'{model_name}.xml', compress_to_fp16=fp16)
    if tok is not None:
        save_tokenizer(tok, out_path)
    if config is not None:
        config.save_pretrained(out_path)


def is_gptq(config):
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] == "gptq"


def patch_gptq(config):
    do_gptq_patching = False
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    do_gptq_patching = quantization_config and quantization_config["quant_method"] == "gptq"
    orig_cuda_check = torch.cuda.is_available
    orig_post_init_model = None
    if do_gptq_patching:
        torch.set_default_dtype(torch.float32)
        torch.cuda.is_available = lambda: True

        from optimum.gptq import GPTQQuantizer

        orig_post_init_model = GPTQQuantizer.post_init_model

        def post_init_model(self, model):
            from auto_gptq import exllama_set_max_input_length

            class StoreAttr(object):
                pass

            model.quantize_config = StoreAttr()
            model.quantize_config.desc_act = self.desc_act
            if self.desc_act and not self.disable_exllama and self.max_input_length is not None:
                model = exllama_set_max_input_length(model, self.max_input_length)
            return model

        GPTQQuantizer.post_init_model = post_init_model
    return orig_cuda_check, orig_post_init_model


def unpatch_gptq(orig_cuda_check, orig_post_init_model):
    from optimum.gptq import GPTQQuantizer
    torch.cuda.is_available = orig_cuda_check
    GPTQQuantizer.post_init_model = orig_post_init_model


class TextDecoderWithPositionIdsOnnxConfig(TextDecoderOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs

        # Decoders based on GPT2 require a position_ids input to avoid
        # generating wrong position_ids in the model itself:
        # https://github.com/huggingface/transformers/blob/v4.33.1/src/transformers/models/gpt2/modeling_gpt2.py#L802
        if not self.no_position_ids and "text-generation" in self.task:
            common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

        return common_inputs


def convert_optimum_causallm_base(model, args):
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = patch_model_for_optimum_export(model)
    model_config = model.config
    gptq_applied = is_gptq(model_config)
    precision = args.precision if not gptq_applied else f"GPTQ_INT4-{args.precision}"
    if gptq_applied and args.compress_weights:
        log.info("Weights compression will be skipped for GPTQ models")
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
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
        monolith=False
    )
    if "decoder_with_past_model" in models_and_onnx_configs:
        models_and_onnx_configs = {"model": models_and_onnx_configs["decoder_with_past_model"]}
    if args.bettertransformer:
        models_and_onnx_configs["model"] = (patch_model_with_bettertransformer(*models_and_onnx_configs["model"]), models_and_onnx_configs["model"][1])
    ov_out_dir = Path(args.output_dir) / 'pytorch/dldt' / precision
    model.config.save_pretrained(ov_out_dir)
    files_subpaths = ["openvino_" + model_name + ".xml" for model_name in models_and_onnx_configs.keys()]
    export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        output_dir=ov_out_dir,
        output_names=files_subpaths,
        input_shapes=dummy_shapes,
        device="cpu",
        fp16=args.precision == "FP16",
        int8=False,
        model_kwargs={},
    )
    save_tokenizer(tok, ov_out_dir)
    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends and not gptq_applied:
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            optimized_dir = get_compressed_path(args.output_dir, args.precision, compress_option)
            model.config.save_pretrained(optimized_dir)
            fp_dir = ov_out_dir
            ir_model = Core().read_model(fp_dir / files_subpaths[0])

            compress_ov_model_weights_helper(ir_model, tok, model.config, optimized_dir, compress_option, args.precision == "FP16", args)

    if pt_compress_weights and not gptq_applied:
        compressed_model = compress_weights(model)
        onnx_config, models_and_onnx_configs = _get_submodels_and_export_configs(
            model=compressed_model,
            task="text-generation-with-past",
            custom_onnx_configs={},
            custom_architecture=None,
            fn_get_submodels=None,
            preprocessors=None,
            _variant="default",
            monolith=False
        )
        pt_out_dir = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        model.config.save_pretrained(pt_out_dir)
        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=pt_out_dir,
            output_names=files_subpaths,
            input_shapes=dummy_shapes,
            device="cpu",
            fp16=args.precision == "FP16",
            int8=False,
            model_kwargs={},
        )
        save_tokenizer(tok, pt_out_dir)
    return


def convert_causal_lm(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=config
    )
    convert_optimum_causallm_base(model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_seq2seq(args):
    tokenizer_id = args.model_id if 'blenderbot-9B' not in args.model_id else 'facebook/blenderbot-3B'
    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    start = time.perf_counter()
    if args.save_orig or pt_compress_weights:
        pt_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
        )
        if args.save_orig:
            pt_out_dir = Path(args.output_dir) / 'pytorch'
            pt_model.save_pretrained(pt_out_dir)
            save_tokenizer(tok, pt_out_dir)
        if pt_compress_weights:
            compressed_pt_model = compress_weights(pt_model)
            onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=pt_model, exporter='onnx', task='text2text-generation')
            onnx_config = onnx_config_constructor(pt_model.config, use_past=True)
            models_and_onnx_configs = get_encoder_decoder_models_for_export(compressed_pt_model, onnx_config)
            encoder_file_name = Path('encoder') / OV_ENCODER_NAME
            decoder_file_name = Path('decoder') / OV_DECODER_NAME
            decoder_with_past_file_name = Path('decoder_with_past') / OV_DECODER_WITH_PAST_NAME

            output_names = [encoder_file_name, decoder_file_name, decoder_with_past_file_name]
            save_dir_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
            try:
                export_models(
                    models_and_onnx_configs=models_and_onnx_configs,
                    opset=onnx_config.DEFAULT_ONNX_OPSET,
                    output_dir=save_dir_path,
                    output_names=output_names,
                )
                save_tokenizer(tok, save_dir_path)
            except Exception as ex:
                log.warning(f'PT weights compression failed with {ex}, please use OpenVINO backend instead')

        del pt_model
        gc.collect()

    model = OVModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        export=True,
        compile=False,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    end = time.perf_counter()
    log.info(f'Conversion total time {end - start}s')

    start1 = time.perf_counter()
    ov_out_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    model.save_pretrained(ov_out_dir)
    end1 = time.perf_counter()
    log.info(f'Serialization total time {end1 - start1}s')

    save_tokenizer(tok, ov_out_dir)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            optimized_dir = get_compressed_path(args.output_dir, args.precision, compress_option)
            compress_ov_model_weights_helper(
                model.encoder.model, tok, model.config, optimized_dir, compress_option,
                args.precision == "FP16", args, "openvino_encoder_model"
            )
            compress_ov_model_weights_helper(
                model.decoder.model, tok, model.config, optimized_dir, compress_option,
                args.precision == "FP16", args, "openvino_decoder_model"
            )
            if model.decoder_with_past:
                compress_ov_model_weights_helper(
                    model.decoder_with_past.model, tok, model.config, optimized_dir, compress_option,
                    args.precision == "FP16", args, "openvino_decoder_with_past_model"
                )

    del model
    gc.collect()


def convert_sd(args):
    start = time.perf_counter()
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    if args.save_orig or pt_compress_weights:
        pt_model = StableDiffusionPipeline.from_pretrained(args.model_id)
        if args.save_orig:
            pt_model.save_pretrained(Path(args.output_dir) / 'pytorch')
        if pt_compress_weights:
            wc_text_encoder = compress_weights(pt_model.text_encoder)
            wc_unet = compress_weights(pt_model.unet)
            wc_vae = compress_weights(pt_model.vae)
            pt_model.text_encoder = wc_text_encoder
            pt_model.unet = wc_unet
            pt_model.vae = wc_vae
            _, models_and_onnx_configs = optimum_main._get_submodels_and_onnx_configs(
                model=pt_model,
                task='stable-diffusion',
                monolith=False,
                custom_onnx_configs={},
                custom_architecture=False,
                _variant='default',
            )
            output = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
            for model_name in models_and_onnx_configs:
                subcomponent = models_and_onnx_configs[model_name][0]
                if hasattr(subcomponent, 'save_config'):
                    subcomponent.save_config(output / model_name)
                elif hasattr(subcomponent, 'config') and hasattr(subcomponent.config, 'save_pretrained'):
                    subcomponent.config.save_pretrained(output / model_name)

            files_subpaths = [Path(name_dir) / OV_XML_FILE_NAME for name_dir in models_and_onnx_configs]

            # Saving the additional components needed to perform inference.
            pt_model.scheduler.save_pretrained(output.joinpath('scheduler'))

            feature_extractor = getattr(pt_model, 'feature_extractor', None)
            if feature_extractor is not None:
                feature_extractor.save_pretrained(output.joinpath('feature_extractor'))

            tokenizer = getattr(pt_model, 'tokenizer', None)
            if tokenizer is not None:
                tokenizer.save_pretrained(output.joinpath('tokenizer'))

            tokenizer_2 = getattr(pt_model, 'tokenizer_2', None)
            if tokenizer_2 is not None:
                tokenizer_2.save_pretrained(output.joinpath('tokenizer_2'))

            pt_model.save_config(output)

            export_models(
                models_and_onnx_configs=models_and_onnx_configs,
                output_dir=output,
                output_names=files_subpaths,
            )

        del pt_model
        gc.collect()

    model = OVStableDiffusionPipeline.from_pretrained(args.model_id, export=True, compile=False)
    end = time.perf_counter()
    log.info(f'Conversion total time {end - start}s')

    if args.precision == 'FP16':
        model.half()
    start1 = time.perf_counter()
    model.save_pretrained(Path(args.output_dir) / 'pytorch/dldt' / args.precision)
    end1 = time.perf_counter()
    log.info(f'Serialization total time {end1 - start1}s')

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        for weigths_compression_option in args.compress_weights:
            if weigths_compression_option != "INT8":
                log.warning("Weights compression {weigths_compression_option} does not supported for SD, will be ignored")
                continue
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            model.text_encoder.model = compress_weights(model.text_encoder.model)
            model.unet.model = compress_weights(model.unet.model)
            model.vae_decoder.model = compress_weights(model.vae_decoder.model)
            model.save_pretrained(ov_int8_dir)

            # Saving the additional components needed to perform inference.
            model.scheduler.save_pretrained(ov_int8_dir.joinpath('scheduler'))

            feature_extractor = getattr(model, 'feature_extractor', None)
            if feature_extractor is not None:
                feature_extractor.save_pretrained(ov_int8_dir.joinpath('feature_extractor'))

            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is not None:
                tokenizer.save_pretrained(ov_int8_dir.joinpath('tokenizer'))

            tokenizer_2 = getattr(model, 'tokenizer_2', None)
            if tokenizer_2 is not None:
                tokenizer_2.save_pretrained(ov_int8_dir.joinpath('tokenizer_2'))

            model.save_config(ov_int8_dir)

    del model
    gc.collect()


def convert_lcm(args):
    start = time.perf_counter()
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    if args.save_orig or pt_compress_weights:
        pt_model = DiffusionPipeline.from_pretrained(args.model_id)
        if args.save_orig:
            pt_model.save_pretrained(Path(args.output_dir) / 'pytorch')
        if pt_compress_weights:
            wc_text_encoder = compress_weights(pt_model.text_encoder)
            wc_unet = compress_weights(pt_model.unet)
            wc_vae = compress_weights(pt_model.vae)
            pt_model.text_encoder = wc_text_encoder
            pt_model.unet = wc_unet
            pt_model.vae = wc_vae
            _, models_and_onnx_configs = optimum_main._get_submodels_and_onnx_configs(
                model=pt_model,
                task='stable-diffusion',
                monolith=False,
                custom_onnx_configs={},
                custom_architecture=False,
                _variant='default',
            )
            output = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
            for model_name in models_and_onnx_configs:
                subcomponent = models_and_onnx_configs[model_name][0]
                if hasattr(subcomponent, 'save_config'):
                    subcomponent.save_config(output / model_name)
                elif hasattr(subcomponent, 'config') and hasattr(subcomponent.config, 'save_pretrained'):
                    subcomponent.config.save_pretrained(output / model_name)

            files_subpaths = [Path(name_dir) / OV_XML_FILE_NAME for name_dir in models_and_onnx_configs]

            # Saving the additional components needed to perform inference.
            pt_model.scheduler.save_pretrained(output.joinpath('scheduler'))

            feature_extractor = getattr(pt_model, 'feature_extractor', None)
            if feature_extractor is not None:
                feature_extractor.save_pretrained(output.joinpath('feature_extractor'))

            tokenizer = getattr(pt_model, 'tokenizer', None)
            if tokenizer is not None:
                tokenizer.save_pretrained(output.joinpath('tokenizer'))

            tokenizer_2 = getattr(pt_model, 'tokenizer_2', None)
            if tokenizer_2 is not None:
                tokenizer_2.save_pretrained(output.joinpath('tokenizer_2'))

            pt_model.save_config(output)

            export_models(
                models_and_onnx_configs=models_and_onnx_configs,
                output_dir=output,
                output_names=files_subpaths,
            )

        del pt_model
        gc.collect()

    model = OVLatentConsistencyModelPipeline.from_pretrained(args.model_id, export=True, compile=False)
    end = time.perf_counter()
    log.info(f'Conversion total time {end - start}s')

    if args.precision == 'FP16':
        model.half()
    start1 = time.perf_counter()
    model.save_pretrained(Path(args.output_dir) / 'pytorch/dldt' / args.precision)
    end1 = time.perf_counter()
    log.info(f'Serialization total time {end1 - start1}s')

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        for weigths_compression_option in args.compress_weights:
            if weigths_compression_option != "INT8":
                log.warning("Weights compression {weigths_compression_option} does not supported for LCM, will be ignored")
                continue
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            model.text_encoder.model = compress_weights(model.text_encoder.model)
            model.unet.model = compress_weights(model.unet.model)
            model.vae_decoder.model = compress_weights(model.vae_decoder.model)
            model.save_pretrained(ov_int8_dir)

            # Saving the additional components needed to perform inference.
            model.scheduler.save_pretrained(ov_int8_dir.joinpath('scheduler'))

            feature_extractor = getattr(model, 'feature_extractor', None)
            if feature_extractor is not None:
                feature_extractor.save_pretrained(ov_int8_dir.joinpath('feature_extractor'))

            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is not None:
                tokenizer.save_pretrained(ov_int8_dir.joinpath('tokenizer'))

            tokenizer_2 = getattr(model, 'tokenizer_2', None)
            if tokenizer_2 is not None:
                tokenizer_2.save_pretrained(ov_int8_dir.joinpath('tokenizer_2'))

            model.save_config(ov_int8_dir)

    del model
    gc.collect()


def convert_sdxl(args):
    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends

    def build_pt_model(model_id):
        model_ids = [idx.replace(" ", "") for idx in model_id.split(',')]
        pt_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_ids[0])
        tiny_vae = False
        if len(model_ids) > 1:
            for additional_model in model_ids[1:]:
                if 'lora' in additional_model:
                    pt_model.load_lora_weights(additional_model)
                    pt_model.fuse_lora()
                    if 'lcm' in additional_model:
                        pt_model.scheduler = LCMScheduler.from_config(pt_model.scheduler.config)
                    continue

                if 'lcm' in additional_model and 'lora' not in additional_model:
                    unet = UNet2DConditionModel.from_pretrained(additional_model)
                    pt_model.unet = unet
                    pt_model.scheduler = LCMScheduler.from_config(pt_model.scheduler.config)
                    continue

                if 'tae' in additional_model:
                    tiny_vae = True
                    vae = AutoencoderTiny.from_pretrained(additional_model)
                    pt_model.vae = vae
                    continue

        preprocessors = maybe_load_preprocessors(model_ids[0])
        return pt_model, preprocessors, tiny_vae

    def convert_pt_to_ov(pt_model, preprocessors, output_dir, fp16, tiny_vae):
        _, models_and_onnx_configs = optimum_main._get_submodels_and_onnx_configs(
            model=pt_model,
            task='stable-diffusion-xl',
            monolith=False,
            custom_onnx_configs={},
            custom_architecture=False,
            _variant='default',
            preprocessors=preprocessors,
            legacy=False
        )
        if tiny_vae:
            models_and_onnx_configs["vae_encoder"][0].forward = (
                lambda sample: {"latent_sample": models_and_onnx_configs["vae_encoder"][0].encode(x=sample)["latents"]}
            )
            models_and_onnx_configs["vae_decoder"][0].forward = (
                lambda latent_sample: models_and_onnx_configs["vae_decoder"][0].decode(latent_sample)
            )
        for model_name in models_and_onnx_configs:
            subcomponent = models_and_onnx_configs[model_name][0]

            if hasattr(subcomponent, 'save_config'):
                subcomponent.save_config(output_dir / model_name)
            elif hasattr(subcomponent, 'config') and hasattr(subcomponent.config, 'save_pretrained'):
                subcomponent.config.save_pretrained(output_dir / model_name)

        files_subpaths = [Path(name_dir) / OV_XML_FILE_NAME for name_dir in models_and_onnx_configs]

        # Saving the additional components needed to perform inference.
        pt_model.scheduler.save_pretrained(output_dir.joinpath('scheduler'))

        feature_extractor = getattr(pt_model, 'feature_extractor', None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output_dir.joinpath('feature_extractor'))

        tokenizer = getattr(pt_model, 'tokenizer', None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir.joinpath('tokenizer'))

        tokenizer_2 = getattr(pt_model, 'tokenizer_2', None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output_dir.joinpath('tokenizer_2'))

        pt_model.save_config(output_dir)

        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=output_dir,
            output_names=files_subpaths,
            fp16=fp16,
            int8=False
        )

    pt_model, preprocessors, tiny_vae = build_pt_model(args.model_id)
    if args.save_orig:
        pt_model.save_pretrained(Path(args.output_dir) / 'pytorch')
    if pt_compress_weights:
        output = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        pt_model.text_encoder = compress_weights(pt_model.text_encoder)
        pt_model.unet = compress_weights(pt_model.unet)
        pt_model.vae = compress_weights(pt_model.vae)
        if getattr(pt_model, 'text_encoder_2', None) is not None:
            pt_model.text_encoder_2 = compress_weights(pt_model.text_encoder_2)
        convert_pt_to_ov(pt_model, output, args.precision == "FP16", tiny_vae)
        del pt_model
        gc.collect()
        pt_model, preprocessors, tiny_vae = build_pt_model(args.model_id)

    fp_out_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    convert_pt_to_ov(pt_model, preprocessors, fp_out_dir, args.precision == "FP16", tiny_vae)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        for weigths_compression_option in args.compress_weights:
            if weigths_compression_option != "INT8":
                log.warning("Weights compression {weigths_compression_option} does not supported for SDXL, will be ignored")
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
        pipeline.save_pretrained(Path(args.output_dir) / 'pytorch')
    unet_example_input = [torch.zeros((1, 6, 128, 128)), torch.tensor(1, dtype=torch.int32)]

    class Decoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, latents):
            return self.model.decode(latents)

    decoder = Decoder(pipeline.vqvae)

    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    compress_to_fp16 = args.precision == 'FP16'
    if pt_compress_weights:
        compressed_unet = compress_weights(pipeline.unet)
        ov_compressed_unet = convert_model(compressed_unet, example_input=unet_example_input)
        ov_compressed_unet.inputs[1].get_node().set_element_type(Type.i32)
        ov_compressed_unet.inputs[1].get_node().set_partial_shape(PartialShape([]))
        ov_compressed_unet.validate_nodes_and_infer_types()
        pt_out_dir = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        save_model(ov_compressed_unet, pt_out_dir / 'unet.xml', compress_to_fp16=compress_to_fp16)
        pipeline.scheduler.save_config(pt_out_dir)
        # Couldn't compress decoder weights (RuntimeError: cdist only supports floating-point dtypes, X2 got: Byte)
        ov_decoder = convert_model(decoder, example_input=torch.zeros((1, 3, 128, 128)))
        save_model(ov_decoder, pt_out_dir / 'vqvae.xml', compress_to_fp16=compress_to_fp16)

    # convert model to OpenVINO IR
    ov_unet = convert_model(pipeline.unet, example_input=unet_example_input)
    ov_unet.inputs[1].get_node().set_element_type(Type.i32)
    ov_unet.inputs[1].get_node().set_partial_shape(PartialShape([]))
    ov_unet.validate_nodes_and_infer_types()
    save_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    save_model(ov_unet, save_dir / 'unet.xml', compress_to_fp16=compress_to_fp16)
    ov_decoder = convert_model(decoder, example_input=torch.zeros((1, 3, 128, 128)))
    save_model(ov_decoder, save_dir / 'vqvae.xml', compress_to_fp16=compress_to_fp16)
    pipeline.scheduler.save_config(save_dir)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        for weigths_compression_option in args.compress_weights:
            if weigths_compression_option != "INT8":
                log.warning("Weights compression {weigths_compression_option} does not supported for LDM, will be ignored")
                continue
            ov_int8_dir = get_compressed_path(args.output_dir, args.precision, weigths_compression_option)
            compressed_ov_unet = compress_weights(ov_unet)
            save_model(compressed_ov_unet, ov_int8_dir / 'unet.xml', compress_to_fp16=compress_to_fp16)
            compressed_ov_decoder = compress_weights(ov_decoder)
            save_model(compressed_ov_decoder, ov_int8_dir / 'vqvae.xml', compress_to_fp16=compress_to_fp16)
            pipeline.scheduler.save_config(ov_int8_dir)


def convert_mpt(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.use_cache = True
        outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long), attention_mask=torch.ones((1, 10), dtype=torch.long))
        old = outs.past_key_values[0][0].ndim == 3
        inputs = ['input_ids']
        outputs = ['logits']

        dynamic_shapes = {'input_ids': {1: 'seq_len'}, 'attention_mask': {1: 'seq_len'}}

        for idx in range(len(outs.past_key_values)):
            inputs.extend([f'past_key_values.{idx}.key', f'past_key_values.{idx}.value'])
            dynamic_shapes[inputs[-1]] = {2: 'past_sequence + sequence'}
            dynamic_shapes[inputs[-2]] = {3 if not old else 2: 'past_sequence + sequence'}
            outputs.extend([f'present.{idx}.key', f'present.{idx}.value'])

        inputs.append('attention_mask')
        dummy_inputs = {
            'input_ids': torch.ones((1, 2), dtype=torch.long),
            'past_key_values': outs.past_key_values,
            'attention_mask': torch.ones((1, 12), dtype=torch.long),
        }
        pt_model.config.torchscript = True
        orig_forward = pt_model.forward

        @wraps(orig_forward)
        def ts_patched_forward(input_ids: torch.Tensor, past_key_values: Tuple[Tuple[torch.Tensor]], attention_mask: torch.Tensor):
            pkv_list = list(past_key_values)
            outs = orig_forward(input_ids=input_ids, past_key_values=pkv_list, attention_mask=attention_mask)
            return (outs.logits, tuple(outs.past_key_values))

        pt_model.forward = ts_patched_forward
        ov_model = convert_model(pt_model, example_input=dummy_inputs)
        pt_model.forward = orig_forward

        for inp_name, m_input, input_data in zip(inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
            input_node = m_input.get_node()
            if input_node.element_type == Type.dynamic:
                m_input.get_node().set_element_type(Type.f32)
            shape = list(input_data.shape)
            if inp_name in dynamic_shapes:
                for k in dynamic_shapes[inp_name]:
                    shape[k] = -1
            input_node.set_partial_shape(PartialShape(shape))
            m_input.get_tensor().set_names({inp_name})

        for out, out_name in zip(ov_model.outputs, outputs):
            out.get_tensor().set_names({out_name})

        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    pt_model.config.use_cache = True
    pt_model.eval()

    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        pt_model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)

    ov_dir = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    compress_to_fp16 = args.precision == 'FP16'

    convert_to_ov(pt_model, tok, ov_dir, compress_to_fp16)
    if args.compress_weights:
        if BackendType.PYTORCH.value in args.compress_weights_backends:
            compressed_pt_model = compress_weights(pt_model)
            pt_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
            convert_to_ov(compressed_pt_model, tok, pt_path, compress_to_fp16)
        if BackendType.OPENVINO.value in args.compress_weights_backends:
            for compress_option in args.compress_weights:
                log.info(f"Compress model weights to {compress_option}")
                ov_model = Core().read_model(ov_dir / 'openvino_model.xml')
                ov_compressed_path = get_compressed_path(args.output_dir, args.precision, compress_option)
                compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_option, compress_to_fp16, args)


def convert_stablelm(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if not config.model_type.startswith('stablelm'):
        return convert_causal_lm(args)
    cuda, post_init = patch_gptq(config)
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    model_type = config.model_type.replace("_", "-")
    NormalizedConfigManager._conf[model_type] = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads"
    )
    TasksManager._SUPPORTED_MODEL_TYPE[model_type] = TasksManager._SUPPORTED_MODEL_TYPE['llama']
    convert_optimum_causallm_base(pt_model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_chatglm2(args):
    class ChatGLM2NormalizedConfig(NormalizedTextConfig):
        NUM_LAYERS = "num_layers"
        VOCAB_SIZE = "padded_vocab_size"

    class ChatGLM2DummyTextInputGenerator(DummyTextInputGenerator):
        SUPPORTED_INPUT_NAMES = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
        }

        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            input = super().generate(input_name, framework, int_dtype, float_dtype)
            if input_name == "attention_mask":
                input = torch.ones(input.shape, dtype=input.dtype)
            if input_name == "position_ids":
                bs = input.shape[0]
                input = torch.range(0, input.shape[1], dtype=input.dtype).repeat(bs, 1)
            return input

    class ChatGLM2DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
        def __init__(
            self,
            task: str,
            normalized_config: NormalizedTextConfig,
            batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
            sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
            random_batch_size_range: Optional[Tuple[int, int]] = None,
            random_sequence_length_range: Optional[Tuple[int, int]] = None,
            **kwargs,
        ):
            super().__init__(
                task=task,
                normalized_config=normalized_config,
                batch_size=batch_size,
                sequence_length=sequence_length,
                random_batch_size_range=random_batch_size_range,
                random_sequence_length_range=random_sequence_length_range,
            )
            self.multi_query_group_num = normalized_config.multi_query_group_num
            self.head_dim = self.hidden_size // self.num_attention_heads

        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            past_key_shape = (
                self.sequence_length,
                self.batch_size,
                self.multi_query_group_num,
                self.head_dim,
            )
            past_value_shape = (
                self.sequence_length,
                self.batch_size,
                self.multi_query_group_num,
                self.head_dim,
            )
            return [
                (
                    self.random_float_tensor(past_key_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(past_value_shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]

    class ChatGLM2OpenVINOConfig(TextDecoderOnnxConfig):
        NORMALIZED_CONFIG_CLASS = ChatGLM2NormalizedConfig
        DUMMY_INPUT_GENERATOR_CLASSES = (ChatGLM2DummyTextInputGenerator, ChatGLM2DummyPastKeyValuesGenerator)
        DUMMY_PKV_GENERATOR_CLASS = ChatGLM2DummyPastKeyValuesGenerator
        no_position_ids = False

        def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
            dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

            dummy_inputs = {}
            input_names = [key for key in self.inputs.keys() if not key.startswith("past_key_values")]
            if self.use_past_in_inputs and self.use_cache_branch is not False:
                input_names.append("past_key_values")

            for input_name in input_names:
                input_was_inserted = False
                for dummy_input_gen in dummy_inputs_generators:
                    if dummy_input_gen.supports_input(input_name):
                        dummy_inputs[input_name] = self.overwrite_shape_and_generate_input(
                            dummy_input_gen,
                            input_name,
                            framework,
                            input_shapes=kwargs,
                        )
                        input_was_inserted = True
                        break
                if not input_was_inserted:
                    raise RuntimeError(
                        f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                    )

            # refer to https://github.com/huggingface/optimum/pull/764
            cond1 = self.use_past_in_inputs
            cond2 = self.PAD_ATTENTION_MASK_TO_PAST
            cond3 = self.use_cache_branch is not False
            cond4 = "attention_mask" in dummy_inputs
            if (cond1 and cond2 and cond3 and cond4):
                # Obtain the past sequence length from the value instead of the key (Bloom).
                past_length = dummy_inputs["past_key_values"][0][1].shape[0]
                for k, v in dummy_inputs.items():
                    if k not in ["attention_mask", "past_key_values"]:
                        dummy_inputs[k] = v[:, -1:]

                dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                    dummy_inputs["attention_mask"],
                    desired_length=past_length + 1,
                    dim=1,
                    dtype=dummy_inputs["attention_mask"].dtype,
                )

            return dummy_inputs

        @property
        def inputs(self) -> Dict[str, Dict[int, str]]:
            common_inputs = super().inputs
            if not self.no_position_ids and self.task == "text-generation":
                common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

            return common_inputs

        def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
            """
            Fills `input_or_outputs` mapping with past_key_values dynamic axes considering the direction.

            Args:
                inputs_or_outputs (`Dict[str, Dict[int, str]]`): The mapping to fill.
                direction (`str`):
                    either "inputs" or "outputs", it specifies whether `input_or_outputs` is the input mapping or the
                    output mapping, this is important for axes naming.
            """
            if direction not in ["inputs", "outputs"]:
                raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

            if direction == "inputs":
                decoder_sequence_name = "past_sequence_length"
                name = "past_key_values"
            else:
                decoder_sequence_name = "past_sequence_length + 1"
                name = "present"

            for i in range(self._normalized_config.num_layers):
                inputs_or_outputs[f"{name}.{i}.key"] = {1: "batch_size", 0: decoder_sequence_name}
                inputs_or_outputs[f"{name}.{i}.value"] = {1: "batch_size", 0: decoder_sequence_name}

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    pt_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.float32
    )
    try:
        pt_model.to(torch.float32)
    except Exception:
        pass

    NormalizedConfigManager._conf[pt_model.config.model_type] = NormalizedTextConfig.with_args(
        num_layers="num_hidden_layers", num_attention_heads="num_attention_heads"
    )
    export_config = ChatGLM2OpenVINOConfig
    TasksManager._SUPPORTED_MODEL_TYPE[pt_model.config.model_type] = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
        'openvino': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
    }
    convert_optimum_causallm_base(pt_model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_chatglm(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        pt_model.config.torchscript = True
        last_token = torch.tensor([[130328]])
        past = torch.zeros(28, 2, 5, 1, 32, 128)
        position_ids = torch.tensor([[[2], [4]]])
        dummy_input = {
            'input_ids': last_token,
            'past_key_values': past,
            'position_ids': position_ids,
        }
        ov_model = convert_model(pt_model, example_input=dummy_input)
        ov_model.outputs[0].get_tensor().set_names({'logits'})
        for i in range(1, len(ov_model.outputs), 2):
            idx = (i - 1) // 2
            ov_model.outputs[i].get_tensor().set_names({f'present.{int(idx)}.key'})
            ov_model.outputs[i + 1].get_tensor().set_names({f'present.{int(idx)}.value'})
        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    pt_model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True),
    )
    pt_model.config.use_cache = True
    pt_model.to(torch.float32)
    pt_model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        pt_model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)

    compress_to_fp16 = args.precision == 'FP16'
    ov_out_path = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16=compress_to_fp16)

    pt_compress_weights = args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends
    if pt_compress_weights:
        compressed_pt_model = compress_weights(pt_model)
        pt_out_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        convert_to_ov(compressed_pt_model, tok, pt_out_path)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_model_path = ov_out_path / 'openvino_model.xml'
        ov_model = Core().read_model(ov_model_path)
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            ov_compressed_path = get_compressed_path(args.output_dir, args.precision, args.compress_weights)
            compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_to_fp16, compress_option, args)


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


def convert_falcon(args):
    def convert_to_ov(pt_model, tok, out_path, compress_to_fp16=False):
        outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long))
        inputs = ['input_ids']
        outputs = ['logits']

        dynamic_shapes = {'input_ids': {1: 'seq_len'}}

        for idx in range(len(outs.past_key_values)):
            inputs.extend([f'past_key_values.{idx}.key', f'past_key_values.{idx}.value'])
            dynamic_shapes[inputs[-1]] = {1: 'past_sequence + sequence'}
            dynamic_shapes[inputs[-2]] = {1: 'past_sequence + sequence'}
            outputs.extend([f'present.{idx}.key', f'present.{idx}.value'])

        dummy_inputs = {'input_ids': torch.ones((1, 2), dtype=torch.long), 'past_key_values': outs.past_key_values}
        flatten_inputs = flattenize_inputs(dummy_inputs.values())
        pt_model.config.torchscript = True
        ov_model = convert_model(pt_model, example_input=dummy_inputs)
        for port, input_data, input_name in zip(ov_model.inputs[1:], flatten_inputs[1:], inputs[1:]):
            port.get_node().set_element_type(Type.f32)
            shape = list(input_data.shape)
            shape[2] = -1
            port.get_node().set_partial_shape(PartialShape(shape))
            port.get_tensor().set_names({input_name})
        for idx, out_name in enumerate(outputs):
            ov_model.outputs[idx].get_tensor().set_names({out_name})
        ov_model.validate_nodes_and_infer_types()
        save_ov_model_helper(ov_model, out_path, fp16=compress_to_fp16, tok=tok, config=pt_model.config)

    pt_model = AutoModelForCausalLM.from_pretrained(args.model_id, config=AutoConfig.from_pretrained(args.model_id))
    pt_model.config.use_cache = True
    pt_model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / 'pytorch'
        pt_model.save_pretrained(pt_out_dir)
        save_tokenizer(tok, pt_out_dir)

    compress_to_fp16 = args.precision == 'FP16'

    ov_out_path = Path(args.output_dir) / 'pytorch/dldt' / args.precision
    convert_to_ov(pt_model, tok, ov_out_path, compress_to_fp16)

    if args.compress_weights and BackendType.PYTORCH.value in args.compress_weights_backends:
        pt_compressed_model = compress_weights(pt_model)
        pt_comp_path = Path(args.output_dir) / 'pytorch/dldt/compressed_weights' / f'PT_{args.precision}-INT8'
        convert_to_ov(pt_compressed_model, tok, pt_comp_path, compress_to_fp16)

    if args.compress_weights and BackendType.OPENVINO.value in args.compress_weights_backends:
        ov_model = Core().read_model(ov_out_path / 'openvino_model.xml')
        for compress_option in args.compress_weights:
            log.info(f"Compress model weights to {compress_option}")
            ov_compressed_path = get_compressed_path(args.output_dir, args.precision, compress_option)
            compress_ov_model_weights_helper(ov_model, tok, pt_model.config, ov_compressed_path, compress_to_fp16, compress_option, args)


def convert_jais(args):
    normalized_config = NormalizedTextConfig.with_args(num_layers='n_layer', num_attention_heads='n_head', hidden_size='n_embd')

    class JaisOpenVINOConfig(TextDecoderOnnxConfig):
        DEFAULT_ONNX_OPSET = 13
        NORMALIZED_CONFIG_CLASS = normalized_config

    TasksManager._SUPPORTED_MODEL_TYPE['jais'] = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(JaisOpenVINOConfig, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(JaisOpenVINOConfig, 'text-generation-with-past'),
        },
    }
    NormalizedConfigManager._conf['jais'] = normalized_config
    return convert_causal_lm(args)


def convert_baichaun(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    normalized_config = NormalizedTextConfig.with_args(num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size')
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.float32)
    try:
        model.to(torch.float32)
        if post_init is None:
            model(torch.ones([1, 10], dtype=torch.long))
    except Exception:
        pass

    class Baichaun2OpenVINOConfig(TextDecoderOnnxConfig):
        DEFAULT_ONNX_OPSET = 13
        NORMALIZED_CONFIG_CLASS = normalized_config

    export_config = Baichaun2OpenVINOConfig
    TasksManager._SUPPORTED_MODEL_TYPE[model.config.model_type] = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
        'openvino': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
    }
    NormalizedConfigManager._conf[model.config.model_type] = normalized_config
    try:
        # workaround issue with initialization
        convert_optimum_causallm_base(model, args)
    except Exception:
        convert_optimum_causallm_base(model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_qwen(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    revision = "c02ede58c0ab0045f5e4788c35842bec6a7baa0a" if post_init is not None else "2abd8e5777bb4ce9c8ab4be7dbbd0fe4526db78d"
    normalized_config = NormalizedTextConfig.with_args(num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size')
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.float32, revision=revision)
    try:
        model.to(torch.float32)
    except Exception:
        pass

    class QwenDummyInputsGenerator(DummyTextInputGenerator):
        SUPPORTED_INPUT_NAMES = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
        }

        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            input = super().generate(input_name, framework, int_dtype, float_dtype)
            if input_name == "input_ids":
                input = torch.tensor([[1583]])
            if input_name == "attention_mask":
                input = torch.ones((1, 7), dtype=input.dtype)
            if input_name == "position_ids":
                input = torch.tensor([[6]])
            return input

    class QwenDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            shape = (
                1,
                6,
                self.num_attention_heads,
                self.hidden_size // self.num_attention_heads,
            )
            return [
                (
                    torch.zeros(shape, dtype=torch.float32),
                    torch.zeros(shape, dtype=torch.float32),
                )
                for _ in range(self.num_layers)
            ]

    class QwenOpenVINOConfig(TextDecoderOnnxConfig):
        DEFAULT_ONNX_OPSET = 13
        NORMALIZED_CONFIG_CLASS = normalized_config
        DUMMY_INPUT_GENERATOR_CLASSES = (QwenDummyInputsGenerator, QwenDummyPastKeyValuesGenerator)
        DUMMY_PKV_GENERATOR_CLASS = QwenDummyPastKeyValuesGenerator

        def generate_dummy_inputs(self, framework: str = "pt", **kwargs):
            dummy_inputs_generators = self._create_dummy_input_generator_classes(**kwargs)

            dummy_inputs = {}
            input_names = [key for key in self.inputs.keys() if not key.startswith("past_key_values")]
            if self.use_past_in_inputs and self.use_cache_branch is not False:
                input_names.append("past_key_values")

            for input_name in input_names:
                input_was_inserted = False
                for dummy_input_gen in dummy_inputs_generators:
                    if dummy_input_gen.supports_input(input_name):
                        dummy_inputs[input_name] = self.overwrite_shape_and_generate_input(
                            dummy_input_gen,
                            input_name,
                            framework,
                            input_shapes=kwargs,
                        )
                        input_was_inserted = True
                        break
                if not input_was_inserted:
                    raise RuntimeError(
                        f'Could not generate dummy input for "{input_name}". Try adding a proper dummy input generator to the model ONNX config.'
                    )

            # refer to https://github.com/huggingface/optimum/pull/764
            cond1 = self.use_past_in_inputs
            cond2 = self.PAD_ATTENTION_MASK_TO_PAST
            cond3 = self.use_cache_branch is not False
            cond4 = "attention_mask" in dummy_inputs
            if (cond1 and cond2 and cond3 and cond4):
                # Obtain the past sequence length from the value instead of the key (Bloom).
                past_length = dummy_inputs["past_key_values"][0][1].shape[1]

                dummy_inputs["attention_mask"] = DummyInputGenerator.pad_input_on_dim(
                    dummy_inputs["attention_mask"],
                    desired_length=past_length + 1,
                    dim=1,
                    dtype=dummy_inputs["attention_mask"].dtype,
                )

            return dummy_inputs

        def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
            """
            Fills `input_or_outputs` mapping with past_key_values dynamic axes considering the direction.

            Args:
                inputs_or_outputs (`Dict[str, Dict[int, str]]`): The mapping to fill.
                direction (`str`):
                    either "inputs" or "outputs", it specifies whether `input_or_outputs` is the input mapping or the
                    output mapping, this is important for axes naming.
            """
            if direction not in ["inputs", "outputs"]:
                raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

            if direction == "inputs":
                decoder_sequence_name = "past_sequence_length"
                name = "past_key_values"
            else:
                decoder_sequence_name = "past_sequence_length + 1"
                name = "present"

            for i in range(self._normalized_config.num_layers):
                inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 1: decoder_sequence_name}
                inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 1: decoder_sequence_name}

    model_type = model.config.model_type.replace("-", "_")
    export_config = QwenOpenVINOConfig
    TasksManager._SUPPORTED_MODEL_TYPE[model_type] = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
        'openvino': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
    }
    NormalizedConfigManager._conf[model_type] = normalized_config
    convert_optimum_causallm_base(model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_mistral(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    if config.model_type in NormalizedConfigManager._conf:
        return convert_causal_lm(args)

    cuda, post_init = patch_gptq(config)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    model.to(torch.float32)
    normalized_config = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)

    class MistralDummyTextInputGenerator(DummyTextInputGenerator):
        SUPPORTED_INPUT_NAMES = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
        }

        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            input = super().generate(input_name, framework, int_dtype, float_dtype)
            if input_name == "position_ids":
                input = input[:, -1:]
            return input

    class MistralDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
        def __init__(
            self,
            task: str,
            normalized_config: NormalizedTextConfig,
            batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
            sequence_length: int = DEFAULT_DUMMY_SHAPES["sequence_length"],
            random_batch_size_range: Optional[Tuple[int, int]] = None,
            random_sequence_length_range: Optional[Tuple[int, int]] = None,
            **kwargs,
        ):
            super().__init__(
                task=task,
                normalized_config=normalized_config,
                batch_size=batch_size,
                sequence_length=sequence_length,
                random_batch_size_range=random_batch_size_range,
                random_sequence_length_range=random_sequence_length_range,
            )
            self.num_key_value_heads = normalized_config.num_key_value_heads

        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            shape = (
                self.batch_size,
                self.num_key_value_heads,
                self.sequence_length,
                self.hidden_size // self.num_attention_heads,
            )
            return [
                (
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]

    class MistralOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
        # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
        DEFAULT_ONNX_OPSET = 14
        DUMMY_INPUT_GENERATOR_CLASSES = (
            MistralDummyTextInputGenerator,
            MistralDummyPastKeyValuesGenerator,
        )
        DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
        NORMALIZED_CONFIG_CLASS = normalized_config
        no_position_ids = False

    export_config = MistralOnnxConfig
    TasksManager._SUPPORTED_MODEL_TYPE[model.config.model_type] = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
        'openvino': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
    }
    NormalizedConfigManager._conf[model.config.model_type] = normalized_config
    convert_optimum_causallm_base(model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


def convert_yi(args):
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    cuda, post_init = patch_gptq(config)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    model.to(torch.float32)
    normalized_config = NormalizedTextConfig

    class YIDummyTextInputGenerator(DummyTextInputGenerator):
        SUPPORTED_INPUT_NAMES = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
        }

        def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
            input = super().generate(input_name, framework, int_dtype, float_dtype)
            if input_name == "position_ids":
                input = input[:, -1:]
            return input

    class YIOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
        # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
        DEFAULT_ONNX_OPSET = 14
        DUMMY_INPUT_GENERATOR_CLASSES = (
            YIDummyTextInputGenerator,
            DummyPastKeyValuesGenerator,
        )
        DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator
        NORMALIZED_CONFIG_CLASS = normalized_config
        no_position_ids = False

    export_config = YIOnnxConfig
    tasks = {
        'onnx': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
        'openvino': {
            'text-generation': make_backend_config_constructor_for_task(export_config, 'text-generation'),
            'text-generation-with-past': make_backend_config_constructor_for_task(export_config, 'text-generation-with-past'),
        },
    }
    TasksManager._SUPPORTED_MODEL_TYPE[model.config.model_type] = tasks
    TasksManager._SUPPORTED_MODEL_TYPE[model.config.model_type.lower()] = tasks
    NormalizedConfigManager._conf[model.config.model_type] = normalized_config
    NormalizedConfigManager._conf[model.config.model_type.lower()] = normalized_config
    convert_optimum_causallm_base(model, args)
    if post_init is not None:
        unpatch_gptq(cuda, post_init)


converters = {
    'decoder': convert_causal_lm,
    'blenderbot': convert_seq2seq,
    't5': convert_seq2seq,
    'stable-diffusion-xl': convert_sdxl,
    'ssd-1b': convert_sdxl,
    'sdxl': convert_sdxl,
    'stable-diffusion': convert_sd,
    'tiny-sd': convert_sd,
    'small-sd': convert_sd,
    'lcm': convert_lcm,
    'ldm': convert_ldm_super_res,
    'mpt': convert_mpt,
    'replit': convert_mpt,
    'chatglm2': convert_chatglm2,
    'chatglm3': convert_chatglm2,
    'chatglm': convert_chatglm,
    'falcon': convert_falcon,
    'stablelm': convert_stablelm,
    'stable-zephyr': convert_stablelm,
    'rocket-': convert_stablelm,
    'jais': convert_jais,
    'baichuan': convert_baichaun,
    'qwen': convert_qwen,
    'mistal': convert_mistral,
    'zephyr': convert_mistral,
    'openchat': convert_mistral,
    'neural-chat': convert_mistral,
    'yi': convert_yi,
}


def get_convert_model_type(model_id):
    default = 'decoder'
    for key in converters:
        if key in model_id:
            return key

    return default


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    parser = ArgumentParser()
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--save_orig', action='store_true')
    parser.add_argument('--precision', choices=['FP32', 'FP16'], default='FP32')
    parser.add_argument('--bettertransformer', action='store_true',
                        help='Apply bettertransformer to enable ScaledDotProductAttention operation for a part of the models')

    compression_group = parser.add_argument_group('Weights compression parameters')
    compression_group.add_argument(
        '-c',
        '--compress_weights',
        type=str,
        choices=['INT8', '4BIT_DEFAULT', 'INT4_SYM', 'INT4_ASYM'],
        nargs='+',
        help=(
            'The weight compression option, e.g. INT8 - INT8 weights, '
            '4BIT_DEFAULT - for 4-bit compression with predefined configs, '
            'INT4_* - for INT4 compressed weights.'
        ),
    )
    compression_group.add_argument(
        '--compress_weights_backends',
        help='Backend names used to compress the input model weights separated by space.',
        choices=[BackendType.PYTORCH.value, BackendType.OPENVINO.value],
        default=BackendType.OPENVINO.value,
        type=str.lower,
        nargs='+',
    )
    compression_group.add_argument(
        '--ratio',
        help='Compression ratio between primary and backup precision, e.g. INT4/INT8',
        default=None,
        type=float,
    )
    compression_group.add_argument(
        '--group_size',
        help='Size of the group of weights that share the same quantization parameters',
        default=None,
        type=int,
    )

    args = parser.parse_args()
    model_type = get_convert_model_type(args.model_id.lower())
    converter = converters[model_type]
    converter(args)


main()
