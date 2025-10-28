# -*- coding: utf-8 -*-
# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from openvino import Core
import openvino as ov
import logging as log
import time
import json
import copy
import types
from llm_bench_utils.hook_common import get_bench_hook
from llm_bench_utils.memory_monitor import MemMonitorWrapper
from llm_bench_utils.hook_forward import MeanStdPair, RawImGenPerfMetrics
from llm_bench_utils.model_utils import get_version_in_format_to_pars
from llm_bench_utils.config_class import (
    UseCaseSpeech2Text,
    UseCaseTextGen,
    PA_ATTENTION_BACKEND
)
from transformers import pipeline
import queue
from transformers.generation.streamers import BaseStreamer


def build_ov_tokenizer(hf_tokenizer):
    try:
        from openvino_tokenizers import convert_tokenizer
    except ImportError:
        log.warn("OV Tokenizer is unavailable, tokenizer conversion will be skipped")
        return hf_tokenizer

    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    return build_ov_tokenizer_wrapper(hf_tokenizer, ov_tokenizer, ov_detokenizer)


def build_ov_tokenizer_wrapper(hf_tokenizer, tokenizer_model, detokenizer_model):
    ov_compiled_tokenizer = ov.compile_model(tokenizer_model, "CPU")
    ov_compiled_detokenizer = ov.compile_model(detokenizer_model, "CPU")

    def encode_ov_tokenizer_full(self, text, *args, **kwargs):
        if isinstance(text, str):
            text = [text]
        return ov_compiled_tokenizer(text)

    def encode_ov_tokenizer(self, text, *args, **kwargs):
        results = encode_ov_tokenizer_full(self, text, *args, **kwargs)
        return results["input_ids"].squeeze(0).tolist()

    def batch_decode_ov_tokenizer(self, sequences, *args, **kwargs):
        result = list(ov_compiled_detokenizer(sequences)["string_output"])
        return result

    def decode_ov_tokenizer(self, token_ids, *args, **kwargs):
        return self.batch_decode([token_ids])[0]

    hf_tokenizer.encode = types.MethodType(encode_ov_tokenizer, hf_tokenizer)
    hf_tokenizer.__call__ = types.MethodType(encode_ov_tokenizer_full, hf_tokenizer)
    hf_tokenizer.batch_decode = types.MethodType(batch_decode_ov_tokenizer, hf_tokenizer)
    hf_tokenizer.decode = types.MethodType(decode_ov_tokenizer, hf_tokenizer)
    return hf_tokenizer


def get_lora_config(lora_paths, lora_alphas, lora_mode=None):
    import openvino_genai

    modes = {
        "auto": openvino_genai.AdapterConfig.Mode.MODE_AUTO,
        "fuse": openvino_genai.AdapterConfig.Mode.MODE_FUSE,
        "dynamic": openvino_genai.AdapterConfig.Mode.MODE_DYNAMIC,
        "static": openvino_genai.AdapterConfig.Mode.MODE_STATIC,
        "static_rank": openvino_genai.AdapterConfig.Mode.MODE_DYNAMIC
    }
    if lora_mode is not None:
        lora_mode = modes[lora_mode]
        log.info(f"LoRA adapters loading mode: {lora_mode}")

    adapter_config = list()
    if not lora_paths:
        return adapter_config

    if len(lora_paths) != len(lora_alphas):
        log.warning('Amount of provided LoRA paths and alphas is not eq. LoRA will be ignored.')
        return adapter_config

    for idx in range(len(lora_paths)):
        if not Path(lora_paths[idx]).exists():
            log.warning(f'LoRA path is not exists: {lora_paths[idx]}. LoRA will be ignored.')
            continue
        adapter_config = openvino_genai.AdapterConfig() if lora_mode is None else openvino_genai.AdapterConfig(mode=lora_mode)
        adapter = openvino_genai.Adapter(lora_paths[idx])
        alpha = float(lora_alphas[idx])
        adapter_config.add(adapter, alpha)

    if adapter_config:
        log.info('LoRA adapter(s) are added to config.')

    return adapter_config


def create_text_gen_model(model_path, device, memory_data_collector, **kwargs):
    """Create text generation model.

    - model_path: can be model_path or IR path
    - device: can be CPU or GPU
    - model_type:
    """
    use_case = kwargs['use_case']
    model_class = use_case.ov_cls
    token_class = use_case.tokenizer_cls
    model_path = Path(model_path)
    # specify the model path
    if model_path.name.endswith('xml'):
        model_path = model_path.parents[2]

    ov_config = kwargs['config']

    model_path_existed = Path(model_path).exists()
    # load model
    if not model_path_existed:
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        if kwargs.get("genai", True) and is_genai_available(log_msg=True):
            if model_class != UseCaseTextGen.ov_cls and "mpt" not in use_case.model_types and "chatglm" not in use_case.model_types:
                log.warning("OpenVINO GenAI based benchmarking is not available for required model type. Will be switched to default benchmarking")
            else:
                log.info("Selected OpenVINO GenAI for benchmarking")
                return create_genai_text_gen_model(model_path, device, ov_config, memory_data_collector, **kwargs)

        log.info("Selected Optimum Intel for benchmarking")
        ov_config.pop("ATTENTION_BACKEND", None)
        remote_code = False
        try:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        except Exception:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            remote_code = True

        if kwargs.get("mem_consumption"):
            memory_data_collector.start()
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(
            model_path,
            device=device,
            ov_config=ov_config,
            config=model_config,
            stateful=kwargs.get("stateful", None),
            trust_remote_code=remote_code,
            from_onnx=kwargs.get("from_onnx", False)
        )
        end = time.perf_counter()
        if kwargs.get("mem_consumption"):
            memory_data_collector.stop_and_collect_data('compilation_phase')
            memory_data_collector.log_data(compilation_phase=True)
    bench_hook = get_bench_hook(kwargs['num_beams'], ov_model)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    # load token
    tokenizer = token_class.from_pretrained(model_path, trust_remote_code=True)
    if kwargs.get("convert_tokenizer", False):
        tokenizer = build_ov_tokenizer(tokenizer)
    return ov_model, tokenizer, from_pretrained_time, bench_hook, False


def get_scheduler_config_genai(config_data, config_name="CB config"):
    import openvino_genai
    user_config = copy.deepcopy(config_data)

    scheduler_config = openvino_genai.SchedulerConfig()
    if user_config:
        log.info(f"Scheduler parameters for {config_name}:\n{user_config}")

        if 'cache_eviction_config' in user_config.keys():
            cache_eviction_kwargs = user_config.pop('cache_eviction_config')
            if "aggregation_mode" in cache_eviction_kwargs.keys():
                cache_eviction_kwargs["aggregation_mode"] = getattr(openvino_genai.AggregationMode, cache_eviction_kwargs["aggregation_mode"])

            if "kvcrush_config" in cache_eviction_kwargs.keys() and isinstance(cache_eviction_kwargs["kvcrush_config"], dict):
                crush_config_kwargs = cache_eviction_kwargs["kvcrush_config"]
                if "anchor_point_mode" in crush_config_kwargs.keys():
                    crush_config_kwargs['anchor_point_mode'] = getattr(openvino_genai.KVCrushAnchorPointMode, crush_config_kwargs['anchor_point_mode'])
                cache_eviction_kwargs["kvcrush_config"] = openvino_genai.KVCrushConfig(**crush_config_kwargs)

            scheduler_config.use_cache_eviction = True
            scheduler_config.cache_eviction_config = openvino_genai.CacheEvictionConfig(**cache_eviction_kwargs)
            log.info("Cache Eviction mode ON")

        if 'sparse_attention_config' in user_config.keys():
            sparse_attention_kwargs = user_config.pop('sparse_attention_config')
            if "mode" in sparse_attention_kwargs.keys():
                sparse_attention_kwargs["mode"] = getattr(openvino_genai.SparseAttentionMode, sparse_attention_kwargs["mode"])

            scheduler_config.use_sparse_attention = True
            scheduler_config.sparse_attention_config = openvino_genai.SparseAttentionConfig(**sparse_attention_kwargs)
            log.info("Sparse Attention mode ON")

        for param, value in user_config.items():
            setattr(scheduler_config, param, value)

    return scheduler_config


def cb_pipeline_required(args):
    return args['config'].get("ATTENTION_BACKEND", PA_ATTENTION_BACKEND) == PA_ATTENTION_BACKEND and args.get("cb_config") is not None and\
        (args["cb_config"].get("cache_eviction_config") is not None or args["cb_config"].get("sparse_attention_config") is not None)


def create_genai_text_gen_model(model_path, device, ov_config, memory_data_collector, **kwargs):
    import openvino_genai
    from packaging.version import parse

    if Path(model_path).suffix != '.gguf'\
       and (not (model_path / "openvino_tokenizer.xml").exists() or not (model_path / "openvino_detokenizer.xml").exists()):
        raise ValueError("OpenVINO Tokenizer model is not found in model directory. Please convert tokenizer using following command:\n"
                         "convert_tokenizer --with-detokenizer MODEL_DIR --output MODEL_DIR ")

    config = {}
    draft_model_path = kwargs.get("draft_model", '')
    cb_config = kwargs.get("cb_config")
    use_streamer_metrics = False
    if cb_config is not None:
        config["scheduler_config"] = get_scheduler_config_genai(cb_config)
        version = get_version_in_format_to_pars(openvino_genai.get_version())
        use_streamer_metrics = parse(version) < parse("2025.0.0") or (draft_model_path and parse(version) < parse("2025.1.0"))

    if draft_model_path:
        if not Path(draft_model_path).exists():
            raise RuntimeError(f'==Failure ==: draft model by path:{draft_model_path} is not exists')
        log.info("Speculative Decoding is activated")
        draft_device = kwargs.get('draft_device', None) or device
        draft_model_load_kwargs = {'scheduler_config': get_scheduler_config_genai(kwargs.get("draft_cb_config"), config_name="draft CB config")}\
            if kwargs.get("draft_cb_config") is not None else {}
        config['draft_model'] = openvino_genai.draft_model(draft_model_path, draft_device.upper(), **draft_model_load_kwargs)

    if kwargs.get('max_ngram_size') and kwargs.get('num_assistant_tokens'):
        log.info("Prompt Lookup decoding is activated")
        config['prompt_lookup'] = True
        use_streamer_metrics = True

    adapter_config = get_lora_config(kwargs.get("lora", None), kwargs.get("lora_alphas", []), kwargs.get("lora_mode", None))
    if adapter_config:
        config['adapters'] = adapter_config

    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    if cb_pipeline_required(kwargs):
        ov_config.pop("ATTENTION_BACKEND", None)
        log.info("Pipeline will be initialized with ContinuousBatchingPipeline")
        llm_pipe = openvino_genai.ContinuousBatchingPipeline(model_path, device=device.upper(), properties=ov_config, **config)
    else:
        llm_pipe = openvino_genai.LLMPipeline(model_path, device.upper(), **config, **ov_config)
    end = time.perf_counter()
    log.info(f'Pipeline initialization time: {end - start:.2f}s')
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)

    class TokenStreamer(openvino_genai.StreamerBase):
        def __init__(self, tokenizer):
            openvino_genai.StreamerBase.__init__(self)
            self.tokenizer = tokenizer
            self.token_generation_time = []
            self.generated_tokens = []
            self.start_time = time.perf_counter()

        def put(self, token_id):
            self.token_generation_time.append(time.perf_counter() - self.start_time)
            self.generated_tokens.append(token_id)
            self.start_time = time.perf_counter()
            return False

        def reset(self):
            self.token_generation_time = []
            self.generated_tokens = []
            self.start_time = time.perf_counter()

        def end(self):
            pass

        def get_tokens(self):
            return self.generated_tokens

        def get_time_list(self):
            return self.token_generation_time
    streamer = TokenStreamer(llm_pipe.get_tokenizer()) if use_streamer_metrics else None

    return llm_pipe, None, end - start, streamer, True


def convert_ov_tokenizer(tokenizer_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    export_tokenizer(hf_tokenizer, tokenizer_path)


def is_inpainting_model(model_args, use_case, model_index_data):
    is_inpainting_by_class_name = "Inpaint" in model_index_data.get("_class_name", "")
    all_inputs_for_inpainting_set = ((model_args.get("media") or model_args.get("images")) and model_args.get("mask_image"))
    return (model_args.get("task", "") == use_case.TASK["inpainting"]["name"] or all_inputs_for_inpainting_set or is_inpainting_by_class_name)


def is_image_to_image_model(model_args, use_case):
    all_inputs_for_im2im_set = (model_args.get("media") or model_args.get("images"))
    return model_args.get("task", "") == use_case.TASK["img2img"]["name"] or all_inputs_for_im2im_set


def is_text_to_image_model(model_args, use_case):
    # default case, if task is not set, this pipeline will be setup
    return model_args.get("task", "") == use_case.TASK["text2img"]["name"] or not model_args.get("task")


def create_image_gen_model(model_path, device, memory_data_collector, **kwargs):
    model_index_data = {}
    with open(str(model_path / "model_index.json"), 'r') as f:
        model_index_data = json.load(f)

    image_gen_use_case = kwargs['use_case']
    model_class = image_gen_use_case.TASK["text2img"]["ov_cls"]
    if is_inpainting_model(kwargs, image_gen_use_case, model_index_data):
        model_class = image_gen_use_case.TASK["inpainting"]["ov_cls"]
    elif is_image_to_image_model(kwargs, image_gen_use_case):
        model_class = image_gen_use_case.TASK["img2img"]["ov_cls"]

    model_path = Path(model_path)
    ov_config = kwargs['config']
    if not Path(model_path).exists():
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        if kwargs.get("genai", True) and is_genai_available(log_msg=True):
            log.info("Selected OpenVINO GenAI for benchmarking")
            return create_genai_image_gen_model(model_path, device, ov_config, model_index_data, memory_data_collector, **kwargs)

        if kwargs.get("mem_consumption"):
            memory_data_collector.start()
        log.info("Selected Optimum Intel for benchmarking")
        start = time.perf_counter()
        if kwargs.get("static_reshape", False):
            ov_model = model_class.from_pretrained(model_path, device=device, ov_config=ov_config, compile=False)
            num_images_per_prompt = kwargs.get("batch_size", 1)
            height = kwargs.get("height", 512)
            width = kwargs.get("width", 512)
            log.info(f"Image Pipeline reshape(batch_size=1, height={height}, width={width}, num_images_per_prompt={num_images_per_prompt})")
            ov_model.reshape(batch_size=1, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
            ov_model.compile()
        else:
            ov_model = model_class.from_pretrained(model_path, device=device, ov_config=ov_config)
        end = time.perf_counter()
        if kwargs.get("mem_consumption"):
            memory_data_collector.stop_and_collect_data('compilation_phase')
            memory_data_collector.log_data(compilation_phase=True)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    return ov_model, from_pretrained_time, False, None


def get_genai_clip_text_encoder(model_index_data, model_path, device, ov_config):
    import openvino_genai
    text_encoder_type = model_index_data.get("text_encoder", [])
    if ("CLIPTextModel" in text_encoder_type):
        text_encoder = openvino_genai.CLIPTextModel(model_path / "text_encoder", device.upper(), **ov_config)
    else:
        raise RuntimeError(f'==Failure ==: model by path:{model_path} has unsupported text encoder type {text_encoder_type}')

    return text_encoder


def get_genai_clip_text_encoder_with_projection(model_index_data, model_path, text_encoder_path, device, ov_config):
    import openvino_genai
    text_encoder_type = model_index_data.get(text_encoder_path, [])
    if ("CLIPTextModelWithProjection" in text_encoder_type):
        text_encoder = openvino_genai.CLIPTextModelWithProjection(model_path / text_encoder_path, device.upper(), **ov_config)
    else:
        raise RuntimeError(f'==Failure ==: model by path:{model_path} has unsupported {text_encoder_path} type {text_encoder_type}')

    return text_encoder


def get_genai_unet_model(model_index_data, model_path, device, ov_config):
    import openvino_genai
    unet_type = model_index_data.get("unet", [])
    if ("UNet2DConditionModel" in unet_type):
        unet = openvino_genai.UNet2DConditionModel(model_path / "unet", device.upper(), **ov_config)
    else:
        raise RuntimeError(f'==Failure ==: model by path:{model_path} has unsupported UNet type {unet_type}')

    return unet


def create_genai_image_gen_model(model_path, device, ov_config, model_index_data, memory_data_collector, **kwargs):
    import openvino_genai

    class PerfCollector:
        def __init__(self, main_model_name="unet") -> types.NoneType:
            self.iteration_time = []
            self.start_time = time.perf_counter()
            self.duration = -1
            self.main_model_name = main_model_name

        def __call__(self, step, num_steps, latents):
            self.iteration_time.append(time.perf_counter() - self.start_time)
            self.start_time = time.perf_counter()
            return False

        def reset(self):
            self.iteration_time = []
            self.start_time = time.perf_counter()
            self.duration = -1

        def get_1st_unet_latency(self):
            return self.iteration_time[0] * 1000 if len(self.iteration_time) > 0 else 0

        def get_2nd_unet_latency(self):
            return sum(self.iteration_time[1:]) / (len(self.iteration_time) - 1) * 1000 if len(self.iteration_time) > 1 else 0

        def get_first_and_other_unet_infer_duration(self):
            first = self.get_1st_unet_latency()
            other = self.get_2nd_unet_latency()
            return (first, other)

        def get_first_and_other_trans_infer_duration(self):
            return self.get_first_and_other_unet_infer_duration()

        def get_text_encoder_infer_duration(self):
            return {}

        def get_unet_infer_duration(self):
            mean = (sum(self.iteration_time) / len(self.iteration_time)) * 1000 if len(self.iteration_time) > 0 else 0
            return MeanStdPair(mean=mean)

        def get_vae_decoder_infer_duration(self):
            if self.duration != -1:
                vae_time = self.duration - sum(self.iteration_time)
                return vae_time * 1000
            return 0

        @property
        def raw_metrics(self):
            return RawImGenPerfMetrics(self.iteration_time)

    image_gen_use_case = kwargs['use_case']
    image_gen_pipeline_class = openvino_genai.Text2ImagePipeline
    if is_inpainting_model(kwargs, image_gen_use_case, model_index_data):
        log.info("Selected Inpainting image generation pipeline")
        image_gen_pipeline_class = openvino_genai.InpaintingPipeline
    elif is_image_to_image_model(kwargs, image_gen_use_case):
        log.info("Selected Image to Image image generation pipeline")
        image_gen_pipeline_class = openvino_genai.Image2ImagePipeline
    elif is_text_to_image_model(kwargs, image_gen_use_case):
        log.info("Selected Text to Image image generation pipeline")
    else:
        log.warning(f'Task {kwargs.get("task")} is not defined. Text to Image image generation pipeline will be used.')

    adapter_config = get_lora_config(kwargs.get("lora", None), kwargs.get("lora_alphas", []), kwargs.get("lora_mode", None))
    if adapter_config:
        ov_config['adapters'] = adapter_config

    model_class_name = model_index_data.get("_class_name", "")
    main_model_name = "unet" if "unet" in model_index_data else "transformer"
    callback = PerfCollector(main_model_name)

    orig_tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    callback.orig_tokenizer = orig_tokenizer

    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()

    scheduler_type = model_index_data.get("scheduler", ["", ""])[1]
    if (scheduler_type not in ["LCMScheduler", "DDIMScheduler", "PNDMScheduler", "EulerDiscreteScheduler",
                               "FlowMatchEulerDiscreteScheduler", "EulerAncestralDiscreteScheduler"]):
        # It's possible we could support --static_reshape here, but initially it seems too complicated to be worth it..
        # (as we'd need to refactor each get_*_model calls below to perform explicit reshape + compile)
        if kwargs.get("static_reshape", False):
            raise RuntimeError(f'Type of scheduler {scheduler_type} is unsupported. Right now this is unsupported if --static_reshape is also specified. ')

        scheduler = openvino_genai.Scheduler.from_config(model_path / "scheduler/scheduler_config.json", openvino_genai.Scheduler.Type.DDIM)
        log.warning(f'Type of scheduler {scheduler_type} is unsupported. Please, be aware that it will be replaced to DDIMScheduler')

        vae_type = model_index_data.get("vae", [])
        if ("AutoencoderKL" in vae_type):
            vae = openvino_genai.AutoencoderKL(model_path / "vae_decoder", device.upper(), **ov_config)
        else:
            raise RuntimeError(f'==Failure ==: model by path:{model_path} has unsupported vae decoder type {vae_type}')

        if model_class_name == "StableDiffusionPipeline":
            text_encoder = get_genai_clip_text_encoder(model_index_data, model_path, device, ov_config)
            unet = get_genai_unet_model(model_index_data, model_path, device, ov_config)
            image_gen_pipe = image_gen_pipeline_class(model_path, device.upper(), **ov_config)
        elif model_class_name == "LatentConsistencyModelPipeline":
            text_encoder = get_genai_clip_text_encoder(model_index_data, model_path, device, ov_config)
            unet = get_genai_unet_model(model_index_data, model_path, device, ov_config)
            image_gen_pipe = image_gen_pipeline_class.latent_consistency_model(scheduler, text_encoder, unet, vae)
        elif model_class_name == "StableDiffusionXLPipeline":
            clip_text_encoder = get_genai_clip_text_encoder(model_index_data, model_path, device, ov_config)
            clip_text_encoder_2 = get_genai_clip_text_encoder_with_projection(model_index_data, model_path, "text_encoder_2", device, ov_config)
            unet = get_genai_unet_model(model_index_data, model_path, device, ov_config)
            image_gen_pipe = image_gen_pipeline_class.stable_diffusion_xl(scheduler, clip_text_encoder, clip_text_encoder_2, unet, vae)
        else:
            raise RuntimeError(f'==Failure ==: model by path:{model_path} has unsupported _class_name {model_class_name}')
    else:
        if kwargs.get("static_reshape", False):
            image_gen_pipe = image_gen_pipeline_class(model_path)
            guidance_scale = kwargs.get("guidance_scale", image_gen_pipe.get_generation_config().guidance_scale)
            num_images_per_prompt = kwargs.get("batch_size", 1)
            height = kwargs.get("height", 512)
            width = kwargs.get("width", 512)
            log.info(f"Image Pipeline reshape(num_images_per_prompt={num_images_per_prompt}, height={height}, width={width}, guidance_scale={guidance_scale})")
            image_gen_pipe.reshape(num_images_per_prompt=num_images_per_prompt, height=height, width=width, guidance_scale=guidance_scale)
            image_gen_pipe.compile(device.upper(), **ov_config)
        else:
            image_gen_pipe = image_gen_pipeline_class(model_path, device.upper(), **ov_config)

    end = time.perf_counter()
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)
    log.info(f'Pipeline initialization time: {end - start:.2f}s')
    return image_gen_pipe, end - start, True, callback


def create_ldm_super_resolution_model(model_path, device, memory_data_collector, **kwargs):
    core = Core()
    ov_config = kwargs['config']
    core.set_property(ov_config)
    model_class = kwargs['use_case'].ov_cls
    model_path = Path(model_path)
    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    ov_model = model_class(model_path, core, device.upper())
    end = time.perf_counter()
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    return ov_model, from_pretrained_time


def create_genai_speech_2_txt_model(model_path, device, memory_data_collector, **kwargs):
    import openvino_genai as ov_genai
    if kwargs.get("genai", True) is False:
        raise RuntimeError('==Failure the command line does not set --genai ==')
    if is_genai_available(log_msg=True) is False:
        raise RuntimeError('==Failure genai is not enable ==')
    ov_config = kwargs['config']
    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    genai_pipe = ov_genai.WhisperPipeline(model_path, device.upper(), **ov_config)
    end = time.perf_counter()
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    processor = AutoProcessor.from_pretrained(model_path)
    return genai_pipe, processor, from_pretrained_time, True


def create_speech_2_txt_model(model_path, device, memory_data_collector, **kwargs):
    """Create speech generation model.
    - model_path: can be model_path or IR path
    - device: can be CPU
    - model_type:
    """
    from optimum.intel.utils.import_utils import is_transformers_version

    use_case = kwargs['use_case']
    model_class = use_case.ov_cls
    model_path = Path(model_path)
    model_path_existed = model_path.exists()
    # load model
    if not model_path_existed:
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        if kwargs.get("genai", True) and is_genai_available(log_msg=True):
            if model_class not in [UseCaseSpeech2Text.ov_cls]:
                log.warning("OpenVINO GenAI based benchmarking is not available for required model type. Will be switched to default benchmarking")
            else:
                log.info("Selected OpenVINO GenAI for benchmarking")
                return create_genai_speech_2_txt_model(model_path, device, memory_data_collector, **kwargs)
        log.info("Selected Optimum Intel for benchmarking")
        ov_config = kwargs['config']
        if kwargs.get("mem_consumption"):
            memory_data_collector.start()
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(
            model_path,
            device=device,
            ov_config=ov_config
        )
        end = time.perf_counter()
        if kwargs.get("mem_consumption"):
            memory_data_collector.stop_and_collect_data('compilation_phase')
            memory_data_collector.log_data(compilation_phase=True)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    if is_transformers_version(">=", "4.51.0"):
        ov_model.config.forced_decoder_ids = None

        if hasattr(ov_model, 'generation_config'):
            if hasattr(ov_model.generation_config, 'forced_decoder_ids'):
                ov_model.generation_config.forced_decoder_ids = None

    processor = AutoProcessor.from_pretrained(model_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=ov_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
    )

    return pipe, processor, from_pretrained_time, False


def get_vlm_processor(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = config.model_type
    if model_type == "llava-qwen2":
        processor = AutoProcessor.from_pretrained(config.mm_vision_tower, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        preprocessors = {"processor": processor, "tokenizer": tokenizer, "config": config}
    elif model_type == "internvl_chat":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        preprocessors = {"processor": None, "tokenizer": tokenizer, "config": config}
    else:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        preprocessors = {"processor": processor, "tokenizer": processor, "config": config}
    return preprocessors


def create_genai_image_text_gen_model(model_path, device, ov_config, memory_data_collector, **kwargs):
    import openvino_genai

    if not (model_path / "openvino_tokenizer.xml").exists() or not (model_path / "openvino_detokenizer.xml").exists():
        convert_ov_tokenizer(model_path)

    cb_config = kwargs.get("cb_config")
    if cb_config is not None:
        ov_config["scheduler_config"] = get_scheduler_config_genai(cb_config)

    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    llm_pipe = openvino_genai.VLMPipeline(model_path, device.upper(), **ov_config)
    end = time.perf_counter()
    log.info("Selected OpenVINO GenAI for benchmarking")
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)
    log.info(f'Pipeline initialization time: {end - start:.2f}s')

    return llm_pipe, None, end - start, None, True


def create_genai_text_embed_model(model_path, device, memory_data_collector, **kwargs):
    import openvino_genai

    pooling_type = kwargs.get("emb_pooling_type")
    max_length = kwargs.get("emb_max_length")
    padding_side = kwargs.get("embedding_padding_side")

    config = openvino_genai.TextEmbeddingPipeline.Config()
    if pooling_type is not None:
        if pooling_type == "mean":
            config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.MEAN
        elif pooling_type == "last_token":
            config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN
        else:
            config.pooling_type = openvino_genai.TextEmbeddingPipeline.PoolingType.CLS
    if max_length is not None:
        config.max_length = max_length
        config.pad_to_max_length = True
    config.normalize = kwargs.get("emb_normalize", False)
    if padding_side:
        config.padding_side = padding_side

    ov_config = kwargs['config']

    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()

    pipe = openvino_genai.TextEmbeddingPipeline(model_path, device.upper(), config, **ov_config)

    end = time.perf_counter()

    log.info("Selected OpenVINO GenAI for benchmarking")
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)
    log.info(f'Pipeline initialization time: {end - start:.2f}s')
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return pipe, tokenizer, end - start, None, True


def create_text_embeddings_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    if model_path.name.endswith('xml'):
        model_path = model_path.parents[2]

    ov_config = kwargs['config']

    model_path_existed = Path(model_path).exists()
    # load model
    if not model_path_existed:
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')

    trust_remote_code = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        trust_remote_code = True
    if kwargs.get("genai", True) and is_genai_available(log_msg=True):
        try:
            return create_genai_text_embed_model(model_path, device, memory_data_collector, **kwargs)
        except Exception as exp:
            log.warning(
                f"Model is not supported by OpenVINO GenAI. "
                f"GenAI pipeline loading failed with following error: {exp}"
                "Benchmark will be switched to Optimum Intel pipeline realization"
            )

        log.info("Selected Optimum Intel for benchmarking")
    model_class = kwargs['use_case'].ov_cls
    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    ov_model = model_class.from_pretrained(
        model_path,
        device=device,
        ov_config=ov_config,
        trust_remote_code=trust_remote_code
    )
    end = time.perf_counter()
    pooling_type = kwargs.get("emb_pooling_type") or "cls"
    normalize = kwargs.get("emb_normalize", False)

    ov_model._embed_forward = ov_model.forward

    def forward_with_pooling(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        import torch
        outputs = self._embed_forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        token_embeddings = outputs.last_hidden_state
        if pooling_type == "cls":
            out_embd = token_embeddings[:, 0]
        elif pooling_type == 'last_token':
            # from transformers Qwen3-Embedding-0.6B model card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                out_embd = token_embeddings[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = token_embeddings.shape[0]
                batch_dim = torch.arange(batch_size, device=token_embeddings.device)
                out_embd = token_embeddings[batch_dim, sequence_lengths]
        else:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            out_embd = sum_embeddings / sum_mask

        if normalize:
            out_embd = torch.nn.functional.normalize(out_embd, p=2, dim=1)

        return out_embd

    ov_model.forward = types.MethodType(forward_with_pooling, ov_model)

    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)
    bench_hook = get_bench_hook(1, ov_model, rag=True)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    return ov_model, tokenizer, from_pretrained_time, bench_hook, False


def create_image_text_gen_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    # specify the model path
    if model_path.name.endswith('xml'):
        model_path = model_path.parents[2]

    ov_config = kwargs['config']

    model_path_existed = Path(model_path).exists()
    # load model
    if not model_path_existed:
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        remote_code = False
        try:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        except Exception:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            remote_code = True
        if kwargs.get("genai", True) and is_genai_available(log_msg=True):
            try:
                return create_genai_image_text_gen_model(model_path, device, ov_config, memory_data_collector, **kwargs)
            except Exception as exp:
                log.warning(
                    f"Model type `{model_config.model_type}` is not supported by OpenVINO GenAI. "
                    f"GenAI pipeline loading failed with following error: {exp}"
                    "Benchmark will be switched to Optimum Intel pipeline realization"
                )

        log.info("Selected Optimum Intel for benchmarking")
        ov_config.pop("ATTENTION_BACKEND", None)
        model_class = kwargs['use_case'].ov_cls
        if kwargs.get("mem_consumption"):
            memory_data_collector.start()
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(
            model_path,
            device=device,
            ov_config=ov_config,
            config=model_config,
            trust_remote_code=remote_code
        )
        end = time.perf_counter()
        if kwargs.get("mem_consumption"):
            memory_data_collector.stop_and_collect_data('compilation_phase')
            memory_data_collector.log_data(compilation_phase=True)
    bench_hook = get_bench_hook(kwargs['num_beams'], ov_model)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    processor_config = get_vlm_processor(model_path)
    return ov_model, processor_config, from_pretrained_time, bench_hook, False


def create_genai_text_2_speech_model(model_path, device, ov_config, memory_data_collector, **kwargs):
    import openvino_genai

    if not (model_path / "openvino_tokenizer.xml").exists() or not (model_path / "openvino_detokenizer.xml").exists():
        convert_ov_tokenizer(model_path)

    tokenizer_class = kwargs['use_case'].tokenizer_cls
    processor = tokenizer_class.from_pretrained(model_path)

    if kwargs.get("mem_consumption"):
        memory_data_collector.start()
    start = time.perf_counter()
    pipe = openvino_genai.Text2SpeechPipeline(model_path, device.upper(), **ov_config)
    end = time.perf_counter()
    log.info("Selected OpenVINO GenAI for benchmarking")
    if kwargs.get("mem_consumption"):
        memory_data_collector.stop_and_collect_data('compilation_phase')
        memory_data_collector.log_data(compilation_phase=True)
    log.info(f'Pipeline initialization time: {end - start:.2f}s')

    return pipe, processor, None, end - start, True


def create_text_2_speech_model(model_path, device, memory_data_collector, **kwargs):
    model_path = Path(model_path)
    # specify the model path
    if model_path.name.endswith('xml'):
        model_path = model_path.parents[2]

    ov_config = kwargs['config']

    model_path_existed = Path(model_path).exists()
    # load model
    if not model_path_existed:
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        remote_code = False
        try:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        except Exception:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            remote_code = True
        if kwargs.get("genai", True) and is_genai_available(log_msg=True):
            try:
                return create_genai_text_2_speech_model(model_path, device, ov_config, memory_data_collector, **kwargs)
            except Exception as exp:
                log.warning(
                    f"Model type `{model_config.model_type}` is not supported by OpenVINO GenAI. "
                    f"GenAI pipeline loading failed with following error: {exp}"
                    "Benchmark will be switched to Optimum Intel pipeline realization"
                )

        log.info("Selected Optimum Intel for benchmarking")
        use_case = kwargs['use_case']
        model_class = use_case.ov_cls
        tokenizer_class = use_case.tokenizer_cls
        if kwargs.get("mem_consumption"):
            memory_data_collector.start()
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(
            model_path,
            device=device,
            ov_config=ov_config,
            config=model_config,
            trust_remote_code=remote_code
        )
        end = time.perf_counter()
        if kwargs.get("mem_consumption"):
            memory_data_collector.stop_and_collect_data('compilation_phase')
            memory_data_collector.log_data(compilation_phase=True)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    processor = tokenizer_class.from_pretrained(model_path)
    vocoder = None
    if kwargs.get('vocoder_path') is not None:
        vocoder = kwargs['use_case'].vocoder_cls.from_pretrained(kwargs.get('vocoder_path'))
    return ov_model, processor, vocoder, from_pretrained_time, False


def is_genai_available(log_msg=False):
    import importlib
    try:
        importlib.import_module('openvino_genai')
    except ImportError as ex:
        if log_msg:
            log.warning("Attempt to load OpenVINO GenaAI package failed. Please install openvino_genai package. Full error message available in debug mode")
            log.warning(ex)
            return False
    return True


def get_genai_chunk_streamer():
    import openvino_genai as ov_genai

    class GenaiChunkStreamer(ov_genai.StreamerBase):
        """
        A custom streamer class for handling token streaming and detokenization with buffering.

        Attributes:
            tokenizer (Tokenizer): The tokenizer used for encoding and decoding tokens.
            tokens_cache (list): A buffer to accumulate tokens for detokenization.
            text_queue (Queue): A synchronized queue for storing decoded text chunks.
            print_len (int): The length of the printed text to manage incremental decoding.
        """

        def __init__(self, tokenizer, tokens_len=1):
            """
            Initializes the IterableStreamer with the given tokenizer.

            Args:
                tokenizer (Tokenizer): The tokenizer to use for encoding and decoding tokens.
            """
            super().__init__()
            self.tokenizer = tokenizer
            self.tokens_cache = []
            self.text_queue = queue.Queue()
            self.print_len = 0
            self.tokens_len = tokens_len

        def __iter__(self):
            """
            Returns the iterator object itself.
            """
            return self

        def __next__(self):
            """
            Returns the next value from the text queue.

            Returns:
                str: The next decoded text chunk.

            Raises:
                StopIteration: If there are no more elements in the queue.
            """
            value = self.text_queue.get()  # get() will be blocked until a token is available.
            if value is None:
                raise StopIteration
            return value

        def get_stop_flag(self):
            """
            Checks whether the generation process should be stopped.

            Returns:
                bool: Always returns False in this implementation.
            """
            return False

        def put_word(self, word: str):
            """
            Puts a word into the text queue.

            Args:
                word (str): The word to put into the queue.
            """
            self.text_queue.put(word)

        def put(self, token_id: int) -> bool:
            """
            Processes a token and manages the decoding buffer. Adds decoded text to the queue.

            Args:
                token_id (int): The token_id to process.

            Returns:
                bool: True if generation should be stopped, False otherwise.
            """
            self.tokens_cache.append(token_id)
            if len(self.tokens_cache) % self.tokens_len == 0:
                text = self.tokenizer.decode(self.tokens_cache)

                word = ''
                if len(text) > self.print_len and '\n' == text[-1]:
                    # Flush the cache after the new line symbol.
                    word = text[self.print_len:]
                    self.tokens_cache = []
                    self.print_len = 0
                elif len(text) >= 3 and text[-1] == chr(65533):
                    # Don't print incomplete text.
                    pass
                elif len(text) > self.print_len:
                    # It is possible to have a shorter text after adding new token.
                    # Print to output only if text lengh is increaesed.
                    word = text[self.print_len:]
                    self.print_len = len(text)
                self.put_word(word)

                if self.get_stop_flag():
                    # When generation is stopped from streamer then end is not called, need to call it here manually.
                    self.end()
                    return True  # True means stop  generation
                else:
                    return False  # False means continue generation
            else:
                return False

        def end(self):
            """
            Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
            """
            text = self.tokenizer.decode(self.tokens_cache)
            if len(text) > self.print_len:
                word = text[self.print_len:]
                self.put_word(word)
                self.tokens_cache = []
                self.print_len = 0
            self.put_word(None)

    return GenaiChunkStreamer


class OptimumChunkStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.
    <Tip warning={true}>
    The API for the streamer classes is still under development and may change in the future.
    </Tip>
    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.
    Examples:
        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
        >>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)
        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False,
                 tokens_len: int = 1, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
        self.tokens_len = tokens_len

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        if len(self.token_cache) % self.tokens_len == 0:
            text = self.tokenizer.decode(
                self.token_cache, **self.decode_kwargs
            )
            # After the symbol for a new line, we flush the cache.
            if text.endswith("\n"):
                printable_text = text[self.print_len:]
                self.token_cache = []
                self.print_len = 0
            # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len:]
                self.print_len += len(printable_text)
            # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
            # which may change with the subsequent token -- there are probably smarter ways to do this!)
            else:
                printable_text = text[self.print_len: text.rfind(" ") + 1]
                self.print_len += len(printable_text)
            self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(
                self.token_cache, **self.decode_kwargs
            )
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""
        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(text, flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True
        return False


def create_genai_text_reranker_model(model_path: Path, device: str, memory_monitor: MemMonitorWrapper, tokenizer: AutoTokenizer, **kwargs):
    import openvino_genai

    config = openvino_genai.TextRerankPipeline.Config()
    if kwargs.get("rerank_top_n") is not None:
        config.top_n = kwargs.get("rerank_top_n")
    if kwargs.get("rerank_max_length") is not None:
        config.max_length = kwargs.get("rerank_max_length")

    ov_config = kwargs['config']

    if kwargs.get("mem_consumption"):
        memory_monitor.start()
    start = time.perf_counter()
    pipe = openvino_genai.TextRerankPipeline(model_path, device.upper(), config, **ov_config)
    end = time.perf_counter()

    log.info("Selected OpenVINO GenAI for benchmarking")
    if kwargs.get("mem_consumption"):
        memory_monitor.stop_and_collect_data('compilation_phase')
        memory_monitor.log_data('for compilation phase')
    log.info(f'Pipeline initialization time: {end - start:.2f}s')
    return pipe, tokenizer, end - start, None, True


def create_text_reranker_model(model_path: Path, device: str, memory_monitor: MemMonitorWrapper, **kwargs):
    if model_path.name.endswith('xml'):
        model_path = model_path.parents[2]

    ov_config = kwargs['config']

    model_path_existed = Path(model_path).exists()
    # load model
    if not model_path_existed:
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')

    trust_remote_code = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        trust_remote_code = True
    if kwargs.get("genai", True) and is_genai_available(log_msg=True):
        try:
            return create_genai_text_reranker_model(model_path, device, memory_monitor, tokenizer, **kwargs)
        except Exception as exp:
            log.warning(
                f"Model is not supported by OpenVINO GenAI. "
                f"GenAI pipeline loading failed with following error: {exp}"
                "Benchmark will be switched to Optimum Intel pipeline realization"
            )
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    kwargs['use_case'].adjust_model_class_by_config(model_config)
    log.info("Selected Optimum Intel for benchmarking")
    if kwargs.get("mem_consumption"):
        memory_monitor.start()
    try:
        start = time.perf_counter()
        ov_model = kwargs['use_case'].ov_cls.from_pretrained(
            model_path, device=device, ov_config=ov_config, trust_remote_code=trust_remote_code
        )
        end = time.perf_counter()
    except ValueError:
        start = time.perf_counter()
        ov_model = kwargs['use_case'].ov_cls.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_cache=False,
            device=device,
            ov_config=ov_config
        )
        end = time.perf_counter()

    if kwargs.get("mem_consumption"):
        memory_monitor.stop_and_collect_data('compilation_phase')
        memory_monitor.log_data('for compilation phase')
    bench_hook = get_bench_hook(1, ov_model, rag=True)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    return ov_model, tokenizer, from_pretrained_time, bench_hook, False
