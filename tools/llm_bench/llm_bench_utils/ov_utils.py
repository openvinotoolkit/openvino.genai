# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from openvino.runtime import Core
import openvino as ov
import logging as log
import torch
import time
import json
import types
from llm_bench_utils.hook_common import get_bench_hook
from llm_bench_utils.config_class import (
    OV_MODEL_CLASSES_MAPPING,
    TOKENIZE_CLASSES_MAPPING,
    DEFAULT_MODEL_CLASSES,
    IMAGE_GEN_CLS
)
import openvino.runtime.opset13 as opset
from transformers import pipeline
import openvino_genai as ov_genai
import queue
from transformers.generation.streamers import BaseStreamer


def generate_simplified(self, *args, **kwargs):
    if len(args):
        raise Exception(f'Not empty args is not supported in generate_simplified, given: {args}')
    # TODO: Check other ignored parameters and report about them

    log.warning('Termination criteria is not supported in overridden generate, max_new_tokens only matters')

    # TODO: Check if unsupported kwargs are provided

    input_ids = kwargs['input_ids']
    attention_mask = kwargs['attention_mask']

    assert kwargs['num_beams'] == 1, "Overridden generate doesn't support num_beams > 1"

    past_key_values = None

    for _i in range(kwargs['max_new_tokens']):
        outputs = self(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)

        next_tokens = outputs.logits  # logits is an old name from original model, when interprocessing is fused it is a token
        # TODO: Apply termination criteria in addition to max_new_tokens
        # TODO: Doing the cat with input_ids here, we will 'uncat' it later in the next forward,
        # avoid doing it by passible next_tokens (without cat) directly to the next forward
        input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        # Depending on whether we are in stateful mode, past_key_values may or may not represent meaningful values,
        # need to pass them anyway to identify the first iteration
        past_key_values = outputs.past_key_values

    return input_ids


def patch_decoding_strategy(hf_model, patch_methods, **kwargs):
    """Fuse post-processing as an extra ops into a model."""
    ov_model = hf_model.model

    if kwargs.get('fuse_decoding_strategy', False):
        ppp = ov.preprocess.PrePostProcessor(ov_model)

        assert kwargs['num_beams'] == 1, "Parameter fuse_decoding_strategy doesn't support beam_search, set num_beams to 1"

        def greedy_search(input_port):
            next_token = opset.gather(input_port, opset.constant(-1), opset.constant(1))  # take last logits only (makes sense at the first iteration only)
            topk = opset.topk(next_token, opset.constant(1), axis=-1, mode='max', sort='none').output(1)
            return topk

        ppp.output(0).postprocess().custom(greedy_search)

        ov_model = ppp.build()
        hf_model.model = ov_model
        if patch_methods:
            hf_model._orig_generate = hf_model.generate
            hf_model.generate = types.MethodType(generate_simplified, hf_model)


def save_model(hf_model, **kwargs):
    xml_file_name = kwargs['save_prepared_model']
    if xml_file_name is not None:
        log.info(f'Saving prepared OpenVINO model to {xml_file_name} ...')
        ov.save_model(hf_model.model, xml_file_name)


def patch_inter_processing_and_compile(hf_model, **kwargs):
    patch_decoding_strategy(hf_model, True, **kwargs)
    save_model(hf_model, **kwargs)
    hf_model.compile()


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


def get_lora_config(lora_paths, lora_alphas):
    import openvino_genai

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
        adapter_config = openvino_genai.AdapterConfig()
        adapter = openvino_genai.Adapter(lora_paths[idx])
        alpha = float(lora_alphas[idx])
        adapter_config.add(adapter, alpha)

    if adapter_config:
        log.info('LoRA adapter(s) are added to config.')

    return adapter_config


def create_text_gen_model(model_path, device, **kwargs):
    """Create text generation model.

    - model_path: can be model_path or IR path
    - device: can be CPU or GPU
    - model_type:
    """
    default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_type = kwargs.get('model_type', default_model_type)
    model_class = OV_MODEL_CLASSES_MAPPING.get(model_type, OV_MODEL_CLASSES_MAPPING[default_model_type])
    token_class = TOKENIZE_CLASSES_MAPPING.get(model_type, TOKENIZE_CLASSES_MAPPING[default_model_type])
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
            if model_class not in [OV_MODEL_CLASSES_MAPPING[default_model_type], OV_MODEL_CLASSES_MAPPING["mpt"], OV_MODEL_CLASSES_MAPPING["chatglm"]]:
                log.warning("OpenVINO GenAI based benchmarking is not available for {model_type}. Will be switched to default benchmarking")
            else:
                log.info("Selected OpenVINO GenAI for benchmarking")
                return create_genai_text_gen_model(model_path, device, ov_config, **kwargs)
        log.info("Selected Optimum Intel for benchmarking")
        remote_code = False
        try:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        except Exception:
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            remote_code = True
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(
            model_path,
            device=device,
            ov_config=ov_config,
            config=model_config,
            stateful=kwargs.get("stateful", None),
            trust_remote_code=remote_code
        )
        if not isinstance(ov_model, OV_MODEL_CLASSES_MAPPING['t5']):
            patch_inter_processing_and_compile(ov_model, **kwargs)
        end = time.perf_counter()
    bench_hook = get_bench_hook(kwargs['num_beams'], ov_model)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    # load token
    tokenizer = token_class.from_pretrained(model_path, trust_remote_code=True)
    if kwargs.get("convert_tokenizer", False):
        tokenizer = build_ov_tokenizer(tokenizer)
    return ov_model, tokenizer, from_pretrained_time, bench_hook, False


def get_scheduler_config_genai(user_config, config_name="CB config"):
    import openvino_genai

    default_cb_config = {"cache_size": 1}
    scheduler_config = openvino_genai.SchedulerConfig()
    scheduler_params = user_config or default_cb_config
    if scheduler_params:
        log.info(f"Scheduler parameters for {config_name}:\n{scheduler_params}")

        for param, value in scheduler_params.items():
            setattr(scheduler_config, param, value)

    return scheduler_config


def create_genai_text_gen_model(model_path, device, ov_config, **kwargs):
    import openvino_tokenizers  # noqa: F401
    import openvino_genai
    from transformers import AutoTokenizer

    if not (model_path / "openvino_tokenizer.xml").exists() or not (model_path / "openvino_detokenizer.xml").exists():
        convert_ov_tokenizer(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    draft_model_path = kwargs.get("draft_model", '')
    cb = kwargs.get("use_cb", False)
    if cb or draft_model_path:
        log.info("Continuous Batching mode activated")
        ov_config["scheduler_config"] = get_scheduler_config_genai(kwargs.get("cb_config"))

    if draft_model_path:
        if not Path(draft_model_path).exists():
            raise RuntimeError(f'==Failure ==: draft model by path:{draft_model_path} is not exists')
        log.info("Speculative Decoding is activated")
        draft_device = kwargs.get('draft_device', None) or device
        draft_model_load_kwargs = {'scheduler_config': get_scheduler_config_genai(kwargs.get("draft_cb_config"), "draft CB config")}\
            if kwargs.get("draft_cb_config") is not None else {}
        ov_config['draft_model'] = openvino_genai.draft_model(draft_model_path, draft_device.upper(), **draft_model_load_kwargs)

    adapter_config = get_lora_config(kwargs.get("lora", None), kwargs.get("lora_alphas", []))
    if adapter_config:
        ov_config['adapters'] = adapter_config

    start = time.perf_counter()
    llm_pipe = openvino_genai.LLMPipeline(model_path, device.upper(), **ov_config)
    end = time.perf_counter()
    log.info(f'Pipeline initialization time: {end - start:.2f}s')

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
    streamer = TokenStreamer(llm_pipe.get_tokenizer()) if cb or draft_model_path else None

    return llm_pipe, tokenizer, end - start, streamer, True


def convert_ov_tokenizer(tokenizer_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    export_tokenizer(hf_tokenizer, tokenizer_path)


def create_image_gen_model(model_path, device, **kwargs):
    model_class = IMAGE_GEN_CLS
    model_path = Path(model_path)
    ov_config = kwargs['config']
    if not Path(model_path).exists():
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        if kwargs.get("genai", True) and is_genai_available(log_msg=True):
            log.info("Selected OpenVINO GenAI for benchmarking")
            return create_genai_image_gen_model(model_path, device, ov_config, **kwargs)

        log.info("Selected Optimum Intel for benchmarking")
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(model_path, device=device, ov_config=ov_config)
        end = time.perf_counter()
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


def create_genai_image_gen_model(model_path, device, ov_config, **kwargs):
    import openvino_genai

    class PerfCollector:
        def __init__(self) -> types.NoneType:
            self.iteration_time = []
            self.start_time = time.perf_counter()
            self.duration = -1

        def __call__(self, step, latents):
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

        def get_unet_latency(self):
            return (sum(self.iteration_time) / len(self.iteration_time)) * 1000 if len(self.iteration_time) > 0 else 0

        def get_vae_decoder_latency(self):
            if self.duration != -1:
                vae_time = self.duration - sum(self.iteration_time)
                return vae_time * 1000
            return 0

        def get_text_encoder_latency(self):
            return -1

        def get_text_encoder_step_count(self):
            return -1

        def get_unet_step_count(self):
            return len(self.iteration_time)

        def get_vae_decoder_step_count(self):
            return 1

    callback = PerfCollector()

    adapter_config = get_lora_config(kwargs.get("lora", None), kwargs.get("lora_alphas", []))
    if adapter_config:
        ov_config['adapters'] = adapter_config

    data = {}
    with open(str(model_path / "model_index.json"), 'r') as f:
        data = json.load(f)

    model_class_name = data.get("_class_name", "")

    start = time.perf_counter()

    scheduler_type = data.get("scheduler", ["", ""])[1]
    if (scheduler_type not in ["LCMScheduler", "DDIMScheduler", "LMSDiscreteScheduler", "EulerDiscreteScheduler", "FlowMatchEulerDiscreteScheduler"]):
        scheduler = openvino_genai.Scheduler.from_config(model_path / "scheduler/scheduler_config.json", openvino_genai.Scheduler.Type.DDIM)
        log.warning(f'Type of scheduler {scheduler_type} is unsupported. Please, be aware that it will be replaced to DDIMScheduler')

        vae_type = data.get("vae", [])
        if ("AutoencoderKL" in vae_type):
            vae = openvino_genai.AutoencoderKL(model_path / "vae_decoder", device.upper(), **ov_config)
        else:
            raise RuntimeError(f'==Failure ==: model by path:{model_path} has unsupported vae decoder type {vae_type}')

        if model_class_name == "StableDiffusionPipeline":
            text_encoder = get_genai_clip_text_encoder(data, model_path, device, ov_config)
            unet = get_genai_unet_model(data, model_path, device, ov_config)
            t2i_pipe = openvino_genai.Text2ImagePipeline.stable_diffusion(scheduler, text_encoder, unet, vae)
        elif model_class_name == "LatentConsistencyModelPipeline":
            text_encoder = get_genai_clip_text_encoder(data, model_path, device, ov_config)
            unet = get_genai_unet_model(data, model_path, device, ov_config)
            t2i_pipe = openvino_genai.Text2ImagePipeline.latent_consistency_model(scheduler, text_encoder, unet, vae)
        elif model_class_name == "StableDiffusionXLPipeline":
            clip_text_encoder = get_genai_clip_text_encoder(data, model_path, device, ov_config)
            clip_text_encoder_2 = get_genai_clip_text_encoder_with_projection(data, model_path, "text_encoder_2", device, ov_config)
            unet = get_genai_unet_model(data, model_path, device, ov_config)
            t2i_pipe = openvino_genai.Text2ImagePipeline.stable_diffusion_xl(scheduler, clip_text_encoder, clip_text_encoder_2, unet, vae)
        else:
            raise RuntimeError(f'==Failure ==: model by path:{model_path} has unsupported _class_name {model_class_name}')
    else:
        t2i_pipe = openvino_genai.Text2ImagePipeline(model_path, device.upper(), **ov_config)

    end = time.perf_counter()
    log.info(f'Pipeline initialization time: {end - start:.2f}s')
    return t2i_pipe, end - start, True, callback


def create_ldm_super_resolution_model(model_path, device, **kwargs):
    core = Core()
    ov_config = kwargs['config']
    core.set_property(ov_config)
    default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_type = kwargs.get('model_type', default_model_type)
    model_class = OV_MODEL_CLASSES_MAPPING[model_type]
    model_path = Path(model_path)
    start = time.perf_counter()
    ov_model = model_class(model_path, core, device.upper())
    end = time.perf_counter()
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    return ov_model, from_pretrained_time


def create_genai_speech_2_txt_model(model_path, device, **kwargs):
    import openvino_genai as ov_genai
    if kwargs.get("genai", True) is False:
        raise RuntimeError('==Failure the command line does not set --genai ==')
    if is_genai_available(log_msg=True) is False:
        raise RuntimeError('==Failure genai is not enable ==')
    start = time.perf_counter()
    genai_pipe = ov_genai.WhisperPipeline(model_path, device.upper())
    end = time.perf_counter()
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    processor = AutoProcessor.from_pretrained(model_path)
    return genai_pipe, processor, from_pretrained_time, True


def create_speech_2txt_model(model_path, device, **kwargs):
    """Create speech generation model.
    - model_path: can be model_path or IR path
    - device: can be CPU
    - model_type:
    """
    default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_type = kwargs.get('model_type', default_model_type)
    model_class = OV_MODEL_CLASSES_MAPPING.get(model_type, OV_MODEL_CLASSES_MAPPING[default_model_type])
    model_path = Path(model_path)
    model_path_existed = model_path.exists()
    # load model
    if not model_path_existed:
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        if kwargs.get("genai", True) and is_genai_available(log_msg=True):
            if model_class not in [OV_MODEL_CLASSES_MAPPING[default_model_type]]:
                log.warning("OpenVINO GenAI based benchmarking is not available for {model_type}. Will be switched to default bencmarking")
            else:
                log.info("Selected OpenVINO GenAI for benchmarking")
                return create_genai_speech_2_txt_model(model_path, device, **kwargs)
        log.info("Selected Optimum Intel for benchmarking")
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(
            model_path,
            device=device
        )
        end = time.perf_counter()
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    processor = AutoProcessor.from_pretrained(model_path)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=ov_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
    )

    return pipe, processor, from_pretrained_time, False


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
            elif len(text) >= 3 and text[-3:] == chr(65533):
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
