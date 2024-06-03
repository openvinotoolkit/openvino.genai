# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from transformers import AutoConfig
from openvino.runtime import Core
import openvino as ov
import logging as log
import torch
import time
import types
import utils.hook_greedy_search
import utils.hook_beam_search

from utils.config_class import OV_MODEL_CLASSES_MAPPING, TOKENIZE_CLASSES_MAPPING, DEFAULT_MODEL_CLASSES
import openvino.runtime.opset13 as opset


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
    ov_compiled_tokenizer = ov.compile_model(tokenizer_model)
    ov_compiled_detokenizer = ov.compile_model(detokenizer_model)

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
        if kwargs.get("genai", False) and is_genai_available(log=True):
            if kwargs["batch_size"] > 1 or kwargs["num_beams"] > 1:
                log.warning("OpenVINO GenAI based benchmarking implmented only for batch_size == 1 and num_beams == 1")
            elif model_class not in [OV_MODEL_CLASSES_MAPPING[default_model_type], OV_MODEL_CLASSES_MAPPING["mpt"]]:
                log.warning("OpenVINO GenAI based benchmarking is not available for {model_type}. Will be switched to default bencmarking") 
            else:
                return create_genai_text_gen_model(model_path, device, ov_config, **kwargs)
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
    if kwargs['num_beams'] > 1:
        bench_hook = utils.hook_beam_search.BeamSearchHook()
    else:
        bench_hook = utils.hook_greedy_search.GreedySearchHook()
    bench_hook.new_forward(ov_model, model_type)
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    # load token
    tokenizer = token_class.from_pretrained(model_path, trust_remote_code=True)
    if kwargs.get("convert_tokenizer", False):
        tokenizer = build_ov_tokenizer(tokenizer)
    return ov_model, tokenizer, from_pretrained_time, bench_hook, False


def create_genai_text_gen_model(model_path, device, ov_config, **kwargs):
    import openvino_genai
    import openvino_tokenizers
    from transformers import AutoTokenizer

    class TokenStreamer(openvino_genai.StreamerBase):
        def __init__(self, tokenizer):
            super().__init__()
            self.tokenizer = tokenizer
            self.token_generation_time = []
            self.generated_tokens = []
            self.start_time = time.perf_counter()

        def put(self, token_id):
            self.token_generation_time.append(time.perf_counter() - self.start_time)
            self.generated_tokens.append(token_id)
            self.start_time = time.perf_counter()

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
    
    if not (model_path / "openvino_tokenizer.xml").exists() or not (model_path / "openvino_detokenizer.xml").exists():
        convert_ov_tokenizer(model_path)
    
    core = Core()
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ov_tok = core.read_model(model_path / "openvino_tokenizer.xml")
    ov_detok = core.read_model(model_path / "openvino_detokenizer.xml")
    hf_tokenizer = build_ov_tokenizer_wrapper(hf_tokenizer, ov_tok, ov_detok)
    
    start = time.perf_counter()

    # TO DO: add plugin config
    llm_pipe = openvino_genai.LLMPipeline(str(model_path), device.upper())
    end = time.perf_counter()
    log.info(f'Pipeline initialization time: {end - start:.2f}s')
    streamer = TokenStreamer(llm_pipe.get_tokenizer())

    return llm_pipe, hf_tokenizer, end - start, streamer, True

def convert_ov_tokenizer(tokenizer_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    export_tokenizer(hf_tokenizer, tokenizer_path)


def create_image_gen_model(model_path, device, **kwargs):
    default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_type = kwargs.get('model_type', default_model_type)
    model_class = OV_MODEL_CLASSES_MAPPING[model_type]
    model_path = Path(model_path)
    ov_config = kwargs['config']
    if not Path(model_path).exists():
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        start = time.perf_counter()
        ov_model = model_class.from_pretrained(model_path, device=device, ov_config=ov_config)
        end = time.perf_counter()
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    return ov_model, from_pretrained_time


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


def is_genai_available(log=False):
    try:
        import openvino_genai
    except ImportError as ex:
        if log:
            log.warning("Attempt to load OpenVINO GenaAI package failed. Please install openvino_genai package. Full error message available in debug mode")
            log.debug(ex)
            return False
    return True
        