# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from transformers import AutoConfig
from openvino.runtime import Core
import openvino as ov
import logging as log
import torch
import time
import types

from utils.config_class import OV_MODEL_CLASSES_MAPPING, TOKENIZE_CLASSES_MAPPING, DEFAULT_MODEL_CLASSES
from .ov_model_classes import register_normalized_configs
from transformers.modeling_outputs import CausalLMOutputWithPast
from openvino import Type, Tensor
import numpy as np


def forward_simplified(
    self,
    input_ids: torch.LongTensor,
    attention_mask=None,
    past_key_values=None,
    position_ids=None,
    **kwargs,
) -> CausalLMOutputWithPast:
    self.compile()

    if self.use_cache and past_key_values is not None:
        input_ids = input_ids[:, -1:]

    inputs = {}
    if not self.use_cache_as_state:
        if past_key_values is not None:
            if self._pkv_precision == Type.bf16:
                # numpy does not support bf16, pretending f16, should change to bf16
                past_key_values = tuple(
                    Tensor(past_key_value, past_key_value.shape, Type.bf16) for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                )
            else:
                # Flatten the past_key_values
                past_key_values = tuple(past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer)
            # Add the past_key_values to the decoder inputs
            inputs = dict(zip(self.key_value_input_names, past_key_values))

        # Create empty past_key_values for decoder_with_past first generation step
        elif self.use_cache:
            shape_input_ids = input_ids.shape
            num_attention_heads = self.normalized_config.num_attention_heads if self.config.model_type == 'bloom' else 1
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = shape_input_ids[0] * num_attention_heads
                if shape[2].is_dynamic:
                    shape[2] = 0
                if shape[1].is_dynamic:
                    shape[1] = 0
                if shape.rank.get_length() == 4 and shape[3].is_dynamic:
                    shape[3] = 0
                inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())
    else:
        # past_key_values are not used explicitly, instead they should be handled inside the model
        if past_key_values is None:
            # Need a marker to differentiate the first generate iteration from the others in
            # the first condition at the function beginning above.
            # It should be something that is not None and it should be True when converted to Boolean.
            past_key_values = ((),)
            # This is the first iteration in a sequence, reset all states
            for state in self.request.query_state():
                state.reset()

    inputs['input_ids'] = np.array(input_ids)

    # Add the attention_mask inputs when needed
    if 'attention_mask' in self.input_names and attention_mask is not None:
        inputs['attention_mask'] = np.array(attention_mask)

    if 'position_ids' in self.input_names and position_ids is not None:
        inputs['position_ids'] = np.array(position_ids)

    if hasattr(self, 'next_beam_idx'):
        inputs['beam_idx'] = np.array(self.next_beam_idx)

    # Run inference
    self.request.start_async(inputs, share_inputs=True)
    self.request.wait()

    # this is probably not real logits but already post-processed values depending on whether post-processing is fused into a model or not
    logits = torch.from_numpy(self.request.get_tensor('logits').data).to(self.device)

    if not self.use_cache_as_state:
        if self.use_cache:
            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv))
        else:
            past_key_values = None

    return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


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
        # Depending on whether we applied make_stateful, past_key_values may or may not represent meaningful values,
        # need to pass them anyway to differentiate the first iteration
        past_key_values = outputs.past_key_values

    return input_ids


def patch_inter_processing(hf_model, **kwargs):
    """Fuse post-processing as an extra ops into a model."""
    ov_model = hf_model.model

    import openvino.runtime.opset12 as opset

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
        hf_model._orig_generate = hf_model.generate
        hf_model.generate = types.MethodType(generate_simplified, hf_model)

    num_beams = kwargs.get('num_beams', 1)
    batch_size = kwargs.get('batch_size', 1)

    not_kv_inputs = [input for input in ov_model.inputs if not any(name in hf_model.key_value_input_names for name in input.get_names())]

    if kwargs.get('fuse_cache_reorder', False) and num_beams > 1:
        # Should be run before make_stateful because of adding pre-processing on kv-cashe inputs
        # Make a new parameter for beam_idx
        # Adding a new parameter to make _reorder_cache inside the model in the beginning of each iteration
        beam_idx = opset.parameter(name='beam_idx', dtype=ov.Type.i32, shape=ov.PartialShape([num_beams * batch_size]))
        beam_idx.output(0).get_tensor().add_names({'beam_idx'})  # why list is not accepted?
        ov_model.add_parameters([beam_idx])
        not_kv_inputs.append(ov_model.inputs[-1])
        # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
        for input_name in hf_model.key_value_input_names:  # 3 == input_ids, attention_mask and new beam_idx
            parameter_output_port = ov_model.input(input_name)
            consumers = parameter_output_port.get_target_inputs()
            batch_idx = 1 if hf_model.config.model_type == 'chatglm' else 0
            gather = opset.gather(parameter_output_port, beam_idx, opset.constant(batch_idx))
            for consumer in consumers:
                consumer.replace_source_output(gather.output(0))
        ov_model.validate_nodes_and_infer_types()
        hf_model.use_cache_as_state = False

        # override _reorder_cache to avoid cache manipulation outside of the model as it is already done inside
        def _reorder_cache_stub(self, past_key_values, beam_idx):
            # TODO: Apply it differently based on model type
            self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
            return past_key_values

        hf_model._reorder_cache = types.MethodType(_reorder_cache_stub, hf_model)
        hf_model.forward = types.MethodType(forward_simplified, hf_model)  # need custom forward to set beam_idx input to OV model
        hf_model.next_beam_idx = np.zeros([num_beams * batch_size], dtype=int)  # initial value for beam_idx is all zeros

    if kwargs.get('make_stateful', False):
        from openvino._offline_transformations import apply_make_stateful_transformation

        input_output_map = {}
        # TODO: Can we derive the dimensions from the model topology?
        num_attention_heads = hf_model.normalized_config.num_attention_heads if hf_model.config.model_type == 'bloom' else 1
        beam_idx_exist = 'beam_idx' in [port.any_name for port in ov_model.inputs]
        assert num_beams == 1 or beam_idx_exist, 'Requested to make_stateful with num_beams > 1 but there is no beam_idx parameter for cache reorder fused'
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = batch_size * num_beams
                input.get_node().set_partial_shape(shape)
            else:
                print(f'[ WARNING ] Rank of {input.get_any_name()} input of the model is not 2, batch size is not set')

        for kv_name_pair in zip(hf_model.key_value_input_names, hf_model.key_value_output_names):
            input_output_map[kv_name_pair[0]] = kv_name_pair[1]
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()

            # By default, batch is the 0-th but chatglm uses 1-st dimension as batch
            # TODO: Deduce from a model via ordinal reshape
            batch_idx = 1 if hf_model.config.model_type == 'chatglm' else 0
            shape[batch_idx] = batch_size * num_attention_heads * num_beams

            input.get_node().set_partial_shape(shape)

        ov_model.validate_nodes_and_infer_types()

        apply_make_stateful_transformation(ov_model, input_output_map)

        hf_model.use_cache_as_state = True
        hf_model.forward = types.MethodType(forward_simplified, hf_model)  # override to avoid cache manipulation outside of the model

    xml_file_name = kwargs['save_prepared_model']
    if xml_file_name is not None:
        log.info(f'Saving prepared OpenVINO model to {xml_file_name} ...')
        ov.save_model(ov_model, xml_file_name)

    hf_model.compile()


def build_ov_tokenizer(hf_tokenizer):
    try:
        from ov_tokenizer import convert_tokenizer, pack_strings, unpack_strings
    except ImportError:
        log.warn("OV Tokenizer is unavailable, tokenizer conversion will be skipped")
        return hf_tokenizer

    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_decoder=True)
    ov_compiled_tokenizer = ov.compile_model(ov_tokenizer)
    ov_compiled_detokenizer = ov.compile_model(ov_detokenizer)

    def encode_ov_tokenizer(self, text, *args, **kwargs):
        if isinstance(text, str):
            text = [text]
        input_tensor = pack_strings(text)
        return ov_compiled_tokenizer(input_tensor)

    def batch_decode_ov_tokenizer(self, sequences, *args, **kwargs):
        result = unpack_strings(ov_compiled_detokenizer(sequences)["string_output"])
        return result

    def decode_ov_tokenizer(self, token_ids, *args, **kwargs):
        return self.batch_decode([token_ids])[0]

    hf_tokenizer.encode = types.MethodType(encode_ov_tokenizer, hf_tokenizer)
    hf_tokenizer.batch_encode = types.MethodType(encode_ov_tokenizer, hf_tokenizer)
    hf_tokenizer.__call__ = types.MethodType(encode_ov_tokenizer, hf_tokenizer)
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
    register_normalized_configs()

    model_path_existed = Path(model_path).exists()
    # load model
    if not model_path_existed:
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        if model_type in ['mpt', 'falcon', 'replit', 'codegen2', 'chatglm']:
            start = time.perf_counter()
            ov_model = model_class.from_pretrained(
                model_path,
                device=device,
                ov_config=ov_config,
                config=AutoConfig.from_pretrained(model_path, trust_remote_code=True),
            )
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            ov_model = model_class.from_pretrained(model_path, device=device, ov_config=ov_config, config=config, compile=False)
            if not isinstance(ov_model, OV_MODEL_CLASSES_MAPPING['t5']):
                patch_inter_processing(ov_model, **kwargs)
            end = time.perf_counter()
    from_pretrained_time = end - start
    log.info(f'From pretrained time: {from_pretrained_time:.2f}s')
    # load token
    tokenizer = token_class.from_pretrained(model_path, trust_remote_code=True)
    if kwargs.get("convert_tokenizer", False):
        tokenizer = build_ov_tokenizer(tokenizer)
    return ov_model, tokenizer, from_pretrained_time


def create_image_gen_model(model_path, device, **kwargs):
    default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_type = kwargs.get('model_type', default_model_type)
    print(model_type)
    model_class = OV_MODEL_CLASSES_MAPPING[model_type]
    model_path = Path(model_path)
    ov_config = kwargs['config']
    if not Path(model_path).exists():
        raise RuntimeError(f'==Failure ==: model path:{model_path} does not exist')
    else:
        log.info(f'model_path={model_path}')
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
