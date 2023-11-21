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
    has_beam_table = getattr(self, 'has_beam_table', False)
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
            if has_beam_table:
                # beam_table is not stored in past_key_values hence it is not affected by kv_cache external reordering
                inputs['beam_table_input'] = self.beam_table

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
            if has_beam_table:
                model_input = self.model.input('beam_table_input')
                shape = model_input.get_partial_shape()
                shape = [d.get_length() if not d.is_dynamic else 0 for d in shape]
                inputs['beam_table_input'] = Tensor(model_input.get_element_type(), shape)
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
            if has_beam_table:
                # beam_table is not a part of potentially externally modified kv_cache, so no need to cast it to numpy
                # as there is no other code that access it except this function
                self.beam_table = self.request.get_tensor('beam_table_output')
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

    if kwargs['fuse_decoding_strategy']:
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

    num_beams = kwargs['num_beams'] if 'num_beams' in kwargs and kwargs['num_beams'] > 1 else 1
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs and kwargs['batch_size'] > 1 else 1
    batch_dim_size = num_beams * batch_size
    hf_model.has_beam_idx = False
    hf_model.has_beam_table = False
    orig_num_params = len(ov_model.get_parameters())

    if kwargs['fuse_cache_reorder'] and num_beams > 1:
        # Should be run before make_stateful because of adding pre-processing on kv-cache inputs
        # Make a new parameter for beam_idx
        # Adding a new parameter to make _reorder_cache inside the model in the beginning of each iteration
        beam_idx = opset.parameter(name='beam_idx', dtype=ov.Type.i32, shape=ov.PartialShape([batch_dim_size]))
        beam_idx.output(0).get_tensor().add_names({'beam_idx'})  # why list is not accepted?

        if kwargs['indirect_cache']:
            # Create one extra input for beam_table with shape [sequence_length, batch_dim_size]
            # TODO: Provide layout for beam_table compatible with axes of kv_cache inputs/outputs
            #       for uniform external cache manipulation. For example [B, 1, S, 1], where B - batch/dim, S - sequence.
            beam_table = opset.parameter(name='beam_table_input', dtype=ov.Type.i32, shape=ov.PartialShape([-1, batch_dim_size]))
            beam_table.output(0).get_tensor().add_names({'beam_table_input'})  # why list is not accepted?
            # Adding beam_table parameter before beam_idx because this parameter will potentially
            # be bound with a corresponding output to keep it as a variable -- need to maintain
            # all such input/output pairs aligned for more convenient application of make_stateful transformation
            ov_model.add_parameters([beam_table])
            hf_model.has_beam_table = True

        ov_model.add_parameters([beam_idx])
        hf_model.has_beam_idx = True

        # Detects if op is Concat in the pattern Parameter -> Concat -> Result, and Parameter is among of the state parameters
        def match_concat(op):
            # use Node.get_name for nodes to match them to other nodes as just node comparison node1 == node2 gives
            # different effect due to the syntax sugar implemented in OV Python API: creation of Equal operation Equal(node1, node2)
            # which is not what we need
            return (op.get_type_name() == "Concat"
                    and op.input_value(0).get_node().get_name() in [x.get_name() for x in ov_model.get_parameters()[2:orig_num_params]]
                    and 'Result' in [consumer.get_node().get_type_name() for consumer in op.output(0).get_target_inputs()])

        if kwargs['indirect_cache']:
            # Search for the topologically first kv_cache Concat
            first_concat = next(op for op in ov_model.get_ordered_ops() if match_concat(op))
            assert first_concat.get_input_size() == 2

            # Build the indirect cache maintainance sub-graph, which
            #  - takes beam_table_input, beam_idx and an in-model produced input for the found first_concat Concat op,
            #  - produces updated value for beam_table, and
            #  - sends it to a newly created Result node beam_table_output.

            beam_table_reordered = opset.gather(beam_table, beam_idx, opset.constant(1))
            # second input to concat is the current k/v-value, need it to calculate the shape of the addon to beam_table
            new_kv_cache_part = first_concat.input_value(1)
            concat_axis = first_concat.get_attributes()['axis']
            new_kv_cache_size = opset.gather(
                opset.shape_of(new_kv_cache_part),
                opset.constant([concat_axis]),
                opset.constant(0))
            repeater = opset.concat([new_kv_cache_size, opset.constant([1])], axis=0)
            # addon to beam_table is repeated slice with 0..(batch_dim_size-1) values
            new_beam_table_part = opset.tile(
                opset.range(
                    opset.constant(0),
                    opset.constant(batch_dim_size),
                    opset.constant(1),
                    output_type='i32'),  # TODO: Why cannot pass ov.Type.i32?
                repeater)
            updated_beam_table = opset.concat([beam_table_reordered, new_beam_table_part], axis=0)
            updated_beam_table.output(0).get_tensor().add_names({'beam_table_output'})
            ov_model.add_results([opset.result(updated_beam_table, name='beam_table_output')])

        transformed_concat_count = 0

        # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx/beam_table
        for i in range(orig_num_params - 2):  # 2 == input_ids, attention_mask
            parameter_output_port = ov_model.inputs[2 + i]
            consumers = list(parameter_output_port.get_target_inputs())
            if kwargs['indirect_cache']:
                matched_concats = tuple(consumer for consumer in consumers if match_concat(consumer.get_node()))
                unexpected_consumers = tuple(consumer for consumer in consumers if consumer.get_node().get_type_name() not in ('Concat', 'ShapeOf'))
                if len(matched_concats) == 1 and len(unexpected_consumers) == 0:
                    # take all consumers of the concat and filter out Results
                    concat = matched_concats[0].get_node()
                    concat_consumers = [consumer for consumer in concat.output(0).get_target_inputs() if consumer.get_node().get_type_name() != 'Result']
                    if len(concat_consumers) == 0:
                        print(f'[ WARNING ] Too deep pattern match failure, cannot find any kv-cache concat consumer except Result for node {concat}')
                        continue

                    # Move gather dimensions to the leading positions in the kv-value shape.
                    # HW plugins are supposed to detect this sequence and replace it with their efficient
                    # indirect cache implementation.
                    # TODO: Rework to handle any appropriate rank, now works for rank = 4 only.
                    # TODO: Propose a new Gather op version with multiple axes to avoid applying Transposes
                    transposed = opset.transpose(concat, opset.constant([2, 0, 1, 3]))
                    # restore correct order of kv-cache according to beam_table
                    reordered = opset.gather(transposed, updated_beam_table, opset.constant(1), batch_dims=1)
                    # reverse transpose to restore original kv-cache layout
                    restored = opset.transpose(reordered, opset.constant([1, 2, 0, 3]))

                    # Replace old input for all concat_consumers by a newly created restored tensor
                    for consumer in concat_consumers:
                        consumer.replace_source_output(restored.output(0))

                    transformed_concat_count += 1
            else:
                gather = opset.gather(parameter_output_port, beam_idx, opset.constant(0))
                for consumer in consumers:
                    consumer.replace_source_output(gather.output(0))

        if kwargs['indirect_cache']:
            assert transformed_concat_count == orig_num_params - 2, (
                f'Failed to find all places in model to apply indirect cache transformation: '
                f'appplied to {transformed_concat_count} state tensors among {orig_num_params - 2} in total'
            )

        ov_model.validate_nodes_and_infer_types()
        hf_model.use_cache_as_state = False

        # override _reorder_cache to avoid cache manipulation outside of the model as it is already done inside
        def _reorder_cache_stub(self, past_key_values, beam_idx):
            # TODO: Apply it differently based on model type
            self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
            return past_key_values

        hf_model._reorder_cache = types.MethodType(_reorder_cache_stub, hf_model)
        hf_model.forward = types.MethodType(forward_simplified, hf_model)  # need custom forward to set beam_idx input to OV model
        hf_model.next_beam_idx = np.zeros([batch_dim_size], dtype=int)  # initial value for beam_idx is all zeros

    if kwargs['make_stateful']:
        from openvino._offline_transformations import apply_make_stateful_transformation

        input_output_map = {}
        # TODO: Can we derive the dimensions from the model topology?
        num_attention_heads = hf_model.normalized_config.num_attention_heads if hf_model.config.model_type == 'bloom' else 1

        assert num_beams == 1 or hf_model.has_beam_idx, (
            'Requested to make_stateful with num_beams > 1 but there is no beam_idx parameter for cache reorder fused'
        )

        left_num_parameters = 2 + int(hf_model.has_beam_idx)
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for i in range(2):
            input = ov_model.inputs[i]
            shape = input.get_partial_shape()
            if shape.rank.get_length() == 2:
                shape[0] = batch_size * num_beams
                input.get_node().set_partial_shape(shape)
            else:
                print(f'[ WARNING ] Rank of {i} input of the model is not 2, batch size is not set')

        for i in range(len(ov_model.inputs) - left_num_parameters):
            input = ov_model.inputs[2 + i]
            output = ov_model.outputs[1 + i]
            input_output_map[input.any_name] = output.any_name

            if not (hf_model.has_beam_table and input.any_name and input.any_name == 'beam_table_input'):
                # Set batch dimension only for original kv-cache inputs, because beam_table has got already correct shape
                # and it has a different layout which makes it incompatible with the shape manipulation code below.
                shape = input.get_partial_shape()
                # suppose 0-th dimension is a batch
                # TODO: Deduce from a model via ordinal reshape
                shape[0] = batch_size * num_attention_heads * num_beams
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
        if model_type in ['mpt', 'falcon', 'replit', 'codegen2', 'chatglm', 'chatglm2']:
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
    return ov_model, tokenizer, from_pretrained_time


def create_image_gen_model(model_path, device, **kwargs):
    default_model_type = DEFAULT_MODEL_CLASSES[kwargs['use_case']]
    model_type = kwargs.get('model_type', default_model_type)
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
