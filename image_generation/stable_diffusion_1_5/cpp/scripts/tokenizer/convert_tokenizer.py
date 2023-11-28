# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Any, Tuple, Union

from openvino.runtime.exceptions import OVTypeError
from openvino.runtime import Model


def convert_tokenizer(
        tokenizer_object: Any, number_of_inputs: int = 1, with_decoder=False
) -> Union[Model, Tuple[Model, Model]]:
    if "transformers" in sys.modules:
        from transformers import PreTrainedTokenizerBase
        from .hf_parser import TransformersTokenizerPipelineParser

        # TODO: Remove this check
        if isinstance(tokenizer_object, PreTrainedTokenizerBase):
            pipeline = TransformersTokenizerPipelineParser(tokenizer_object).parse(
                number_of_inputs=number_of_inputs
            )
            ov_tokenizer = pipeline.get_encoder_ov_subgraph()
            if with_decoder:
                ov_detokenizer = pipeline.get_decoder_ov_subgraph()
            output_names = tokenizer_object.model_input_names

            ov_tokenizer_output_names = ["input_ids", "attention_mask"]
            if len(output_names) == 3 and len(ov_tokenizer.outputs) == 3:
                ov_tokenizer_output_names.insert(1, "token_type_ids")

            filtered_outputs = []
            for i, output_name in enumerate(ov_tokenizer_output_names):
                current_output = next(
                    (output for output in ov_tokenizer.outputs if output.any_name == output_name), False
                )
                if current_output:
                    filtered_outputs.append(current_output)
                    continue

                if output_name in output_names:
                    ov_tokenizer.output(i).tensor.add_names({output_name})
                    filtered_outputs.append(ov_tokenizer.output(i))

            if with_decoder:
                return Model(filtered_outputs, ov_tokenizer.get_parameters()), ov_detokenizer

            return Model(filtered_outputs, ov_tokenizer.get_parameters())

    raise OVTypeError(f"Tokenizer type is not supported: {type(tokenizer_object)}")


def connect_models(model1: Model, model2: Model, name_map=None, *, by_indices=None, by_names=None) -> Model:
    # TODO: Relax this limitation by not connecting some inputs/outputs together
    #print(len(model2.inputs))
    #print(len(model1.outputs))
    #assert len(model2.inputs) == len(model1.outputs)

    if by_indices is None and by_names is None:
        by_names = True

    if name_map is not None:
        by_names = True

    # TODO: Check only one of by_indices and by_names is set

    if by_indices:
        aligned_model1_outputs = model1.outputs
        aligned_model2_inputs = model2.inputs
    elif by_names:
        if name_map is None:
            aligned_model1_outputs = model1.outputs
            aligned_model2_inputs = [model2.input(model1_output.get_any_name()) for model1_output in aligned_model1_outputs]

            '''
            aligned_model1_outputs = []
            aligned_model2_inputs = []
            for model2_input in model2.inputs:
                # Search for corresponding model1 output by all possible names
                for model1_output in model2.outputs
            '''

        else:
            aligned_model1_outputs = [model1.output(name1) for name1, _ in name_map]
            aligned_model2_inputs = [model2.input(name2) for _, name2 in name_map]

    for model2_input, model1_output in zip(aligned_model2_inputs, aligned_model1_outputs):
        #print(f'Connecting: {model1_output.get_any_name()} -> {model2_input.get_any_name()}')
        for target in model2_input.get_target_inputs():
            target.replace_source_output(model1_output.get_node().input_value(0))
            #target.replace_source_output(model1_output)  # TODO: Produces incorrect topology

    connected_model = Model(model2.outputs, model1.get_parameters())
    # TODO: Cleanup model1 and mode2 to avoid using them, they are ill-formed after the reconnection
    connected_model.validate_nodes_and_infer_types()
    return connected_model
