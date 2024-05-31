# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from optimum.exporters.onnx.config import TextDecoderOnnxConfig, TextDecoderWithPositionIdsOnnxConfig
from optimum.exporters.tasks import TasksManager
from optimum.utils import (
    NormalizedTextConfig,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
)
from optimum.exporters.openvino.model_configs import register_in_tasks_manager


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


@register_in_tasks_manager('yi', *["text-generation", "text-generation-with-past"])
class YIOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14
    DUMMY_INPUT_GENERATOR_CLASSES = (
        YIDummyTextInputGenerator,
        DummyPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = DummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    no_position_ids = False


@register_in_tasks_manager("jais", *["text-generation", "text-generation-with-past"])
class JaisOpenVINOConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers='n_layer', num_attention_heads='n_head', hidden_size='n_embd')


TasksManager._SUPPORTED_MODEL_TYPE['stablelm_epoch'] = TasksManager._SUPPORTED_MODEL_TYPE['stablelm']
TasksManager._SUPPORTED_MODEL_TYPE['stablelm-epoch'] = TasksManager._SUPPORTED_MODEL_TYPE['stablelm']
TasksManager._SUPPORTED_MODEL_TYPE['stablelm2'] = TasksManager._SUPPORTED_MODEL_TYPE['stablelm']
TasksManager._SUPPORTED_MODEL_TYPE["aquila"] = TasksManager._SUPPORTED_MODEL_TYPE["stablelm"]
TasksManager._SUPPORTED_MODEL_TYPE["codegen2"] = TasksManager._SUPPORTED_MODEL_TYPE["codegen"]
