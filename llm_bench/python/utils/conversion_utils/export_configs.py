# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Callable, Dict, Type, Optional, Tuple

from optimum.exporters.onnx import TextDecoderOnnxConfig
from optimum.exporters.tasks import TasksManager, make_backend_config_constructor_for_task
from optimum.utils import (
    NormalizedTextConfig, DEFAULT_DUMMY_SHAPES,
    DummyPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyInputGenerator
)


class TextDecoderWithPositionIdsOnnxConfig(TextDecoderOnnxConfig):
    no_position_ids = False

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs

        # Decoders based on GPT2 require a position_ids input to avoid
        # generating wrong position_ids in the model itself:
        # https://github.com/huggingface/transformers/blob/v4.33.1/src/transformers/models/gpt2/modeling_gpt2.py#L802
        if not self.no_position_ids and "text-generation" in self.task:
            common_inputs["position_ids"] = {0: "batch_size", 1: "sequence_length"}

        return common_inputs


def create_register(overwrite_existing: bool = False):
    def wrapper(model_type: str, *supported_tasks: str) -> Callable[[Type], Type]:
        def decorator(config_cls: Type) -> Type:
            mapping = TasksManager._SUPPORTED_MODEL_TYPE.get(model_type, {})
            mapping_backend = mapping.get("onnx", {})
            for task in supported_tasks:
                normalized_task = task
                if "-with-past" in task:
                    normalized_task = task.split("-with-past")[0]
                if normalized_task not in TasksManager.get_all_tasks():
                    known_tasks = ", ".join(TasksManager.get_all_tasks())
                    raise ValueError(
                        f'The TasksManager does not know the task called "{task}", known tasks: {known_tasks}.'
                    )
                if not overwrite_existing and task in mapping_backend:
                    continue
                mapping_backend[task] = make_backend_config_constructor_for_task(config_cls, task)
            mapping["onnx"] = mapping_backend
            TasksManager._SUPPORTED_MODEL_TYPE[model_type] = mapping
            return config_cls

        return decorator

    return wrapper


register_in_tasks_manager = create_register()
register_in_tasks_manager_with_override = create_register(True)


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


@register_in_tasks_manager('mistral', *["text-generation", "text-generation-with-past"])
class MistralOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    # The ONNX export of this architecture needs the Trilu operator support, available since opset 14
    DEFAULT_ONNX_OPSET = 14
    DUMMY_INPUT_GENERATOR_CLASSES = (
        MistralDummyTextInputGenerator,
        MistralDummyPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = MistralDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_key_value_heads="num_key_value_heads", allow_new=True)
    no_position_ids = False


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


@register_in_tasks_manager("qwen", *["text-generation", "text-generation-with-past"])
class QwenOpenVINOConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size'
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (QwenDummyInputsGenerator, QwenDummyPastKeyValuesGenerator)
    DUMMY_PKV_GENERATOR_CLASS = QwenDummyPastKeyValuesGenerator
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
            past_length = dummy_inputs["past_key_values"][0][1].shape[1]

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
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 1: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 1: decoder_sequence_name}


@register_in_tasks_manager("baichuan", *["text-generation", "text-generation-with-past"])
class Baichaun2OpenVINOConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        num_layers='num_hidden_layers', num_attention_heads='num_attention_heads', hidden_size='hidden_size'
    )


@register_in_tasks_manager("jais", *["text-generation", "text-generation-with-past"])
class JaisOpenVINOConfig(TextDecoderOnnxConfig):
    DEFAULT_ONNX_OPSET = 13
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(num_layers='n_layer', num_attention_heads='n_head', hidden_size='n_embd')


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


@register_in_tasks_manager("chatglm", *["text-generation", "text-generation-with-past"])
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


TasksManager._SUPPORTED_MODEL_TYPE['stablelm_epoch'] = TasksManager._SUPPORTED_MODEL_TYPE['llama']
TasksManager._SUPPORTED_MODEL_TYPE['stablelm-epoch'] = TasksManager._SUPPORTED_MODEL_TYPE['llama']
TasksManager._SUPPORTED_MODEL_TYPE['stablelm'] = TasksManager._SUPPORTED_MODEL_TYPE['llama']
TasksManager._SUPPORTED_MODEL_TYPE["aquila"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
TasksManager._SUPPORTED_MODEL_TYPE["codegen2"] = TasksManager._SUPPORTED_MODEL_TYPE["codegen"]
TasksManager._SUPPORTED_MODEL_TYPE["mixtral"] = TasksManager._SUPPORTED_MODEL_TYPE['mistral']
TasksManager._SUPPORTED_MODEL_TYPE["minicpm"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]
TasksManager._SUPPORTED_MODEL_TYPE["qwen2"] = TasksManager._SUPPORTED_MODEL_TYPE["llama"]


@register_in_tasks_manager('phi', *["text-generation", "text-generation-with-past"])
class PhiOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig


class GemmaDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
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
        self.head_dim = normalized_config.head_dim
        self.num_kv_heads = normalized_config.num_key_value_heads

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = (
            self.batch_size,
            self.num_kv_heads,
            self.sequence_length,
            self.head_dim,
        )
        return [
            (
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
            )
            for _ in range(self.num_layers)
        ]


@register_in_tasks_manager("gemma", *["text-generation", "text-generation-with-past"])
class GemmaOnnxConfig(TextDecoderWithPositionIdsOnnxConfig):
    DEFAULT_ONNX_OPSET = 14
    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        GemmaDummyPastKeyValuesGenerator,
    )
    DUMMY_PKV_GENERATOR_CLASS = GemmaDummyPastKeyValuesGenerator
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    no_position_ids = False
