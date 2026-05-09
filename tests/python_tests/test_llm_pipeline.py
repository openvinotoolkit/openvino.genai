# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Literal, Callable
from pydantic import BaseModel, Field
from unittest.mock import MagicMock
from optimum.intel.utils.import_utils import is_transformers_version

import openvino as ov
import openvino_genai as ov_genai

from utils.constants import extra_generate_kwargs
from utils.hugging_face import generation_config_to_hf, download_and_convert_model, OVConvertedModelSchema

# model_tmp_path fixture import is required so it could be triggered by pytest
from utils.tokenizers import (
    delete_rt_info,
    model_tmp_path,  # noqa: F401
)
from utils.ov_genai_pipelines import (
    ALL_PIPELINE_TYPES,
    LINEAR_ATTENTION_PIPELINE_TYPES,
    create_ov_pipeline,
    generate_and_compare,
    MAIN_PIPELINE_TYPES,
    PipelineType,
    GenerationChatInputsType,
)
from data.models import get_models_list, CHAT_MODELS_LIST, LINEAR_ATTENTION_MODELS_LIST
from utils.custom_op import assert_ir_contains_op_type, get_extension_model, get_extension_lib_path, CustomAdd


def assert_hf_equals_genai(hf_reference, genai_output, **kwargs) -> None:
    __tracebackhide__ = True
    assert hf_reference == genai_output, f"HF reference:\n{hf_reference}\nGenAI output:\n{genai_output}" + (
        f"\nAdditional info: {json.dumps(kwargs, indent=4)}" if kwargs else ""
    )


#
# e2e work
#

GREEDY_INPUTS_TEST_CASES = [
    (
        {"max_new_tokens": 20},
        "你好！ 你好嗎？",
    ),
]

if is_transformers_version("<", "5.0"):
    # beam search fails with optimum-intel 423b423 and transformers>=5.0
    # restore after fix of CVS-185790
    INPUTS_TEST_CASES = [
        (
            {
                "max_new_tokens": 30,
                "num_beams": 15,
                "num_beam_groups": 3,
                "num_return_sequences": 15,
                "diversity_penalty": 1.0,
            },
            "Why is the Sun yellow?",
        ),
    ]
else:
    INPUTS_TEST_CASES = [*GREEDY_INPUTS_TEST_CASES]

PERF_METRICS_TEST_CASES = [
    ({"max_new_tokens": 20}, "table is made of"),
    ({"max_new_tokens": 20, "num_beams": 4}, "table is made of"),
]

PERF_METRICS_STRUCTURED_OUTPUT_TEST_CASES = [
    ({"max_new_tokens": 20}, "Generate json of a person"),
]

CUSTOM_EXTENSIONS_TEST_CASES = [
    ({"max_new_tokens": 20}, "Generate json of a person"),
]


INPUT_TENSORS_LIST = [
    # input_ids, attention_mask
    (np.array([[1, 4, 42]], dtype=np.int64), None),
    (np.array([[1, 4, 42]], dtype=np.int64), np.array([[1, 1, 1]], dtype=np.int64)),
]

GREEDY_TEST_CONFIGS = [{"max_new_tokens": 20}]
if is_transformers_version("<", "5.0"):
    # beam search fails with optimum-intel 423b423 and transformers>=5.0
    # restore after fix of CVS-185790
    TEST_CONFIGS = [
        {"max_new_tokens": 20, "num_beam_groups": 2, "num_beams": 6, "diversity_penalty": 1.0},
    ]
else:
    TEST_CONFIGS = [*GREEDY_TEST_CONFIGS]

BATCHED_PROMPTS = [
    ["table is made", "They sky is blue because", "Difference between Jupiter and Mars is that"],
    ["hello", "Here is the longest nowel ever: "],
    ["Alan Turing was a", "return 0", "你好！ 你好嗎？"],
    ["table is made", "table is made [force left pad tokens]"],
]

CHAT_INPUTS = []
if is_transformers_version("<", "5.0"):
    # beam search fails with optimum-intel 423b423 and transformers>=5.0
    # restore after fix of CVS-185790
    CHAT_INPUTS = [
        (
            {
                "max_new_tokens": 10,
                "num_beam_groups": 3,
                "num_beams": 15,
                "num_return_sequences": 1,
                "diversity_penalty": 1.0,
            },
            "",
        ),
    ]
else:
    CHAT_INPUTS = [
        ({"max_new_tokens": 20}, ""),
        ({"max_new_tokens": 20}, "Pretend that 1+1=1"),
    ]

MODELS_LIST = get_models_list()


QUESTIONS = ["1+1=", "What is the previous answer?", "Why is the Sun yellow?", "What was my first question?"]

CALLBACK_QUESTIONS = ["1+1=", "Why is the Sun yellow?", "What is the previous answer?", "What was my first question?"]


def user_defined_callback(subword):
    logging.info(subword)


def user_defined_status_callback(subword):
    logging.info(subword)
    return ov_genai.StreamingStatus.RUNNING


CALLBACK_FUNCTIONS = [
    logging.info,
    user_defined_callback,
    user_defined_status_callback,
    lambda subword: logging.info(subword),
]


@pytest.fixture(scope="module")
def llm_model(request: pytest.FixtureRequest) -> OVConvertedModelSchema:
    return download_and_convert_model(request.param)


@pytest.fixture(scope="module")
def ov_pipe(llm_model: OVConvertedModelSchema) -> ov_genai.LLMPipeline:
    return create_ov_pipeline(llm_model.models_path)


@pytest.mark.transformers_dependent(
    reason="Some cases with beam search fails with optimum-intel 423b423 and transformers>=5.0, CVS-185790"
)
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("generation_config_dict,prompt", INPUTS_TEST_CASES)
@pytest.mark.parametrize("pipeline_type", MAIN_PIPELINE_TYPES)
def test_string_inputs(
    llm_model: OVConvertedModelSchema,
    generation_config_dict: dict,
    prompt: str,
    pipeline_type: PipelineType,
) -> None:
    generate_and_compare(
        model_schema=llm_model,
        prompts=[prompt],
        generation_config=generation_config_dict,
        pipeline_type=pipeline_type,
    )


@pytest.mark.transformers_dependent(
    reason="qwen3_next is not supported by optimum-intel 423b423 with transformers>=5.0"
)
@pytest.mark.parametrize("llm_model", LINEAR_ATTENTION_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("generation_config_dict,prompt", GREEDY_INPUTS_TEST_CASES)  # exclude beam search case
@pytest.mark.parametrize("pipeline_type", LINEAR_ATTENTION_PIPELINE_TYPES)
def test_linear_attention_string_inputs(
    llm_model: OVConvertedModelSchema,
    generation_config_dict: dict,
    prompt: str,
    pipeline_type: PipelineType,
) -> None:
    generate_and_compare(
        model_schema=llm_model,
        prompts=[prompt],
        generation_config=generation_config_dict,
        pipeline_type=pipeline_type,
    )


ENCODED_INPUTS_MODELS_LIST = [*LINEAR_ATTENTION_MODELS_LIST]
if is_transformers_version(">=", "5.0"):
    # LINEAR_ATTENTION_MODELS_LIST depends on the tranformers version, but MODELS_LIST is the same
    # to eliminate duplication of tests, MODELS_LIST will be added for transformers>=5.0 only
    ENCODED_INPUTS_MODELS_LIST += MODELS_LIST


@pytest.mark.transformers_dependent(
    reason="qwen3_next is not supported by optimum-intel 423b423 with transformers>=5.0"
)
@pytest.mark.parametrize("llm_model", ENCODED_INPUTS_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("inputs", INPUT_TENSORS_LIST)
def test_encoded_inputs(
    llm_model: OVConvertedModelSchema,
    ov_pipe: ov_genai.LLMPipeline,
    inputs: tuple[np.ndarray, np.ndarray | None],
) -> None:
    ov_generation_config = ov_genai.GenerationConfig(max_new_tokens=20)
    hf_generation_config = generation_config_to_hf(llm_model.opt_model.generation_config, ov_generation_config)

    input_ids, attention_mask = inputs
    prompt_len = input_ids.shape[1]

    if attention_mask is not None:
        inputs_ov = ov_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        inputs_hf = {"inputs": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)}
    else:
        inputs_hf = {"inputs": torch.tensor(input_ids)}
        inputs_ov = ov.Tensor(input_ids)

    hf_output = llm_model.opt_model.generate(
        **inputs_hf, generation_config=hf_generation_config, **extra_generate_kwargs(hf_generation_config)
    ).sequences[0]
    ov_output = ov_pipe.generate(inputs_ov, ov_generation_config)

    hf_res = hf_output[prompt_len:].numpy()
    ov_res = np.array(ov_output.tokens, dtype=np.int64)
    assert np.all(ov_res == hf_res)


@pytest.mark.parametrize("llm_model", ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"], indirect=True)
def test_readonly_input_tensor(ov_pipe: ov_genai.LLMPipeline) -> None:
    input_ids = np.array([[1, 4, 42]], dtype=np.int64)
    input_ids.flags.writeable = False

    attention_mask = np.array([[1, 1, 1]], dtype=np.int64)
    attention_mask.flags.writeable = False

    inputs_ov = ov_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
    ov_pipe.generate(inputs_ov, max_new_tokens=5)

    readonly_tensor = ov.Tensor(input_ids)
    ov_pipe.generate(readonly_tensor, max_new_tokens=5)


@pytest.mark.transformers_dependent(
    reason="Some cases with beam search fails with optimum-intel 423b423 and transformers>=5.0, CVS-185790"
)
@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("generation_config_dict", TEST_CONFIGS)
@pytest.mark.parametrize("prompts", BATCHED_PROMPTS)
@pytest.mark.parametrize("pipeline_type", MAIN_PIPELINE_TYPES)
def test_batch_string_inputs(
    llm_model: OVConvertedModelSchema,
    generation_config_dict: dict,
    prompts: list[str],
    pipeline_type: PipelineType,
) -> None:
    generate_and_compare(
        model_schema=llm_model,
        prompts=prompts,
        generation_config=generation_config_dict,
        pipeline_type=pipeline_type,
    )


@pytest.mark.transformers_dependent(
    reason="qwen3_next is not supported by optimum-intel 423b423 with transformers>=5.0"
)
@pytest.mark.parametrize("llm_model", LINEAR_ATTENTION_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("pipeline_type", LINEAR_ATTENTION_PIPELINE_TYPES)
@pytest.mark.parametrize("generation_config_dict", GREEDY_TEST_CONFIGS)  # exclude beam search config
@pytest.mark.parametrize("prompts", BATCHED_PROMPTS)
def test_linear_attention_batch_string_inputs(
    llm_model: OVConvertedModelSchema,
    generation_config_dict: dict,
    prompts: list[str],
    pipeline_type: PipelineType,
) -> None:
    generate_and_compare(
        model_schema=llm_model,
        prompts=prompts,
        generation_config=generation_config_dict,
        pipeline_type=pipeline_type,
    )


@pytest.mark.parametrize("llm_model", ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"], indirect=True)
def test_batch_size_switch(ov_pipe: ov_genai.LLMPipeline) -> None:
    ov_pipe.generate(["a"], max_new_tokens=2)
    ov_pipe.generate(["1", "2"], max_new_tokens=2)
    ov_pipe.generate(["a"], max_new_tokens=2)


@pytest.mark.parametrize("llm_model", ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"], indirect=True)
def test_empty_encoded_inputs_throw(ov_pipe: ov_genai.LLMPipeline) -> None:
    with pytest.raises(RuntimeError):
        ov_pipe.generate(ov.Tensor(np.array([[]], dtype=np.int64)), max_new_tokens=2)


@pytest.mark.parametrize("llm_model", CHAT_MODELS_LIST, indirect=True)
def test_different_input_types_works_same_and_change_nothing(
    llm_model: OVConvertedModelSchema,
    ov_pipe: ov_genai.LLMPipeline,
) -> None:
    ov_generation_config = ov_genai.GenerationConfig()
    ov_generation_config.max_new_tokens = 30
    ov_generation_config.apply_chat_template = False

    res_string_input_1 = ov_pipe.generate(QUESTIONS[0], generation_config=ov_generation_config)

    tokenizer = ov_pipe.get_tokenizer()
    ov_tokens = tokenizer.encode(QUESTIONS[0], add_special_tokens=True)
    res_encoded_input = ov_pipe.generate(ov_tokens, generation_config=ov_generation_config)
    res_encoded_input_str = llm_model.hf_tokenizer.decode(res_encoded_input.tokens[0], skip_special_tokens=True)

    assert res_string_input_1 == res_encoded_input_str

    res_string_input_2 = ov_pipe.generate(QUESTIONS[0], generation_config=ov_generation_config)

    assert res_string_input_1 == res_string_input_2


@pytest.mark.transformers_dependent(
    reason="qwen3_next is not supported by optimum-intel 423b423 with transformers>=5.0"
)
@pytest.mark.parametrize("llm_model", LINEAR_ATTENTION_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("pipeline_type", LINEAR_ATTENTION_PIPELINE_TYPES)
@pytest.mark.parametrize("prompt", [prompt for prompts in BATCHED_PROMPTS for prompt in prompts])
def test_linear_model_deterministic(
    llm_model: OVConvertedModelSchema, pipeline_type: PipelineType, prompt: str
) -> None:
    ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type=pipeline_type)
    config = ov_genai.GenerationConfig(max_new_tokens=20, apply_chat_template=False, do_sample=False)
    result1 = ov_pipe.generate(prompt, generation_config=config)
    result2 = ov_pipe.generate(prompt, generation_config=config)
    assert result1 == result2


@pytest.mark.transformers_dependent(
    reason="qwen3_next is not supported by optimum-intel 423b423 with transformers>=5.0"
)
@pytest.mark.parametrize("llm_model", LINEAR_ATTENTION_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("pipeline_type", LINEAR_ATTENTION_PIPELINE_TYPES)
def test_linear_attention_batch_input_same_as_individual(
    llm_model: OVConvertedModelSchema,
    pipeline_type: PipelineType,
) -> None:
    prompts = ["table is made", "They sky is blue because", "Difference between Jupiter and Mars is that"]
    generation_config = ov_genai.GenerationConfig(max_new_tokens=20)

    ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type=pipeline_type)

    batch_result = ov_pipe.generate(prompts, generation_config=generation_config)
    for i, prompt in enumerate(prompts):
        individual_result = ov_pipe.generate(prompt, generation_config=generation_config)
        assert batch_result.texts[i] == individual_result, (
            f"Idx: {i}\n"
            f"Batch result:      {batch_result.texts[i]}\n"
            f"Individual result: {individual_result}\n"
            f"Prompt: {prompt}"
        )


#
# Chat scenario
#
@pytest.mark.transformers_dependent(
    reason="Some cases with beam search fails with optimum-intel 423b423 and transformers>=5.0, CVS-185790"
)
@pytest.mark.parametrize("llm_model", CHAT_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("inputs", CHAT_INPUTS)
@pytest.mark.parametrize(
    "input_type",
    [GenerationChatInputsType.STRING, GenerationChatInputsType.ENCODED_INPUTS, GenerationChatInputsType.CHAT_HISTORY],
)
def test_chat_scenario(
    llm_model: OVConvertedModelSchema,
    inputs: tuple[dict, str],
    input_type: GenerationChatInputsType,
) -> None:
    chat_history_hf = []
    chat_history_ov = ov_genai.ChatHistory() if input_type == GenerationChatInputsType.CHAT_HISTORY else []

    if input_type == GenerationChatInputsType.ENCODED_INPUTS:
        # chat is not supported for PA backend with encoded_inputs format
        ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type=PipelineType.STATEFUL)
    else:
        ov_pipe = create_ov_pipeline(llm_model.models_path)

    generation_config_kwargs, system_message = inputs

    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(llm_model.opt_model.generation_config, ov_generation_config)

    prev_chat_len = 0

    ov_pipe.start_chat(system_message)
    chat_history_hf.append({"role": "system", "content": system_message})
    chat_history_ov.append({"role": "system", "content": system_message})

    for prompt in QUESTIONS:
        chat_history_hf.append({"role": "user", "content": prompt})
        chat_history_ov.append({"role": "user", "content": prompt})

        chat_prompt = llm_model.hf_tokenizer.apply_chat_template(
            chat_history_hf, tokenize=False, add_generation_prompt=True
        )
        tokenized = llm_model.hf_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = tokenized["input_ids"].numel()

        answer = llm_model.opt_model.generate(
            **tokenized, generation_config=hf_generation_config, **extra_generate_kwargs(hf_generation_config)
        ).sequences[0]
        answer_str = llm_model.hf_tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
        chat_history_hf.append({"role": "assistant", "content": answer_str})

        if input_type == GenerationChatInputsType.STRING:
            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
        elif input_type == GenerationChatInputsType.CHAT_HISTORY:
            result_ov: ov_genai.DecodedResults = ov_pipe.generate(
                chat_history_ov, generation_config=ov_generation_config
            )
            answer_ov = result_ov.texts[0]
        elif input_type == GenerationChatInputsType.ENCODED_INPUTS:
            input_ids = np.array([tokenized["input_ids"][0][prev_chat_len:]], dtype=np.int64)
            attention_mask = np.array([tokenized["attention_mask"][0][prev_chat_len:]], dtype=np.int64)
            inputs_ov = ov_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))

            result_ov = ov_pipe.generate(inputs_ov, generation_config=ov_generation_config).tokens[0]

            answer_ov = llm_model.hf_tokenizer.decode(result_ov, skip_special_tokens=True)
            prev_chat_len = len(tokenized["input_ids"][0]) + len(result_ov)

        chat_history_ov.append({"role": "assistant", "content": answer_ov})

    ov_pipe.finish_chat()

    chat_history_messages_ov = (
        chat_history_ov.get_messages() if input_type == GenerationChatInputsType.CHAT_HISTORY else chat_history_ov
    )
    assert_hf_equals_genai(chat_history_hf, chat_history_messages_ov)

    # Test chat history generate without start_chat/finish_chat matches the same chat scenario
    if input_type == GenerationChatInputsType.CHAT_HISTORY:
        chat_history_ov = ov_genai.ChatHistory()
        chat_history_ov.append({"role": "system", "content": system_message})

        for prompt in QUESTIONS:
            chat_history_ov.append({"role": "user", "content": prompt})
            result_ov: ov_genai.DecodedResults = ov_pipe.generate(
                chat_history_ov, generation_config=ov_generation_config
            )
            answer_ov = result_ov.texts[0]
            chat_history_ov.append({"role": "assistant", "content": answer_ov})

        chat_history_messages_ov = chat_history_ov.get_messages()
        assert_hf_equals_genai(chat_history_hf, chat_history_messages_ov)


@pytest.mark.transformers_dependent(
    reason="qwen3_next is not supported by optimum-intel 423b423 with transformers>=5.0"
)
@pytest.mark.parametrize("llm_model", LINEAR_ATTENTION_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("inputs", CHAT_INPUTS[:1])  # exclude beam search config
@pytest.mark.parametrize(
    "input_type",
    [GenerationChatInputsType.ENCODED_INPUTS, GenerationChatInputsType.CHAT_HISTORY],
)
def test_linear_attention_chat_scenario(
    llm_model: OVConvertedModelSchema,
    inputs: tuple[dict, str],
    input_type: GenerationChatInputsType,
) -> None:
    chat_history_hf = []
    chat_history_ov = ov_genai.ChatHistory() if input_type == GenerationChatInputsType.CHAT_HISTORY else []

    ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type=PipelineType.STATEFUL)
    generation_config_kwargs, system_message = inputs

    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(llm_model.opt_model.generation_config, ov_generation_config)

    chat_history_hf.append({"role": "system", "content": system_message})
    chat_history_ov.append({"role": "system", "content": system_message})

    for prompt in QUESTIONS:
        chat_history_hf.append({"role": "user", "content": prompt})
        chat_history_ov.append({"role": "user", "content": prompt})

        chat_prompt = llm_model.hf_tokenizer.apply_chat_template(
            chat_history_hf, tokenize=False, add_generation_prompt=True
        )
        tokenized = llm_model.hf_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = tokenized["input_ids"].numel()

        answer = llm_model.opt_model.generate(
            **tokenized, generation_config=hf_generation_config, **extra_generate_kwargs(hf_generation_config)
        ).sequences[0]
        answer_str = llm_model.hf_tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
        chat_history_hf.append({"role": "assistant", "content": answer_str})

        if input_type == GenerationChatInputsType.CHAT_HISTORY:
            result_ov: ov_genai.DecodedResults = ov_pipe.generate(
                chat_history_ov, generation_config=ov_generation_config
            )
            answer_ov = result_ov.texts[0]
        elif input_type == GenerationChatInputsType.ENCODED_INPUTS:
            input_ids = np.array([tokenized["input_ids"][0]], dtype=np.int64)
            attention_mask = np.array([tokenized["attention_mask"][0]], dtype=np.int64)
            inputs_ov = ov_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))

            result_ov = ov_pipe.generate(inputs_ov, generation_config=ov_generation_config).tokens[0]
            answer_ov = llm_model.hf_tokenizer.decode(result_ov, skip_special_tokens=True)

        chat_history_ov.append({"role": "assistant", "content": answer_ov})

    chat_history_messages_ov = (
        chat_history_ov.get_messages() if input_type == GenerationChatInputsType.CHAT_HISTORY else chat_history_ov
    )
    assert_hf_equals_genai(chat_history_hf, chat_history_messages_ov)


@pytest.mark.parametrize("llm_model", [CHAT_MODELS_LIST[0]], indirect=True)
def test_chat_scenario_several_chats_in_series(
    llm_model: OVConvertedModelSchema,
    ov_pipe: ov_genai.LLMPipeline,
) -> None:
    generation_config_kwargs, _ = CHAT_INPUTS[0]
    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(llm_model.opt_model.generation_config, ov_generation_config)

    for i in range(2):
        chat_history_hf = []
        chat_history_ov = []
        ov_pipe.start_chat()
        for prompt in QUESTIONS[:2]:
            chat_history_hf.append({"role": "user", "content": prompt})
            chat_history_ov.append({"role": "user", "content": prompt})

            chat_prompt = llm_model.hf_tokenizer.apply_chat_template(
                chat_history_hf, tokenize=False, add_generation_prompt=True
            )
            tokenized = llm_model.hf_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
            prompt_len = tokenized["input_ids"].numel()

            answer = llm_model.opt_model.generate(
                **tokenized, generation_config=hf_generation_config, **extra_generate_kwargs(hf_generation_config)
            ).sequences[0]
            answer_str = llm_model.hf_tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
            chat_history_hf.append({"role": "assistant", "content": answer_str})

            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
            chat_history_ov.append({"role": "assistant", "content": answer_ov})

        ov_pipe.finish_chat()

        assert_hf_equals_genai(chat_history_hf, chat_history_ov)


@pytest.mark.transformers_dependent(
    reason="qwen3_next is not supported by optimum-intel 423b423 with transformers>=5.0"
)
@pytest.mark.parametrize("llm_model", LINEAR_ATTENTION_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("pipeline_type", LINEAR_ATTENTION_PIPELINE_TYPES)
def test_chat_scenario_several_chats_in_series_linear_cache(
    llm_model: OVConvertedModelSchema,
    pipeline_type: PipelineType,
) -> None:
    ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type=pipeline_type)
    generation_config_kwargs, _ = CHAT_INPUTS[0]
    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(llm_model.opt_model.generation_config, ov_generation_config)

    for i in range(2):
        chat_history_hf = []
        chat_history_ov = []
        ov_pipe.start_chat()
        for prompt in QUESTIONS[:2]:
            chat_history_hf.append({"role": "user", "content": prompt})
            chat_history_ov.append({"role": "user", "content": prompt})

            chat_prompt = llm_model.hf_tokenizer.apply_chat_template(
                chat_history_hf, tokenize=False, add_generation_prompt=True
            )
            tokenized = llm_model.hf_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
            prompt_len = tokenized["input_ids"].numel()

            answer = llm_model.opt_model.generate(
                **tokenized, generation_config=hf_generation_config, **extra_generate_kwargs(hf_generation_config)
            ).sequences[0]
            answer_str = llm_model.hf_tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
            chat_history_hf.append({"role": "assistant", "content": answer_str})

            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
            chat_history_ov.append({"role": "assistant", "content": answer_ov})

        ov_pipe.finish_chat()

        assert_hf_equals_genai(chat_history_hf, chat_history_ov, chat_number=i)


@pytest.mark.parametrize("llm_model", CHAT_MODELS_LIST, indirect=True)
def test_chat_scenario_several_start(ov_pipe: ov_genai.LLMPipeline) -> None:
    generation_config_kwargs, _ = CHAT_INPUTS[0]
    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)

    ov_pipe.start_chat()
    ov_pipe.start_chat()
    ov_pipe.generate(QUESTIONS[0], generation_config=ov_generation_config)
    ov_pipe.finish_chat()


@pytest.mark.parametrize("llm_model", CHAT_MODELS_LIST, indirect=True)
def test_generate_works_same_before_and_after_chat(ov_pipe: ov_genai.LLMPipeline) -> None:
    generation_config_kwargs, _ = CHAT_INPUTS[0]
    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    ov_generation_config.apply_chat_template = False

    res_before_chat = ov_pipe.generate(QUESTIONS[0], generation_config=ov_generation_config)

    ov_pipe.start_chat()
    ov_pipe.generate(QUESTIONS[0], generation_config=ov_generation_config)
    ov_pipe.finish_chat()

    res_after_chat = ov_pipe.generate(QUESTIONS[0], generation_config=ov_generation_config)

    assert res_after_chat == res_before_chat


@pytest.mark.transformers_dependent(
    reason="qwen3_next is not supported by optimum-intel 423b423 with transformers>=5.0"
)
@pytest.mark.parametrize("llm_model", LINEAR_ATTENTION_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("pipeline_type", LINEAR_ATTENTION_PIPELINE_TYPES)
@pytest.mark.parametrize("questions", [QUESTIONS[:2]])
def test_linear_attention_chat_matches_last_answer_after_cache_reset(llm_model, pipeline_type, questions):
    ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type=pipeline_type)

    config = ov_genai.GenerationConfig(max_new_tokens=20, do_sample=False)
    chat_history = ov_genai.ChatHistory()
    for question in questions:
        chat_history.append({"role": "user", "content": question})
        decoded_results: ov_genai.DecodedResults = ov_pipe.generate(chat_history, config)
        chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})

    last_answer = chat_history.pop()
    decoded_results: ov_genai.DecodedResults = ov_pipe.generate(chat_history, config)
    assert last_answer["content"] == decoded_results.texts[0]


#
# Streaming with callback
#


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("callback", CALLBACK_FUNCTIONS)
def test_callback_one_string(
    ov_pipe: ov_genai.LLMPipeline,
    callback: Callable,
) -> None:
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    ov_pipe.generate("table is made of", generation_config, callback)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("callback", CALLBACK_FUNCTIONS)
def test_callback_batch_throws(
    ov_pipe: ov_genai.LLMPipeline,
    callback: Callable,
) -> None:
    with pytest.raises(RuntimeError):
        ov_pipe.generate(["1", "2"], ov_pipe.get_generation_config(), callback)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("callback", CALLBACK_FUNCTIONS)
def test_callback_kwargs_one_string(
    ov_pipe: ov_genai.LLMPipeline,
    callback: Callable,
) -> None:
    ov_pipe.generate("table is made of", max_new_tokens=10, streamer=callback)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("callback", CALLBACK_FUNCTIONS)
def test_callback_decoding_metallama(
    llm_model: OVConvertedModelSchema,
    ov_pipe: ov_genai.LLMPipeline,
    callback: Callable,
) -> None:
    # On metallama this prompt generates output which can shorten after adding new tokens.
    # Test that streamer correctly handles such cases.
    prompt = "I have an interview about product speccing with the company Weekend Health. Give me an example of a question they might ask with regards about a new feature"
    if llm_model.model_id != "meta-llama/Meta-Llama-3-8B-Instruct":
        pytest.skip()
    ov_pipe.generate(prompt, max_new_tokens=300, streamer=callback)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("callback", CALLBACK_FUNCTIONS)
def test_callback_kwargs_batch_throws(
    ov_pipe: ov_genai.LLMPipeline,
    callback: Callable,
) -> None:
    with pytest.raises(RuntimeError):
        ov_pipe.generate(["1", "2"], max_new_tokens=10, streamer=callback)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
def test_callback_terminate_by_bool(ov_pipe: ov_genai.LLMPipeline) -> None:
    current_iter = 0
    num_iters = 10

    def callback(subword):
        nonlocal current_iter
        current_iter += 1
        return current_iter == num_iters

    max_new_tokens = 100
    ov_generation_config = ov_genai.GenerationConfig(max_new_tokens=max_new_tokens, ignore_eos=True)

    # without attention mask
    input_ids, _ = INPUT_TENSORS_LIST[0]
    inputs_ov = ov.Tensor(input_ids)
    ov_output = ov_pipe.generate(inputs_ov, ov_generation_config, streamer=callback)

    assert len(ov_output.tokens[0]) < max_new_tokens


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
def test_callback_terminate_by_status(ov_pipe: ov_genai.LLMPipeline) -> None:
    current_iter = 0
    num_iters = 10

    def callback(subword):
        nonlocal current_iter
        current_iter += 1
        return ov_genai.StreamingStatus.STOP if current_iter == num_iters else ov_genai.StreamingStatus.RUNNING

    max_new_tokens = 100
    ov_generation_config = ov_genai.GenerationConfig(max_new_tokens=max_new_tokens, ignore_eos=True)

    # without attention mask
    input_ids, _ = INPUT_TENSORS_LIST[0]
    inputs_ov = ov.Tensor(input_ids)
    ov_output = ov_pipe.generate(inputs_ov, ov_generation_config, streamer=callback)

    assert len(ov_output.tokens[0]) < max_new_tokens


@pytest.mark.parametrize("llm_model", CHAT_MODELS_LIST + LINEAR_ATTENTION_MODELS_LIST, indirect=True)
def test_chat_scenario_callback_cancel(
    llm_model: OVConvertedModelSchema,
    ov_pipe: ov_genai.LLMPipeline,
) -> None:
    generation_config_kwargs = {"max_new_tokens": 20}

    chat_history_hf = []
    chat_history_ov = []

    ov_generation_config = ov_genai.GenerationConfig(**generation_config_kwargs)
    hf_generation_config = generation_config_to_hf(llm_model.opt_model.generation_config, ov_generation_config)

    current_iter = 0
    num_iters = 3

    def callback(subword):
        nonlocal current_iter
        current_iter += 1
        return ov_genai.StreamingStatus.CANCEL if current_iter == num_iters else ov_genai.StreamingStatus.RUNNING

    ov_pipe.start_chat()
    for prompt in CALLBACK_QUESTIONS:
        if prompt != CALLBACK_QUESTIONS[1]:
            chat_history_hf.append({"role": "user", "content": prompt})
            chat_history_ov.append({"role": "user", "content": prompt})

            chat_prompt = llm_model.hf_tokenizer.apply_chat_template(
                chat_history_hf, tokenize=False, add_generation_prompt=True
            )
            tokenized = llm_model.hf_tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)
            prompt_len = tokenized["input_ids"].numel()

            answer = llm_model.opt_model.generate(
                **tokenized, generation_config=hf_generation_config, **extra_generate_kwargs(hf_generation_config)
            ).sequences[0]
            answer_str = llm_model.hf_tokenizer.decode(answer[prompt_len:], skip_special_tokens=True)
            chat_history_hf.append({"role": "assistant", "content": answer_str})

            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config)
            chat_history_ov.append({"role": "assistant", "content": answer_ov})
        else:
            answer_ov = ov_pipe.generate(prompt, generation_config=ov_generation_config, streamer=callback)

    ov_pipe.finish_chat()

    assert_hf_equals_genai(chat_history_hf, chat_history_ov)


class PrinterStatus(ov_genai.StreamerBase):
    def __init__(self, tokenizer):
        # super() may work, but once you begin mixing Python and C++
        # multiple inheritance, things will fall apart due to
        # differences between Python’s MRO and C++’s mechanisms.
        ov_genai.StreamerBase.__init__(self)
        self.tokenizer = tokenizer

    def write(self, token_id):
        # print(self.tokenizer.decode([token_id]))  # Incorrect way to print, but easy to implement
        print(token_id)  # print only token because self.tokenizer.decode([token_id]) are not implemented yet
        return ov_genai.StreamingStatus.RUNNING

    def end(self):
        print("end")


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("streamer_base", [PrinterStatus])
def test_streamer_one_string(
    ov_pipe: ov_genai.LLMPipeline,
    streamer_base: type,
) -> None:
    generation_config = ov_pipe.get_generation_config()
    generation_config.max_new_tokens = 10
    printer = streamer_base(ov_pipe.get_tokenizer())
    ov_pipe.generate("table is made of", generation_config, printer)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
def test_streamer_batch_throws(ov_pipe: ov_genai.LLMPipeline) -> None:
    printer = PrinterStatus(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe.generate(["1", "2"], ov_pipe.get_generation_config(), printer)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
def test_streamer_kwargs_one_string(ov_pipe: ov_genai.LLMPipeline) -> None:
    printer = PrinterStatus(ov_pipe.get_tokenizer())
    ov_pipe.generate("table is made of", max_new_tokens=10, do_sample=False, streamer=printer)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
def test_streamer_kwargs_batch_throws(ov_pipe: ov_genai.LLMPipeline) -> None:
    printer = PrinterStatus(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe.generate("", num_beams=2, streamer=printer)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("callback", CALLBACK_FUNCTIONS)
def test_operator_with_callback_one_string(
    ov_pipe: ov_genai.LLMPipeline,
    callback: Callable,
) -> None:
    ten_tokens = ov_pipe.get_generation_config()
    ten_tokens.max_new_tokens = 10
    ov_pipe("talbe is made of", ten_tokens, callback)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("callback", CALLBACK_FUNCTIONS)
def test_operator_with_callback_batch_throws(
    ov_pipe: ov_genai.LLMPipeline,
    callback: Callable,
) -> None:
    with pytest.raises(RuntimeError):
        ov_pipe(["1", "2"], ov_pipe.get_generation_config(), callback)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
@pytest.mark.parametrize("streamer_base", [PrinterStatus])
def test_operator_with_streamer_kwargs_one_string(
    ov_pipe: ov_genai.LLMPipeline,
    streamer_base: type,
) -> None:
    printer = streamer_base(ov_pipe.get_tokenizer())
    ov_pipe("hi", max_new_tokens=10, do_sample=True, streamer=printer)


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
def test_operator_with_streamer_kwargs_batch_throws(ov_pipe: ov_genai.LLMPipeline) -> None:
    printer = PrinterStatus(ov_pipe.get_tokenizer())
    with pytest.raises(RuntimeError):
        ov_pipe("", num_beams=2, streamer=printer)


#
# Tests on generation configs handling
#


def load_genai_pipe_with_configs(configs: list[tuple], temp_path):
    # Load LLMPipeline where all configs are cleared.
    # remove existing jsons from previous tests
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()
    delete_rt_info(configs, temp_path)

    for config_json, config_name in configs:
        with (temp_path / config_name).open("w", encoding="utf-8") as f:
            json.dump(config_json, f)

    ov_pipe = ov_genai.LLMPipeline(temp_path, "CPU")

    for _, config_name in configs:
        os.remove(temp_path / config_name)

    return ov_pipe


def test_eos_token_is_inherited_from_default_generation_config(model_tmp_path):  # noqa: F811
    _, temp_path = model_tmp_path
    ov_pipe = load_genai_pipe_with_configs([({"eos_token_id": 37}, "config.json")], temp_path)

    config = ov_genai.GenerationConfig()
    config.do_sample = True  # no eos_token_id but it's loaded from config.json
    ov_pipe.set_generation_config(config)

    assert 37 == ov_pipe.get_generation_config().eos_token_id


@pytest.mark.parametrize("llm_model", MODELS_LIST, indirect=True)
def test_pipeline_validates_generation_config(ov_pipe: ov_genai.LLMPipeline) -> None:
    invalid_generation_config = {
        "num_beam_groups": 3,
        "num_beams": 15,
        "do_sample": True,
    }  # beam sample is not supported
    with pytest.raises(RuntimeError):
        ov_pipe.generate("dummy prompt", **invalid_generation_config)


#
# Work with Unicode in Python API
#

# Model, prompt and max_new_tokens that generate unfinished utf-8 string
UNICODE_PYBIND_DECODING_TEST_CASES: list[tuple[str, str, int]] = [
    ("optimum-intel-internal-testing/tiny-random-PhiForCausalLM", ",", 3)
]


@pytest.mark.parametrize("llm_model,prompt,max_new_tokens", UNICODE_PYBIND_DECODING_TEST_CASES, indirect=["llm_model"])
def test_unicode_pybind_decoding_one_string(ov_pipe: ov_genai.LLMPipeline, prompt: str, max_new_tokens: int) -> None:
    res_str = ov_pipe.generate(prompt, max_new_tokens=max_new_tokens, apply_chat_template=False)
    assert "�" == res_str[-1]


@pytest.mark.parametrize("llm_model,prompt,max_new_tokens", UNICODE_PYBIND_DECODING_TEST_CASES, indirect=["llm_model"])
def test_unicode_pybind_decoding_batched(ov_pipe: ov_genai.LLMPipeline, prompt: str, max_new_tokens: int) -> None:
    res_str = ov_pipe.generate([prompt], max_new_tokens=max_new_tokens, apply_chat_template=False)
    assert "�" == res_str.texts[0][-1]


@pytest.mark.parametrize("llm_model,prompt,max_new_tokens", UNICODE_PYBIND_DECODING_TEST_CASES, indirect=["llm_model"])
def test_unicode_pybind_decoding_one_string_streamer(
    ov_pipe: ov_genai.LLMPipeline, prompt: str, max_new_tokens: int
) -> None:
    res_str = []
    ov_pipe.generate(
        prompt, max_new_tokens=max_new_tokens, apply_chat_template=False, streamer=lambda x: res_str.append(x)
    )
    assert "�" == "".join(res_str)[-1]


#
# Perf metrics
#


@pytest.mark.parametrize("llm_model", ["optimum-intel-internal-testing/tiny-random-gemma2"], indirect=True)
@pytest.mark.parametrize("generation_config,prompt", PERF_METRICS_TEST_CASES)
@pytest.mark.parametrize("pipeline_type", [PipelineType.STATEFUL, PipelineType.PAGED_ATTENTION])
def test_perf_metrics(
    llm_model: OVConvertedModelSchema,
    generation_config: dict,
    prompt: str,
    pipeline_type: PipelineType,
) -> None:
    import time

    start_time = time.perf_counter()
    ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type)
    load_time_in_test = (time.perf_counter() - start_time) * 1000
    start_generate = time.perf_counter()

    result = ov_pipe.generate([prompt], **generation_config)

    generate_time = (time.perf_counter() - start_generate) * 1000
    perf_metrics = result.perf_metrics
    is_beam_search = generation_config.get("num_beams", 1) > 1

    # Check that load time is adequate.
    load_time = perf_metrics.get_load_time()
    assert 0 < load_time < load_time_in_test

    # Check that num input and generated tokens are adequate.
    num_generated_tokens = perf_metrics.get_num_generated_tokens()
    assert 0 < num_generated_tokens <= generation_config["max_new_tokens"]

    num_input_tokens = perf_metrics.get_num_input_tokens()
    assert 0 < num_input_tokens <= len(prompt)

    mean_ttft, std_ttft = perf_metrics.get_ttft()
    assert (mean_ttft, std_ttft) == (perf_metrics.get_ttft().mean, perf_metrics.get_ttft().std)
    assert 0 < mean_ttft < generate_time

    raw_metrics = perf_metrics.raw_metrics
    durations = np.array(raw_metrics.m_durations) / 1000
    # Check that prefill is not included in durations for TPOT calculation.
    # For the very long prompt prefill is slow and TTFT is much larger than any other token generation duration.
    # For beam search TTFT is sometimes smaller than the duration of subsequent token generations (CVS-176478).
    if not is_beam_search:
        assert np.all(mean_ttft > durations)

    mean_tpot, std_tpot = perf_metrics.get_tpot()
    assert (mean_tpot, std_tpot) == (perf_metrics.get_tpot().mean, perf_metrics.get_tpot().std)
    assert 0 < mean_tpot < generate_time / num_generated_tokens

    mean_throughput, std_throughput = perf_metrics.get_throughput()
    assert (mean_throughput, std_throughput) == (perf_metrics.get_throughput().mean, perf_metrics.get_throughput().std)
    assert 0 < mean_throughput
    assert (num_generated_tokens - 1) / ((generate_time - mean_ttft) / 1000.0) < mean_throughput

    mean_gen_duration, std_gen_duration = perf_metrics.get_generate_duration()
    assert (mean_gen_duration, std_gen_duration) == (
        perf_metrics.get_generate_duration().mean,
        perf_metrics.get_generate_duration().std,
    )
    assert 0 < mean_gen_duration < generate_time
    assert std_gen_duration == 0

    mean_tok_duration, std_tok_duration = perf_metrics.get_tokenization_duration()
    assert (mean_tok_duration, std_tok_duration) == (
        perf_metrics.get_tokenization_duration().mean,
        perf_metrics.get_tokenization_duration().std,
    )
    assert 0 < mean_tok_duration < generate_time
    assert std_tok_duration == 0

    mean_detok_duration, std_detok_duration = perf_metrics.get_detokenization_duration()
    assert (mean_detok_duration, std_detok_duration) == (
        perf_metrics.get_detokenization_duration().mean,
        perf_metrics.get_detokenization_duration().std,
    )
    assert 0 < mean_detok_duration < generate_time
    assert std_detok_duration == 0

    # assert that calculating statistics manually from the raw counters we get the same restults as from PerfMetrics
    assert np.allclose(mean_tpot, np.mean(durations))
    assert np.allclose(std_tpot, np.std(durations))

    raw_dur = np.array(raw_metrics.generate_durations) / 1000
    assert np.allclose(mean_gen_duration, np.mean(raw_dur))
    assert np.allclose(std_gen_duration, np.std(raw_dur))

    raw_dur = np.array(raw_metrics.tokenization_durations) / 1000
    assert np.allclose(mean_tok_duration, np.mean(raw_dur))
    assert np.allclose(std_tok_duration, np.std(raw_dur))

    raw_dur = np.array(raw_metrics.detokenization_durations) / 1000
    assert np.allclose(mean_detok_duration, np.mean(raw_dur))
    assert np.allclose(std_detok_duration, np.std(raw_dur))

    assert len(raw_metrics.m_times_to_first_token) > 0
    assert len(raw_metrics.m_batch_sizes) > 0
    assert len(raw_metrics.m_durations) > 0
    assert len(raw_metrics.m_durations) == num_generated_tokens - 1


@pytest.mark.parametrize("llm_model", ["optimum-intel-internal-testing/tiny-random-gemma2"], indirect=True)
@pytest.mark.parametrize("generation_config,prompt", PERF_METRICS_STRUCTURED_OUTPUT_TEST_CASES)
def test_perf_metrics_with_structured_output(
    ov_pipe: ov_genai.LLMPipeline,
    generation_config: dict,
    prompt: str,
) -> None:
    class Person(BaseModel):
        name: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
        surname: str = Field(pattern=r"^[A-Z][a-z]{1,20}$")
        age: int
        city: Literal["Dublin", "Dubai", "Munich"]

    generation_config.update(
        {
            "structured_output_config": ov_genai.StructuredOutputConfig(
                json_schema=json.dumps(Person.model_json_schema())
            )
        }
    )
    perf_metrics = ov_pipe.generate([prompt], **generation_config).perf_metrics
    raw_metrics = perf_metrics.raw_metrics

    assert len(perf_metrics.get_grammar_compiler_init_times()) > 0
    assert (
        "xgrammar" in perf_metrics.get_grammar_compiler_init_times()
        and perf_metrics.get_grammar_compiler_init_times()["xgrammar"] > 0.0
    )

    assert len(raw_metrics.grammar_compile_times) > 0

    raw_compile_times = np.array(raw_metrics.grammar_compile_times) / 1000
    assert np.allclose(np.mean(raw_compile_times), perf_metrics.get_grammar_compile_time().mean)
    assert np.allclose(np.std(raw_compile_times), perf_metrics.get_grammar_compile_time().std)
    assert np.allclose(np.min(raw_compile_times), perf_metrics.get_grammar_compile_time().min)
    assert np.allclose(np.max(raw_compile_times), perf_metrics.get_grammar_compile_time().max)

    # Check that metrics are correctly accumulated/concatenated
    perf_metrics_2 = ov_pipe.generate([prompt], **generation_config).perf_metrics
    raw_metrics_2 = perf_metrics_2.raw_metrics
    accumulated_metrics = perf_metrics + perf_metrics_2
    assert (
        accumulated_metrics.raw_metrics.grammar_compile_times
        == raw_metrics.grammar_compile_times + raw_metrics_2.grammar_compile_times
    )


@pytest.mark.parametrize("llm_model", [CHAT_MODELS_LIST[0]], indirect=True)
def test_perf_metrics_with_apply_chat_template(ov_pipe: ov_genai.LLMPipeline) -> None:
    chat_history = [
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]

    metrics = [None] * len(chat_history)
    for i, msg in enumerate(chat_history):
        metrics[i] = ov_pipe.generate([msg], max_new_tokens=20, apply_chat_template=True).perf_metrics
    perf_metrics = sum(metrics[1:], start=metrics[0])

    raw_metrics = perf_metrics.raw_metrics

    # sanity check
    assert perf_metrics.get_generate_duration().mean > perf_metrics.get_chat_template_duration().mean

    raw_chat_template_duration = np.array(raw_metrics.chat_template_durations) / 1000
    assert np.allclose(np.mean(raw_chat_template_duration), perf_metrics.get_chat_template_duration().mean)
    assert np.allclose(np.std(raw_chat_template_duration), perf_metrics.get_chat_template_duration().std)


@pytest.mark.parametrize("llm_model", [CHAT_MODELS_LIST[0]], indirect=True)
def test_perf_metrics_without_apply_chat_template(ov_pipe: ov_genai.LLMPipeline) -> None:
    result = ov_pipe.generate(["What is the capital of France?"], max_new_tokens=20, apply_chat_template=False)

    assert result.perf_metrics.raw_metrics.chat_template_durations == []
    assert result.perf_metrics.get_chat_template_duration().mean == -1
    assert result.perf_metrics.get_chat_template_duration().std == -1


@pytest.mark.parametrize("llm_model", ["facebook/opt-125m"], indirect=True)
@pytest.mark.parametrize("pipeline_type", MAIN_PIPELINE_TYPES)
@pytest.mark.parametrize("stop_str", {True, False})
def test_pipelines_generate_with_streaming(
    llm_model: OVConvertedModelSchema,
    pipeline_type: PipelineType,
    stop_str: bool,
) -> None:
    mock_streamer = MagicMock(return_value=False)

    prompt = "Prompt example is"

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 10
    if stop_str:
        generation_config.stop_strings = {" the", "Prom"}
        generation_config.include_stop_str_in_output = False

    generate_and_compare(
        model_schema=llm_model,
        prompts=prompt,
        generation_config=generation_config,
        pipeline_type=pipeline_type,
        streamer=mock_streamer,
    )
    if stop_str:
        mock_streamer.assert_not_called()
    else:
        mock_streamer.assert_called()


@pytest.mark.parametrize("llm_model", ["katuni4ka/tiny-random-phi3"], indirect=True)
@pytest.mark.parametrize("generation_config,prompt", CUSTOM_EXTENSIONS_TEST_CASES)
def test_llm_pipeline_add_extension(
    llm_model: OVConvertedModelSchema,
    generation_config: dict,
    prompt: str,
    tmp_path: Path,
) -> None:
    myadd_model_path = get_extension_model(llm_model.models_path, tmp_path, "MyAdd")
    assert_ir_contains_op_type(myadd_model_path, "MyAdd")

    # The custom op "MyAdd" is provided by a compiled extension library and is intended to behave like OpenVINO "Add".
    properties = {"extensions": [str(get_extension_lib_path())]}
    ov_pipe_extension_path = ov_genai.LLMPipeline(myadd_model_path, "CPU", **properties)
    result_extension_path = ov_pipe_extension_path.generate([prompt], **generation_config)

    # The Python custom op "CustomAdd" is intended to behave like OpenVINO "Add", but it exercises Python ov.Op registration and callback-based evaluation.
    customadd_model_path = get_extension_model(llm_model.models_path, tmp_path, "CustomAdd")
    assert_ir_contains_op_type(customadd_model_path, "CustomAdd")
    CustomAdd.evaluate_calls = 0
    properties = {"extensions": [ov.OpExtension(CustomAdd)]}
    ov_pipe_extension_obj = ov_genai.LLMPipeline(customadd_model_path, "CPU", **properties)
    result_extension_obj = ov_pipe_extension_obj.generate([prompt], **generation_config)
    assert CustomAdd.evaluate_calls > 0, "Python custom op 'CustomAdd' was not called"

    # Reference result with the original model and without custom extensions.
    ov_pipe_ref = ov_genai.LLMPipeline(llm_model.models_path, "CPU")
    result_ref = ov_pipe_ref.generate([prompt], **generation_config)

    assert result_extension_path.texts[0].strip() == result_ref.texts[0].strip(), (
        "Result should be the same for model with extension 'MyAdd' and reference model."
    )
    assert result_extension_obj.texts[0].strip() == result_ref.texts[0].strip(), (
        "Result should be the same for model with extension 'CustomAdd' and reference model."
    )


# Speculative decoding and prompt lookup require extra GenerationConfig params;
# exclude them from generic RNG-seed tests.
_NON_ASSISTANT_PIPELINE_TYPES = tuple(
    pt
    for pt in ALL_PIPELINE_TYPES
    if pt not in (PipelineType.SPECULATIVE_DECODING, PipelineType.PROMPT_LOOKUP_DECODING)
)


def _extract_texts(result) -> list[str]:
    """Return a flat list of generated strings from either str, DecodedResults, or list[GenerationResult]."""
    if isinstance(result, str):
        # LLMPipeline.generate(str, ...) returns a plain str for STATEFUL/PA/AUTO pipelines
        return [result]
    if isinstance(result, list):
        # ContinuousBatchingPipeline.generate() returns list[GenerationResult]
        return [gen_id for r in result for gen_id in r.m_generation_ids]
    return list(result.texts)


@pytest.mark.parametrize("llm_model", ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"], indirect=True)
@pytest.mark.parametrize("pipeline_type", _NON_ASSISTANT_PIPELINE_TYPES)
def test_same_rng_seed_produces_identical_output(
    llm_model: OVConvertedModelSchema, pipeline_type: PipelineType
) -> None:
    """Two generate() calls with the same rng_seed must produce identical output text."""
    ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type=pipeline_type)
    config = ov_genai.GenerationConfig(do_sample=True, temperature=1.0, max_new_tokens=20, rng_seed=42)
    prompt = "Which season is better, summer or winter?"

    texts1 = _extract_texts(ov_pipe.generate(prompt, generation_config=config))
    texts2 = _extract_texts(ov_pipe.generate(prompt, generation_config=config))

    assert texts1 == texts2, (
        f"generate() with rng_seed=42 must be reproducible.\nFirst call:  {texts1}\nSecond call: {texts2}"
    )


@pytest.mark.parametrize("llm_model", ["optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM"], indirect=True)
@pytest.mark.parametrize("pipeline_type", _NON_ASSISTANT_PIPELINE_TYPES)
def test_different_rng_seed_produces_different_output(
    llm_model: OVConvertedModelSchema, pipeline_type: PipelineType
) -> None:
    """Different rng_seeds must produce at least one distinct output across multiple seeds."""
    ov_pipe = create_ov_pipeline(llm_model.models_path, pipeline_type=pipeline_type)
    rng_seeds = [42, 123, 777, 2024]
    prompt = "Which season is better, summer or winter?"

    text_tuples = [
        tuple(
            _extract_texts(
                ov_pipe.generate(
                    prompt,
                    generation_config=ov_genai.GenerationConfig(
                        do_sample=True, temperature=1.3, max_new_tokens=40, rng_seed=seed
                    ),
                )
            )
        )
        for seed in rng_seeds
    ]

    assert len(set(text_tuples)) > 1, (
        f"generate() with different rng_seeds {rng_seeds} must produce at least one distinct output, "
        f"but all produced: {text_tuples[0]!r}"
    )
