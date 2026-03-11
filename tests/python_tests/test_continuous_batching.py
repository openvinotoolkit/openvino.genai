# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import math
import sys

from pathlib import Path
from shutil import rmtree

from openvino_genai import ContinuousBatchingPipeline, LLMPipeline, GenerationConfig, SchedulerConfig, draft_model, GenerationFinishReason, ChatHistory

from test_sampling import RandomSamplingTestStruct, get_current_platform_ref_texts

from utils.generation_config import get_greedy, get_beam_search, \
    get_multinomial_all_parameters, get_multinomial_temperature_and_num_return_sequence, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature, get_multinomial_temperature_and_top_p
from utils.hugging_face import OVConvertedModelSchema, download_and_convert_model, run_hugging_face
from utils.ov_genai_pipelines import (
    create_ov_pipeline,
    create_ov_cb_pipeline,
    PipelineType,
    dict_to_scheduler_config,
    generate_and_compare,
    prepare_generation_config_by_pipe_type,
    convert_decoded_results_to_generation_result,
    GenerationChatInputsType,
)
from utils.comparation import compare_generation_results
from data.models import CHAT_MODELS_LIST
from data.test_dataset import get_test_dataset

#
# e2e tests on random and real models
#

FILE_DIR_NAME = Path(__file__).parent

COMMON_QUESTIONS = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?'
]


COMMON_QUESTIONS_SHORT = [
    '1+1=',
    'Why is the Sun yellow?',
]


def read_models_list(file_name: str) -> list[str]:
    models = []
    with open(file_name, encoding="utf-8") as f:
        for model_name in f:
            model_name = model_name.strip()
            # skip comment in model scope file
            if model_name.startswith('#'):
                continue
            models.append(model_name)
    return models

@pytest.fixture(scope="module")
def llm_model(request: pytest.FixtureRequest) -> OVConvertedModelSchema:
    return download_and_convert_model(request.param)


@pytest.fixture(scope="module")
def model_facebook_opt_125m() -> OVConvertedModelSchema:
    model_id : str = "facebook/opt-125m"
    return download_and_convert_model(model_id)


@pytest.mark.parametrize("llm_model", read_models_list(FILE_DIR_NAME / "models" / "lightweight"), indirect=True)
def test_e2e_lightweight_models(llm_model: OVConvertedModelSchema):
    prompts, generation_configs = get_test_dataset()
    generate_and_compare(
        model_schema=llm_model,
        prompts=prompts,
        generation_config=generation_configs,
        pipeline_type=PipelineType.CONTINUOUS_BATCHING,
    )


@pytest.mark.real_models
@pytest.mark.parametrize("llm_model", read_models_list(FILE_DIR_NAME / "models" / "real_models"), indirect=True)
def test_e2e_real_models(llm_model: OVConvertedModelSchema):
    prompts, generation_config = get_test_dataset()
    generate_and_compare(
        model_schema=llm_model,
        prompts=prompts,
        generation_config=generation_config,
        pipeline_type=PipelineType.CONTINUOUS_BATCHING,
    )

#
# Comparison with stateful
# TODO: remove these tests once test_llm_pipeline.py are generalized 
# and parametrized to test both Stateful and PA paths
@pytest.mark.parametrize(
    "generation_config", 
    [
        {"max_new_tokens": 20},
        {"max_new_tokens": 200, "ignore_eos": True},
        {"max_new_tokens": 20, "num_beam_groups": 3, "num_beams": 15, "diversity_penalty": 1.0},
    ]
)
@pytest.mark.parametrize(
    "prompt",
    [
        ['hello', 'Here is the longest nowel ever: '],
        ['Alan Turing was a', 'return 0', '你好！ 你好嗎？'],
        ['table is made', 'table is made [force left pad tokens]'],
    ]
)  # num_beams=15 diverges on the first prompt.
@pytest.mark.skip(reason="CVS-162891: Fix test_continuous_batching_vs_stateful tests after we started to compare cb vs sdpa")
def test_continuous_batching_vs_stateful(model_facebook_opt_125m: OVConvertedModelSchema, prompt, generation_config):
    models_path = model_facebook_opt_125m.models_path
    cb_pipe = create_ov_pipeline(models_path, pipeline_type=PipelineType.PAGED_ATTENTION)
    ov_pipe = create_ov_pipeline(models_path, pipeline_type=PipelineType.STATEFUL)

    generated = cb_pipe.generate(prompt, **generation_config)
    reference = ov_pipe.generate(prompt, **generation_config)

    assert generated.texts == reference.texts
    if 1 != generation_config.get("num_return_sequences", 1):
        # Stateful puts zeroes to generated.scores. Don't compare them.
        for gen, ref in zip(generated.scores, reference.scores):
            assert math.isclose(gen, ref, abs_tol=0.0003)


@pytest.mark.parametrize(
    "prompt", 
    [
        'The Sun is yellow because', 
        'Difference between Jupiter and Mars is that', 
        'table is made of'
    ]
)
def test_cb_streamer_vs_return_vs_stateful(model_facebook_opt_125m: OVConvertedModelSchema, prompt: str):
    models_path = model_facebook_opt_125m.models_path

    ov_pipe = create_ov_pipeline(models_path, pipeline_type=PipelineType.STATEFUL)
    cb_pipe = create_ov_pipeline(models_path, pipeline_type=PipelineType.PAGED_ATTENTION)

    streamed = []
    generated = cb_pipe.generate(prompt, max_new_tokens=20, streamer=lambda subword: streamed.append(subword))
    reference = ov_pipe.generate(prompt, max_new_tokens=20)
    assert generated == "".join(streamed)
    assert "".join(streamed) == reference


@pytest.mark.parametrize(
    "generation_config_kwargs", 
    [
        {
            "do_sample": False, 
            "num_beam_groups": 3, 
            "num_beams": 15, 
            "num_return_sequences": 1, 
            "max_new_tokens": 10, 
            "diversity_penalty": 1.0, 
            "repetition_penalty": 1.0,
        }
    ]
)
@pytest.mark.parametrize("llm_model", CHAT_MODELS_LIST, indirect=True)
@pytest.mark.parametrize(
    "pipeline_type", 
    [
        PipelineType.PAGED_ATTENTION, 
        PipelineType.PROMPT_LOOKUP_DECODING, 
        PipelineType.SPECULATIVE_DECODING,
    ] 
)
@pytest.mark.parametrize("input_type", [
    GenerationChatInputsType.STRING,
    GenerationChatInputsType.CHAT_HISTORY])
def test_chat_scenario_vs_stateful(
    llm_model: OVConvertedModelSchema, 
    generation_config_kwargs: dict, 
    pipeline_type: PipelineType,
    input_type: GenerationChatInputsType
):
    models_path = llm_model.models_path

    ov_pipe = create_ov_pipeline(models_path, pipeline_type=PipelineType.STATEFUL)
    cb_pipe = create_ov_pipeline(models_path, pipeline_type=pipeline_type)

    generation_config = GenerationConfig(**generation_config_kwargs)
    # assisted generation is not supported for beam search
    if generation_config.is_beam_search() and pipeline_type != PipelineType.PAGED_ATTENTION:
        return

    generation_config = prepare_generation_config_by_pipe_type(
        generation_config=generation_config, 
        pipeline_type=pipeline_type,
    )

    ov_pipe.set_generation_config(generation_config)
    
    if input_type == GenerationChatInputsType.STRING:
        ov_pipe.start_chat()
        cb_pipe.start_chat()
    
        for question in COMMON_QUESTIONS:
            generated = cb_pipe.generate(question, generation_config=generation_config)
            reference = ov_pipe.generate(question)
            assert generated == reference

        # Test that finish_chat() doesn't fail just in case.
        cb_pipe.finish_chat()
    elif input_type == GenerationChatInputsType.CHAT_HISTORY:
        chat_history = ChatHistory()
        for question in COMMON_QUESTIONS:
            chat_history.append({"role": "user", "content": question})
            cb_decoded_results = cb_pipe.generate(chat_history, generation_config=generation_config)
            generated = cb_decoded_results.texts[0]
            stateful_decoded_results = ov_pipe.generate(chat_history)
            reference = stateful_decoded_results.texts[0]
            chat_history.append({"role": "assistant", "content": generated})
            assert generated == reference


@pytest.mark.parametrize("llm_model", CHAT_MODELS_LIST, indirect=True)
@pytest.mark.parametrize(
    "generation_config_kwargs",
    [
        {"do_sample": False, "max_new_tokens": 20},
        {"do_sample": True, "max_new_tokens": 20, "temperature": 0.7},
        {"do_sample": False, "num_beam_groups": 3, "num_beams": 15, "num_return_sequences": 1, "max_new_tokens": 10, "diversity_penalty": 1.0, "repetition_penalty": 1.0},
    ]
)
@pytest.mark.parametrize(
    "pipeline_type", 
    [
        PipelineType.CONTINUOUS_BATCHING, 
        PipelineType.SPECULATIVE_DECODING, 
        PipelineType.PROMPT_LOOKUP_DECODING,
    ]
)
def test_continuous_batching_add_request_health_check(
    llm_model: OVConvertedModelSchema, 
    generation_config_kwargs: dict, 
    pipeline_type: PipelineType
):
    models_path = llm_model.models_path

    cb_pipe = create_ov_cb_pipeline(models_path, pipeline_type=pipeline_type)

    generation_config = GenerationConfig(**generation_config_kwargs)

    if generation_config.is_beam_search() and pipeline_type != PipelineType.CONTINUOUS_BATCHING:
        pytest.skip("Assisted generation does not support beam search")

    generation_config = prepare_generation_config_by_pipe_type(generation_config=generation_config, pipeline_type=pipeline_type)
    handles = []
    for idx, question in enumerate(COMMON_QUESTIONS_SHORT):
        handle = cb_pipe.add_request(idx, question, generation_config=generation_config)
        handles.append(handle)

    while cb_pipe.has_non_finished_requests():
        cb_pipe.step()
        
    for handle in handles:
        outputs = handle.read_all()
        for output in outputs:
            assert output.finish_reason == GenerationFinishReason.STOP or output.finish_reason == GenerationFinishReason.LENGTH

@pytest.mark.parametrize(
    "generation_config_kwargs", 
    [
        {"max_length": 1, "ignore_eos": True}, # max_length smaller than number of prompt tokens, generation should stop right away
    ]
)
@pytest.mark.parametrize("llm_model", CHAT_MODELS_LIST, indirect=True)
@pytest.mark.parametrize("pipeline_type", [PipelineType.CONTINUOUS_BATCHING, PipelineType.SPECULATIVE_DECODING, PipelineType.PROMPT_LOOKUP_DECODING,])
def test_continuous_batching_add_request_fails(
    llm_model: OVConvertedModelSchema, 
    generation_config_kwargs: dict, 
    pipeline_type: PipelineType,
):
    models_path = llm_model.models_path

    cb_pipe = create_ov_cb_pipeline(models_path, pipeline_type=pipeline_type)

    generation_config = GenerationConfig(**generation_config_kwargs)

    if generation_config.is_beam_search() and pipeline_type != PipelineType.CONTINUOUS_BATCHING:
        pytest.skip("Assisted generation does not support beam search")

    generation_config = prepare_generation_config_by_pipe_type(
        generation_config=generation_config, pipeline_type=pipeline_type
    )
    for idx, question in enumerate(COMMON_QUESTIONS_SHORT):
        with pytest.raises(RuntimeError):
            cb_pipe.add_request(idx, question, generation_config=generation_config)

#
# Stress tests to check OOM case
#

# todo: iefode: bug reproducer!!!
@pytest.mark.parametrize(
    "sampling_config", 
    [get_greedy(), get_beam_search(), get_multinomial_all_parameters()],
    ids=["greedy", "beam_search", "multinomial_all_parameters"]
)
def test_post_oom_health(model_facebook_opt_125m: OVConvertedModelSchema, sampling_config):
    generation_config = sampling_config
    generation_config.ignore_eos = True
    generation_config.max_new_tokens = 1000000

    scheduler_config = dict_to_scheduler_config()
    scheduler_config.num_kv_blocks = 10 # Low cache size to trigger OOM quickly

    models_path = model_facebook_opt_125m.models_path

    cb_pipe = create_ov_pipeline(
        models_path,
        pipeline_type=PipelineType.CONTINUOUS_BATCHING,
        device="CPU",
        scheduler_config=scheduler_config,
    )

    # First run should return incomplete response
    output = cb_pipe.generate(["What is OpenVINO?"], [generation_config])
    assert (len(output))
    assert (len(output[0].m_generation_ids))

    # Same for the second run, here we want to make sure the cleanup works and we have free blocks after recent OOM
    output = cb_pipe.generate(["What is OpenVINO?"], [generation_config])
    assert (len(output))
    assert (len(output[0].m_generation_ids))

#
# Pre-emption
#

def get_parallel_sampling_seq_len_300() -> GenerationConfig:
    generation_config = GenerationConfig()
    # TODO: add generation_config.generator and return parameters below
    # generation_config.num_return_sequences = 3
    # generation_config.do_sample = True
    # generation_config.top_k = 10
    # generation_config.top_p = 0.5
    generation_config.max_new_tokens = 300
    return generation_config


def get_beam_search_seq_len_300() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.diversity_penalty = 1
    generation_config.max_new_tokens = 300
    generation_config.num_return_sequences = generation_config.num_beams
    return generation_config


@pytest.mark.parametrize(
    "params", 
    [
        ({"num_kv_blocks": 2, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
        ({"num_kv_blocks": 2, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
        ({"num_kv_blocks": 10, "dynamic_split_fuse": True}, get_parallel_sampling_seq_len_300()),
        ({"num_kv_blocks": 10, "dynamic_split_fuse": False}, get_parallel_sampling_seq_len_300()),
        ({"num_kv_blocks": 34, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search()),
        ({"num_kv_blocks": 34, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search()),
        ({"num_kv_blocks": 100, "dynamic_split_fuse": True}, get_beam_search_seq_len_300()),
        ({"num_kv_blocks": 100, "dynamic_split_fuse": False}, get_beam_search_seq_len_300()),
    ]
)
def test_preemption(model_facebook_opt_125m: OVConvertedModelSchema, params):
    scheduler_params = params[0]
    generation_config = params[1]

    prompts, _ = get_test_dataset()
    generate_and_compare(
        model_schema=model_facebook_opt_125m,
        prompts=prompts,
        pipeline_type=PipelineType.CONTINUOUS_BATCHING,
        scheduler_config=scheduler_params,
        generation_config=generation_config
    )

multinomial_params = RandomSamplingTestStruct(
    generation_config=[
        get_multinomial_temperature(),
        get_multinomial_temperature_and_top_p(),
        get_multinomial_temperature_and_top_k(),
    ],
    prompts=[
        "What is OpenVINO?",
        "How are you?",
        "Tell me something about Canada?",
    ],
    ref_texts=get_current_platform_ref_texts({
        "linux": [
            [
                "\n\nOpenVINO is a live platform that allows users to create and manage a new library for open source applications.\n\nOpenVINO is"
            ],
            [
                "  You're getting much better results from doing this, than you are by not doing this.  I have a BH and I was so far"
            ],
            [
                "\nI'm from Canada, and I'm from the US, so I'm not sure.\nI think you mean the Canadian version."
            ],
        ],
        "win32": [
            [
                "\n\nOpenVINO is a live platform that allows users to create and manage a new library of applications on the Virtuoso server, which can"
            ],
            [
                "  You're getting much better results from doing this, than you are by not doing this.  If you are truly trying to do something good,"
            ],
            [
                "\nI'm from Canada, and I'm from the US, so I'm not sure what you're talking about.\nI'm Canadian and I"
            ],
        ],
    }),
)


# todo: Anastasiia Pnevskaya: fix the test because it is hanging according max_new_tokens = std::numeric_limits<std::size_t>::max()
@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.skip(
    reason=(
        "Random sampling results are non deterministic due to: discrete_distribution impl depends on platform, "
        "model inference results may depend on CPU. Test passes on CI but fails locally."
    )
)
def test_preemption_with_multinomial(model_facebook_opt_125m: OVConvertedModelSchema, dynamic_split_fuse):
    generation_configs = multinomial_params.generation_config
    for config in generation_configs:
        config.max_new_tokens = 30

    scheduler_config = dict_to_scheduler_config({"num_kv_blocks": 3, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare(
        model_schema=model_facebook_opt_125m,
        pipeline_type=PipelineType.CONTINUOUS_BATCHING,
        prompts=multinomial_params.prompts,
        ref=multinomial_params.ref_texts,
        generation_config=generation_configs,
        scheduler_config=scheduler_config,
    )


multinomial_params_n_seq = RandomSamplingTestStruct(
    generation_config=[
        get_multinomial_temperature(),
        get_multinomial_temperature_and_num_return_sequence(),
        get_multinomial_all_parameters(),
    ],
    prompts=[
        "Artificial intelligence ",
        "What is the current",
        "Tell me something about UAE?",
    ],
    ref_texts=get_current_platform_ref_texts({
        "linux": [
            [
                "\nI've seen this expression used too many times without making sense.\nAs an AI engineer, and as a scientist, we should make everything easier"
            ],
            [
                " position of the Z-shaped groove?\n0.41\nWhat is the current position of the Z-shaped groove?\n0.11\n",
                " status of all of this? I can't stop thinking about it.\nIt's been a while since I've seen it. I found it a",
                " status of your blog? Do you accept feedback?\nYes, I’m happy to accept feedback at this time (I’m a"
            ],
            [
                "\nIt's in the middle of nowhere if you haven’t seen one yet! It might be more convenient there than anywhere else.. maybe take",
                "\nUAE is a country with some great culture that has been living under Islamic oppression for almost 60 years now (including 20 years as part of Arab",
                "\nNope, just wanted to say how awesome and beautiful it was when my brother came back from an adventure trip across Asia - our 2nd year",
                "\nI don't know anything.  I'm not sure what kind this sub wants though... but apparently they are pretty bad at making videos/photos",
            ],
        ],
        "win32": [
            [
                "\nI've had a friend with the capacity to test this in his own words.\nThe big problem with real-world results is the economics of"
            ],
            [
                " position of the patent application number of the present invention?\n\nIn the present invention, the present invention relates to an improved method for manufacturing a semic",
                " status of your town? How many houses do you have?\nThere are about three houses in our town. The closest place to us is about 25",
                " status of all the other passengers?\nWe're the only ones left, so no...\nI don't think they'll really leave.\nThey"
            ],
            [
                "\nI don't have any knowledge on them. We are based out near Dubai so hopefully they will take care of us soon enough :) thanks though :",
                "\nUAE is not one of the richest countries in Asia but definitely among those most corrupt nations because this corruption (and its own endemic practices) still",
                "\nNope, I'm just going through my first semester there right now and it was nice to see some people who were doing well haha - we",
                "\nIt's a country where your parents can never give you anything at all!  It also has an extremely low education system for many years... You",
            ],
        ],
        "darwin": [
            [
                "\nI've had a friend with the capacity to test this in his own words.\nThe big problem with real-world results is the rigidity"
            ],
            [
               " position of the patent application number of the present invention?\n\nIn the present invention, the present invention relates to an improved method for manufacturing a semic",
               " status of your town? How many houses do you have?\nThere are about three houses in our town. The closest place to us is about 25",
               " status of all the other passengers?\nWe're the only ones left, so no...\nI don't think they'll really leave.\nThey"
            ],
            [
                "\nI don't have any knowledge on them. We are based out near Dubai so hopefully they will take care of us soon enough :) thanks though :",
                "\nUAE is not one of the richest countries in Asia but definitely among those most corrupt nations because this corruption (and its own endemic practices) still",
                "\nNope, I'm just going through my first semester there right now and it was nice to see some people who were doing well haha - we",
                "\nIt's a country where your parents can never give you anything at all!  It also has an extremely low education system for many years... You",
            ],
        ],
    }),
)


@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.skip(reason="Random sampling results are non deterministic due to: discrete_distribution impl depends on platform, model inference results may depend on CPU. Test passes on CI but fails locally.")
def test_preemption_with_multinomial_n_seq(model_facebook_opt_125m: OVConvertedModelSchema, dynamic_split_fuse):
    # needed kv_blocks - 16 (2 blocks per sequence (30 tokens to generated text + prompt (> 2 tokens)) * (1 + 3 + 4) seq )
    scheduler_config = dict_to_scheduler_config({"num_kv_blocks": 8, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare(
        model_schema=model_facebook_opt_125m,
        pipeline_type=PipelineType.CONTINUOUS_BATCHING,
        prompts=multinomial_params_n_seq.prompts,
        ref=multinomial_params_n_seq.ref_texts,
        generation_config=multinomial_params_n_seq.generation_config,
        scheduler_config=scheduler_config
    )


def test_dynamic_split_fuse_doesnt_affect_generated_text():
    model_id : str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    pipeline_type = PipelineType.PROMPT_LOOKUP_DECODING
    models_path = download_and_convert_model(model_id).models_path

    scheduler_config_ref = dict_to_scheduler_config({"dynamic_split_fuse": False, "max_num_batched_tokens": sys.maxsize})
    cb_pipe_ref = create_ov_pipeline(models_path, scheduler_config=scheduler_config_ref, pipeline_type=pipeline_type)

    scheduler_config_target = dict_to_scheduler_config({"dynamic_split_fuse": True, "max_num_batched_tokens": 5})
    cb_pipe_target = create_ov_pipeline(models_path, scheduler_config=scheduler_config_target, pipeline_type=pipeline_type)

    generation_config = GenerationConfig(do_sample=False, max_new_tokens=20, eos_token_id=cb_pipe_ref.get_tokenizer().get_eos_token_id())

    generation_config = prepare_generation_config_by_pipe_type(generation_config=generation_config, pipeline_type=pipeline_type)
    cb_pipe_ref.set_generation_config(generation_config)
    cb_pipe_target.set_generation_config(generation_config)

    question = "Why is the Sun yellow?"
    reference = cb_pipe_ref.generate(question, generation_config=generation_config)
    generated = cb_pipe_target.generate(question, generation_config=generation_config)
    assert generated == reference


eagle_models_and_input = [
    (
        "Qwen/Qwen3-1.7B",
        "AngelSlim/Qwen3-1.7B_eagle3",
        """Code:
def add(a, b):
    return a + b

Question: Can you please add 2 and 3
A:""",
    )
]

speculative_cases = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", None, "Why is the Sun yellow?"),
    eagle_models_and_input[0],
]

@pytest.mark.parametrize(
    "pipeline_type", 
    [
        PipelineType.PAGED_ATTENTION,
        PipelineType.SPECULATIVE_DECODING,
    ]
)
@pytest.mark.parametrize("main_model_id,draft_model_id, prompt", speculative_cases)
def test_speculative_decoding_extended_perf_metrics(pipeline_type: PipelineType, main_model_id, draft_model_id, prompt):
    def run_extended_perf_metrics_collection(
        model_id: str,
        generation_config: GenerationConfig,
        prompt: str,
        pipeline_type: PipelineType,
        draft_model_id: str,
    ):
        model_path = download_and_convert_model(model_id).models_path
        draft_model_path = None
        if draft_model_id is not None:
            draft_model_path = download_and_convert_model(draft_model_id).models_path
        ov_pipe = create_ov_pipeline(model_path, pipeline_type=pipeline_type, draft_model_path=draft_model_path)
        return ov_pipe.generate([prompt], generation_config).extended_perf_metrics

    import time
    start_time = time.perf_counter()
    generation_config = GenerationConfig(
        do_sample=False, 
        max_new_tokens=20, 
        ignore_eos=True, 
        num_assistant_tokens=5,
    )
    extended_perf_metrics = None
    if draft_model_id is None:
        extended_perf_metrics = run_extended_perf_metrics_collection(
            main_model_id, generation_config, prompt, pipeline_type, draft_model_id
        )
        total_time = (time.perf_counter() - start_time) * 1000

    else:
        if pipeline_type == PipelineType.SPECULATIVE_DECODING:
            generation_config = GenerationConfig(
                do_sample=False, max_new_tokens=20, ignore_eos=True, num_assistant_tokens=5
            )
            extended_perf_metrics = run_extended_perf_metrics_collection(
                main_model_id, generation_config, prompt, pipeline_type, draft_model_id
            )
            total_time = (time.perf_counter() - start_time) * 1000

    if pipeline_type == PipelineType.SPECULATIVE_DECODING:
        assert not extended_perf_metrics is None
        assert not extended_perf_metrics.main_model_metrics is None
        assert not extended_perf_metrics.draft_model_metrics is None

        assert extended_perf_metrics.get_num_accepted_tokens() > 0

        num_generated_tokens_main = extended_perf_metrics.main_model_metrics.get_num_generated_tokens()
        assert num_generated_tokens_main > 0 and num_generated_tokens_main <= generation_config.max_new_tokens

        # max num_generated_tokens for draft model will be reached if it will generate num_assistant_tokens at each step
        # plus fist token, which was generated by main model
        num_generated_tokens_draft = extended_perf_metrics.draft_model_metrics.get_num_generated_tokens()
        assert num_generated_tokens_draft > 0 and num_generated_tokens_draft < ((generation_config.max_new_tokens - 1) * generation_config.num_assistant_tokens + 1)

        total_iteration_number_main = len(extended_perf_metrics.main_model_metrics.raw_metrics.m_durations)
        assert total_iteration_number_main > 0 and total_iteration_number_main <= generation_config.max_new_tokens

        total_iteration_number_draft = len(extended_perf_metrics.draft_model_metrics.raw_metrics.m_durations)
        assert total_iteration_number_draft > 0 and total_iteration_number_draft < ((generation_config.max_new_tokens - 1) * generation_config.num_assistant_tokens + 1)

        for model_metrics in [
            extended_perf_metrics.main_model_metrics, 
            extended_perf_metrics.draft_model_metrics,
        ]:
            mean_ttst, std_ttst = model_metrics.get_ttst()
            assert (mean_ttst, std_ttst) == (model_metrics.get_ttst().mean, model_metrics.get_ttst().std)
            assert mean_ttst > 0 and mean_ttst < model_metrics.get_ttft().mean
            assert std_ttst == 0

            mean_latency, std_latency = model_metrics.get_latency()
            assert (mean_latency, std_latency) == (model_metrics.get_latency().mean, model_metrics.get_latency().std)
            assert mean_latency > 0 and mean_latency < 1000.0

            mean_gen_duration, std_gen_duration = model_metrics.get_generate_duration()
            assert (mean_gen_duration, std_gen_duration) == (model_metrics.get_generate_duration().mean, model_metrics.get_generate_duration().std)
            assert mean_gen_duration > 0 and mean_gen_duration < total_time
            assert std_gen_duration == 0
    else:
        assert extended_perf_metrics is None


devices = [("CPU", "CPU")]


@pytest.mark.parametrize("main_model,draft_model,prompt", eagle_models_and_input)
@pytest.mark.parametrize("main_device,draft_device", devices)
def test_eagle3_sd_string_inputs(main_model, main_device, draft_model, draft_device, prompt):
    # Download and convert model:
    main_model_schema = download_and_convert_model(main_model)
    main_opt_model = main_model_schema.opt_model
    main_hf_tokenizer = main_model_schema.hf_tokenizer
    main_model_path = main_model_schema.models_path

    draft_model_path = download_and_convert_model(draft_model).models_path

    # Create OpenVINO GenAI pipeline:

    ov_pipe = create_ov_pipeline(
        main_model_path, pipeline_type=PipelineType.SPECULATIVE_DECODING, draft_model_path=draft_model_path
    )

    # Run reference HF model:
    ov_generation_config = GenerationConfig(max_new_tokens=20)
    ref_gen_results = run_hugging_face(main_opt_model, main_hf_tokenizer, [prompt], ov_generation_config)

    # Run OpenVINO GenAI pipeline:
    ov_decoded_results = ov_pipe.generate([prompt], ov_generation_config)
    ov_gen_results = convert_decoded_results_to_generation_result(ov_decoded_results, 1, 1, False)

    del ov_pipe

    # Compare results:
    compare_generation_results([prompt], ref_gen_results, ov_gen_results, ov_generation_config)


def compare_results_for_dynamic_split_fuse_config(main_model_id, draft_model_id):
    main_model_path = download_and_convert_model(main_model_id).models_path
    draft_model_path = download_and_convert_model(draft_model_id).models_path

    scheduler_config_ref = dict_to_scheduler_config(
        {"dynamic_split_fuse": False, "max_num_batched_tokens": sys.maxsize}
    )
    ov_pipe_ref = create_ov_pipeline(
        main_model_path,
        pipeline_type=PipelineType.SPECULATIVE_DECODING,
        draft_model_path=draft_model_path,
        scheduler_config=scheduler_config_ref,
    )

    scheduler_config_target = dict_to_scheduler_config({"dynamic_split_fuse": True, "max_num_batched_tokens": 5})
    ov_pipe_target = create_ov_pipeline(
        main_model_path,
        pipeline_type=PipelineType.SPECULATIVE_DECODING,
        draft_model_path=draft_model_path,
        scheduler_config=scheduler_config_target,
    )

    generation_config = GenerationConfig(max_new_tokens=20, num_assistant_tokens=4)
    prompt = "Why is the Sun yellow?"
    result_ref = ov_pipe_ref.generate([prompt], generation_config)
    result_gen = ov_pipe_target.generate([prompt], generation_config)

    reference = result_ref.texts[0]
    generated = result_gen.texts[0]
    assert generated == reference

    extended_perf_metrics_gen = result_gen.extended_perf_metrics
    total_iteration_number_main = len(extended_perf_metrics_gen.main_model_metrics.raw_metrics.m_durations)
    num_generated_tokens_main = extended_perf_metrics_gen.main_model_metrics.get_num_generated_tokens()
    assert total_iteration_number_main > 0 and total_iteration_number_main < num_generated_tokens_main


def test_dynamic_split_fuse_for_speculative_decoding():
    compare_results_for_dynamic_split_fuse_config("HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-135M")


def test_dynamic_split_fuse_for_eagle3():
    compare_results_for_dynamic_split_fuse_config("Qwen/Qwen3-1.7B", "AngelSlim/Qwen3-1.7B_eagle3")
