# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import numpy as np 
import logging

import openvino as ov
import openvino_genai as ov_genai

from utils.constants import get_default_llm_properties
from utils.hugging_face import generation_config_to_hf, download_and_convert_model, run_hugging_face
from utils.comparation import compare_generation_results
from utils.ov_genai_pipelines import create_ov_pipeline, generate_and_compare, get_main_pipeline_types, PipelineType, convert_decoded_results_to_generation_result

test_cases = [
    ('CPU', 'CPU'),
    ('CPU', 'NPUW:CPU'),
    ('NPUW:CPU', 'CPU'),
    ('NPUW:CPU', 'NPUW:CPU')
]
@pytest.mark.parametrize("main_device,draft_device", test_cases)
@pytest.mark.precommit
def test_string_inputs(main_device, draft_device):
    # FIXME: For now SmolLM2-135M is used as a main and a draft model in the test.
    #        However, it is more desirable to use SmolLM2-360M as a main one to simulate the real case
    #        for speculative decoding.
    #        It seems like temporary directory from model downloading stage isn't removed after test
    #        launch for SmolLM2-360M model, that is why it is not used now.
    MODEL_UNDER_TEST = {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "convert_args": ['--trust-remote-code']
    }
    prompt = "Alan Turing was a"

    # Download and convert model:
    main_opt_model, main_hf_tokenizer, main_model_path = download_and_convert_model(MODEL_UNDER_TEST["name"])
    draft_model_path = main_model_path

    # Create OpenVINO GenAI pipeline:
    draft_config = get_default_llm_properties()
    if draft_device == "NPUW:CPU":
        draft_device = "NPU"
        draft_config["NPUW_DEVICES"] = "CPU"
        draft_config["GENERATE_HINT"] = "BEST_PERF"
        # FIXME: Currently, the same draft and main model fails to work in NPUW_WEIGHTS_BANK: shared mode.
        #        To workaround this, we name banks differently for draft and main.
        draft_config["NPUW_WEIGHTS_BANK"] = "draft"
    ov_draft_model = ov_genai.draft_model(draft_model_path, draft_device, **draft_config)

    main_config = get_default_llm_properties()
    if main_device == "NPUW:CPU":
        main_device = "NPU"
        main_config["NPUW_DEVICES"] = "CPU"
        # FIXME: SmolLM-135M with GENERATE_HINT: FAST_COMPILE will output garbage on NPUW:CPU if used with configuration
        #        NPUW_LLM_MAX_GENERATION_TOKEN_LEN > 1.
        #        Setting GENERATE_HINT: BEST_PERF to workaround an issue currently.
        main_config["GENERATE_HINT"] = "BEST_PERF"
        # FIXME: Currently, the same draft and main model fails to work in NPUW_WEIGHTS_BANK: shared mode.
        #        To workaround this, we name banks differently for draft and main.
        main_config["NPUW_WEIGHTS_BANK"] = "main"
    main_config["ATTENTION_BACKEND"] = "SDPA"
    ov_pipe = ov_genai.LLMPipeline(main_model_path, main_device, main_config, draft_model=ov_draft_model)

    # Run reference HF model:
    ov_generation_config = ov_genai.GenerationConfig(max_new_tokens=20)
    main_hf_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    ref_gen_results = run_hugging_face(main_opt_model, main_hf_tokenizer, [prompt], ov_generation_config)  

    # Run OpenVINO GenAI pipeline:
    ov_decoded_results = ov_pipe.generate([prompt], ov_generation_config)
    ov_gen_results = convert_decoded_results_to_generation_result(ov_decoded_results, 1, 1, False)

    del ov_pipe

    # Compare results:
    compare_generation_results([prompt], ref_gen_results, ov_gen_results, ov_generation_config)

@pytest.mark.precommit
def test_perf_metrics():
    import time
    start_time = time.perf_counter()
    model_id = 'katuni4ka/tiny-random-gemma2'
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=20, ignore_eos=True, num_assistant_tokens=5)
    _, _, model_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(model_path, pipeline_type=PipelineType.STATEFUL_SPECULATIVE_DECODING)
    prompt = 'table is made of'
    perf_metrics = ov_pipe.generate([prompt], generation_config).perf_metrics
    total_time = (time.perf_counter() - start_time) * 1000

    # Check that load time is adequate.
    load_time = perf_metrics.get_load_time()
    assert load_time > 0 and load_time < total_time

    # Check that num input and generated tokens are adequate.
    num_generated_tokens = perf_metrics.get_num_generated_tokens()
    assert num_generated_tokens > 0 and num_generated_tokens <= generation_config.max_new_tokens

    num_input_tokens = perf_metrics.get_num_input_tokens()
    assert num_input_tokens > 0 and num_input_tokens <= len(prompt)

    mean_ttft, std_ttft = perf_metrics.get_ttft()
    assert (mean_ttft, std_ttft) == (perf_metrics.get_ttft().mean, perf_metrics.get_ttft().std)
    assert mean_ttft > 0 and mean_ttft < 1000.0

    raw_metrics = perf_metrics.raw_metrics
    durations = np.array(raw_metrics.m_durations) / 1000
    # Check that prefill is not included in durations for TPOT calculation.
    # For the very long prompt prefill is slow and TTFT is much larger than any other token generation duration.
    assert np.all(mean_ttft > durations)

    mean_tpot, std_tpot = perf_metrics.get_tpot()
    assert (mean_tpot, std_tpot) == (perf_metrics.get_tpot().mean, perf_metrics.get_tpot().std)
    assert mean_tpot > 0 and mean_ttft < 1000.0

    mean_throughput, std_throughput = perf_metrics.get_throughput()
    assert (mean_throughput, std_throughput) == (perf_metrics.get_throughput().mean, perf_metrics.get_throughput().std)
    assert mean_throughput > 0 and mean_throughput < 20000.0

    mean_gen_duration, std_gen_duration = perf_metrics.get_generate_duration()
    assert (mean_gen_duration, std_gen_duration) == (perf_metrics.get_generate_duration().mean, perf_metrics.get_generate_duration().std)
    assert mean_gen_duration > 0 and load_time + mean_gen_duration < total_time
    assert std_gen_duration == 0

    mean_tok_duration, std_tok_duration = perf_metrics.get_tokenization_duration()
    assert (mean_tok_duration, std_tok_duration) == (perf_metrics.get_tokenization_duration().mean, perf_metrics.get_tokenization_duration().std)
    assert mean_tok_duration > 0 and mean_tok_duration < mean_gen_duration
    assert std_tok_duration == 0

    mean_detok_duration, std_detok_duration = perf_metrics.get_detokenization_duration()
    assert (mean_detok_duration, std_detok_duration) == (perf_metrics.get_detokenization_duration().mean, perf_metrics.get_detokenization_duration().std)
    assert mean_detok_duration > 0 and mean_detok_duration < mean_gen_duration
    assert std_detok_duration == 0

    # assert that calculating statistics manually from the raw counters we get the same restults as from PerfMetrics
    assert np.allclose(mean_tpot, np.mean(durations))
    assert np.allclose(std_tpot, np.std(durations), atol=0.00002)

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

@pytest.mark.precommit
def test_extended_perf_metrics():
    import time
    start_time = time.perf_counter()
    model_id : str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    generation_config = ov_genai.GenerationConfig(do_sample=False, max_new_tokens=20, ignore_eos=True, num_assistant_tokens=5)
    _, _, model_path = download_and_convert_model(model_id)
    ov_pipe = create_ov_pipeline(model_path, pipeline_type=PipelineType.STATEFUL_SPECULATIVE_DECODING)
    extended_perf_metrics = ov_pipe.generate(["Why is the Sun yellow?"], generation_config).extended_perf_metrics
    total_time = (time.perf_counter() - start_time) * 1000

    assert not extended_perf_metrics is None
    assert not extended_perf_metrics.main_model_metrics is None
    assert not extended_perf_metrics.draft_model_metrics is None

    assert extended_perf_metrics.get_num_accepted_tokens() > 0

    num_generated_tokens_main = extended_perf_metrics.main_model_metrics.get_num_generated_tokens()
    assert num_generated_tokens_main > 0 and num_generated_tokens_main <= generation_config.max_new_tokens

    num_generated_tokens_draft = extended_perf_metrics.draft_model_metrics.get_num_generated_tokens()
    # As Stateful Speculative Decoding pipeline is dynamically adjusting its number of candidates at
    # each step, here we check that generated tokens is less than upper candidates limit multiplied by
    # maximum number of generated tokens.
    assert num_generated_tokens_draft > 0 and \
        num_generated_tokens_draft < ((generation_config.max_new_tokens - 1) * \
                                      generation_config.num_assistant_tokens * 2 + 1)

    total_iteration_number_main = len(extended_perf_metrics.main_model_metrics.raw_metrics.m_durations)
    assert total_iteration_number_main > 0 and total_iteration_number_main <= generation_config.max_new_tokens

    total_iteration_number_draft = len(extended_perf_metrics.draft_model_metrics.raw_metrics.m_durations)
    assert total_iteration_number_draft > 0 and \
        total_iteration_number_draft < ((generation_config.max_new_tokens - 1) * \
                                       generation_config.num_assistant_tokens * 2 + 1)

    for model_metrics in [extended_perf_metrics.main_model_metrics, extended_perf_metrics.draft_model_metrics]:
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
