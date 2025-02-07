# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List, Tuple, Callable

from openvino_genai import SchedulerConfig, GenerationResult, GenerationConfig, StreamerBase

from utils.hf_utils import download_and_convert_model, run_hugging_face
from utils.ov_genai_utils import StreamerWithResults, run_llm_pipeline, dict_to_scheduler_config, run_continuous_batching
from utils.comparation_utils import compare_generation_results
from utils.generation_config_samples import get_greedy, get_beam_search


TESTS_ROOT = Path(__file__).parent

def run_llm_pipeline_with_ref(model_id: str, 
                              prompts: List[str], 
                              generation_config: GenerationConfig | dict, 
                              tmp_path: Path, 
                              use_cb : bool = False,
                              streamer: StreamerWithResults | Callable | StreamerBase = None):
    if type(generation_config) is dict:
        generation_config = GenerationConfig(**generation_config)

    models_path, opt_model, hf_tokenizer = download_and_convert_model(model_id, tmp_path)
    ov_results = run_llm_pipeline(models_path, prompts, generation_config, use_cb, streamer=streamer.accumulate if isinstance(streamer, StreamerWithResults) else streamer)
    hf_results = run_hugging_face(opt_model, hf_tokenizer, prompts, generation_config)

    compare_generation_results(prompts, hf_results, ov_results, generation_config)


def get_test_dataset() -> Tuple[List[str], List[GenerationConfig]]:
    prompts = [
        "What is OpenVINO?",
        "How are you?",
        "What is your name?",
        "Tell me something about Canada"
    ]
    generation_configs = [
        get_greedy(),
        get_beam_search(),
        get_greedy(),
        get_beam_search(),
    ]
    return (prompts, generation_configs)


def run_cb_pipeline_with_ref(tmp_path: str,
                             model_id: str,
                             scheduler_params: dict = {},
                             generation_config : GenerationConfig | dict = None):
    prompts, generation_configs = get_test_dataset()
    scheduler_config = dict_to_scheduler_config(scheduler_params)

    # override dataset's generation config
    if generation_config is not None:
        if type(generation_config) is dict:
            generation_config = GenerationConfig(**generation_config)
        generation_configs = [generation_config] * len(prompts)

    models_path, opt_model, hf_tokenizer = download_and_convert_model(model_id, tmp_path)

    hf_results = run_hugging_face(opt_model, hf_tokenizer, prompts, generation_configs)
    ov_results = run_continuous_batching(models_path, scheduler_config, prompts, generation_configs)

    compare_generation_results(prompts, hf_results, ov_results, generation_configs)


# TODO: remove after Generator property is supported by LLMPipeline / VLMPipeline
def generate_and_compare_with_reference_text(models_path: Path,
                                             prompts: List[str],
                                             reference_texts_per_prompt: List[List[str]],
                                             generation_configs: List[GenerationConfig],
                                             scheduler_config: SchedulerConfig):
    ov_results : List[GenerationResult] = run_continuous_batching(models_path, scheduler_config, prompts, generation_configs)

    assert len(prompts) == len(reference_texts_per_prompt)
    assert len(prompts) == len(ov_results)

    for prompt, ref_texts_for_this_prompt, ov_result in zip(prompts, reference_texts_per_prompt, ov_results):
        print(f"Prompt = {prompt}\nref text = {ref_texts_for_this_prompt}\nOV result = {ov_result.m_generation_ids}")

        assert len(ref_texts_for_this_prompt) == len(ov_result.m_generation_ids)
        for ref_text, ov_text in zip(ref_texts_for_this_prompt, ov_result.m_generation_ids):
            assert ref_text == ov_text


