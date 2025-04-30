# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import GenerationResult, GenerationConfig

def compare_generation_result(hf_result: GenerationResult,
                              ov_result: GenerationResult,
                              generation_config: GenerationConfig):
    if generation_config.is_beam_search():
        assert len(hf_result.m_scores) == len(ov_result.m_scores)
        for hf_score, ov_score in zip(hf_result.m_scores, ov_result.m_scores):
            # Note, that for fp32 / fp16 models scores are different less than 0.001
            assert abs(hf_score - ov_score) < 0.02

    if not generation_config.include_stop_str_in_output and len(generation_config.stop_strings) > 0:
        assert len(hf_result.m_generation_ids) >= len(ov_result.m_generation_ids)
        for hf_text, ov_text in zip(hf_result.m_generation_ids, ov_result.m_generation_ids):
            assert ov_text in hf_text
    else:
        assert len(hf_result.m_generation_ids) == len(ov_result.m_generation_ids)
        for hf_text, ov_text in zip(hf_result.m_generation_ids, ov_result.m_generation_ids):
            assert hf_text == ov_text
            

def compare_generation_results(prompts: list[str],
                               hf_results: list[GenerationResult],
                               ov_results: list[GenerationResult],
                               generation_configs: list[GenerationConfig] | GenerationConfig):
    if type(generation_configs) is not list:
        generation_configs = [generation_configs]

    assert len(prompts) == len(hf_results)
    assert len(prompts) == len(ov_results)

    for prompt, ref_result, ov_result, generation_config in zip(prompts, hf_results, ov_results, generation_configs):
        print(f"Prompt = {prompt}\nReference result = {ref_result}\nOpenVINO result = {ov_result.m_generation_ids}")
        compare_generation_result(ref_result, ov_result, generation_config)


# TODO: remove this function after Generator property is supported by LLMPipeline / VLMPipeline
def compare_generation_results_vs_ref(prompts: list[str],
                                      ref: list[list[str]],
                                      ov_results: list[GenerationResult]):
    assert len(prompts) == len(ref)
    assert len(prompts) == len(ov_results)

    for prompt, ref_texts_for_this_prompt, ov_result in zip(prompts, ref, ov_results):
        print(f"Prompt = {prompt}\nref text = {ref_texts_for_this_prompt}\nOV result = {ov_result.m_generation_ids}")

        assert len(ref_texts_for_this_prompt) == len(ov_result.m_generation_ids)
        for ref_text, ov_text in zip(ref_texts_for_this_prompt, ov_result.m_generation_ids):
            assert ref_text == ov_text