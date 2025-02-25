# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# todo: CVS-162108: remove this file to habdle generation config directly in tests

from openvino_genai import GenerationConfig

def get_greedy() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_greedy_with_penalties() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.presence_penalty = 2.0
    generation_config.frequency_penalty = 0.2
    generation_config.max_new_tokens = 30
    return generation_config

def get_beam_search() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.diversity_penalty = 1
    generation_config.max_new_tokens = 30
    generation_config.num_return_sequences = 3
    generation_config.num_return_sequences = generation_config.num_beams
    return generation_config

def get_multinomial_temperature() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_num_return_sequence() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.7
    generation_config.num_return_sequences = 3
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_top_p() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.top_p = 0.9
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_top_k() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.temperature = 0.8
    generation_config.top_k = 2
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_top_p_and_top_k() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.top_p = 0.9
    generation_config.num_return_sequences = 1
    generation_config.top_k = 2
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_repetition_penalty() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.temperature = 0.8
    generation_config.repetition_penalty = 2.0
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_all_parameters() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.num_return_sequences = 4
    generation_config.temperature = 0.9
    generation_config.top_p = 0.8
    generation_config.top_k = 20
    generation_config.repetition_penalty = 2.0
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_frequence_penalty() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.frequency_penalty = 0.5
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_temperature_and_presence_penalty() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.presence_penalty = 0.1
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 30
    return generation_config

def get_multinomial_max_and_min_token() -> GenerationConfig:
    multinomial = GenerationConfig()
    multinomial.do_sample = True
    multinomial.temperature = 0.9
    multinomial.top_p = 0.9
    multinomial.top_k = 20
    multinomial.num_return_sequences = 3
    multinomial.presence_penalty = 0.01
    multinomial.frequency_penalty = 0.1
    multinomial.min_new_tokens = 15
    multinomial.max_new_tokens = 30
    return multinomial
