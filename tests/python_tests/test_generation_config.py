# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino_genai import GenerationConfig
import json
import os
import pytest


def verify_set_values(generation_config, kwargs):
    generation_config.validate()
    for key, value in kwargs.items():
        if key == "stop_token_ids":
            continue
        assert getattr(generation_config, key) == value
    if "eos_token_id" in kwargs:
        assert kwargs["eos_token_id"] in generation_config.stop_token_ids
        if "stop_token_ids" in kwargs:
            for stop_id in kwargs["stop_token_ids"]:
                assert stop_id in generation_config.stop_token_ids

configs = [
    # stop conditions
    dict(max_new_tokens=12),
    dict(max_length=12),
    dict(stop_token_ids={2}),
    dict(eos_token_id=1),
    dict(eos_token_id=1, stop_token_ids={1}),
    dict(eos_token_id=1, stop_token_ids={2}),
    dict(stop_strings={"a", "b"}),
    dict(ignore_eos=True, max_new_tokens=10),
    dict(ignore_eos=True, max_length=10),
    dict(max_new_tokens=0, echo=True),
    dict(min_new_tokens=1, max_new_tokens=1),
    # multinomial
    dict(max_new_tokens=1, do_sample=True, num_return_sequences=2),
    dict(max_new_tokens=1, do_sample=True, top_k=1),
    dict(max_new_tokens=1, do_sample=True, top_p=0.5),
    dict(max_new_tokens=1, do_sample=True, temperature=0.5),
    # parameters requiring multimonial are ignored when do_sample=False
    dict(max_new_tokens=1, top_k=1), # requires do_sample=True
    dict(max_new_tokens=1, top_p=0.5), # requires do_sample=True
    dict(max_new_tokens=1, temperature=2.0), # requires do_sample=True
    # beam search
    dict(max_new_tokens=1, num_beams=2),
    dict(max_new_tokens=1, num_beams=2, num_return_sequences=1),
    dict(max_new_tokens=1, num_beams=2, num_return_sequences=2),
    dict(max_new_tokens=1, num_beams=4, num_beam_groups=2, diversity_penalty=1.0),
    dict(max_new_tokens=1, num_beams=4, length_penalty=1.0),
    dict(max_new_tokens=1, num_beams=4, no_repeat_ngram_size=2),
    # parameters requiring beam search are ignored when num_beams == 1
    dict(max_new_tokens=1, num_beam_groups=2), # requiring beam search
    dict(max_new_tokens=1, no_repeat_ngram_size=2), # requiring beam search
    dict(max_new_tokens=1, diversity_penalty=1.0), # requiring beam search
    dict(max_new_tokens=1, length_penalty=2), # requiring beam search
    # assistant generation
    dict(max_new_tokens=1, assistant_confidence_threshold=0.5),
    dict(max_new_tokens=1, num_assistant_tokens=2),
    dict(max_new_tokens=1, num_assistant_tokens=2, max_ngram_size=2), # prompt lookup
    dict(max_new_tokens=1, apply_chat_template=True),
    dict(max_new_tokens=1, apply_chat_template=False),
]
@pytest.mark.parametrize("generation_config_kwargs", configs)
@pytest.mark.precommit
def test_valid_configs(generation_config_kwargs):
    config = GenerationConfig(**generation_config_kwargs)
    verify_set_values(config, generation_config_kwargs)

    config = GenerationConfig()
    config.update_generation_config(**generation_config_kwargs)
    verify_set_values(config, generation_config_kwargs)


invalid_configs = [
    dict(num_return_sequences=0), # no reason to run with empty output
    dict(num_return_sequences=2), # beam search or multimonial is required
    # stop conditions
    dict(), # no stop conditions at all
    dict(ignore_eos=True),  # no 'max_new_tokens', no 'max_length' with 'ignore_eos'
    dict(stop_token_ids={-1}), # value in 'stop_token_ids' must be non-negative 
    dict(max_new_tokens=0), # max new tokens cannot be empty (only when 'echo' is True)
    dict(max_new_tokens=10, min_new_tokens=20), # 'max_new_tokens' must be >= 'min_new_tokens'
    # penalties
    dict(max_new_tokens=1, repetition_penalty=-1.0), # invalid repetition_penalty
    dict(max_new_tokens=1, presence_penalty=-3.0), # invalid presence_penalty
    dict(max_new_tokens=1, frequency_penalty=3.0), # invalid frequency_penalty
    # multinomial sampling
    dict(max_new_tokens=1, do_sample=True, top_p=1.1), # 'top_p' must be within (0, 1] when 'do_sample' is True
    dict(max_new_tokens=1, do_sample=True, top_p=0), # 'top_p' must be within (0, 1] when 'do_sample' is True
    dict(max_new_tokens=1, do_sample=True, temperature=-1.0), # invalid temp
    # beam search
    dict(max_new_tokens=1, num_beams=2, num_return_sequences=3), # 'num_beams' must be >= 'num_return_sequences'
    dict(max_new_tokens=1, num_beams=3, num_beam_groups=2), # 'num_beams' must be divisible by 'num_beam_groups'
    dict(max_new_tokens=1, num_beams=3, do_sample=True), # 'beam sample is not supported
    dict(max_new_tokens=1, num_beams=3, no_repeat_ngram_size=0), # invalid 'no_repeat_ngram_size'
    dict(max_new_tokens=1, num_beams=4, num_beam_groups=2, diversity_penalty=0.0), # 'diversity_penalty' should not be a default value
    dict(max_new_tokens=1, num_beams=4, diversity_penalty=1.0), # 'diversity_penalty' is used only for grouped beam search
    dict(max_new_tokens=1, num_beams=2, frequency_penalty=1.0), # 'frequency_penalty' is not supported by beam search
    dict(max_new_tokens=1, num_beams=2, presence_penalty=1.0), # 'presence_penalty' is not supported by beam search
    dict(max_new_tokens=1, num_beams=2, repetition_penalty=0.0), # 'repetition_penalty' is not supported by beam search
    # assistant generation
    dict(max_new_tokens=1, num_assistant_tokens=2, do_sample=True, num_return_sequences=2), # 'num_return_sequences' must be 1, as we cannot use different number of tokens per sequence within a group
    dict(max_new_tokens=1, assistant_confidence_threshold=1.0, do_sample=True, num_return_sequences=2), # 'num_return_sequences' must be 1, as we cannot use different number of tokens per sequence within a group
    dict(max_new_tokens=1, num_assistant_tokens=2, num_beams=2), # beam search is not compatible with assistant generation
    dict(max_new_tokens=1, assistant_confidence_threshold=1.0, num_assistant_tokens=2), # 'assistant_confidence_threshold' and 'num_assistant_tokens' are mutually exclusive
    dict(max_new_tokens=1, max_ngram_size=1), # 'max_ngram_size' is for prompt lookup, but assistant generation is turned off ('num_assistant_tokens' is 0)
    # TODO: add tests for invalid properties
]
@pytest.mark.parametrize("generation_config_kwargs", invalid_configs)
@pytest.mark.precommit
def test_invalid_generation_configs_throws(generation_config_kwargs):
    config = GenerationConfig(**generation_config_kwargs)
    with pytest.raises(RuntimeError):
        config.validate()

    config = GenerationConfig()
    config.update_generation_config(**generation_config_kwargs)
    with pytest.raises(RuntimeError):
        config.validate()


@pytest.mark.parametrize("fields", invalid_configs + [
    dict(eos_token_id=1), # 'stop_token_ids' does not contain 'eos_token_id'
    dict(eos_token_id=1, stop_token_ids={2}), # 'stop_token_ids' is not empty, but does not contain 'eos_token_id'
])
@pytest.mark.precommit
def test_invalid_fields_assinment_rises(fields):
    config = GenerationConfig()
    for key, val in fields.items():
        setattr(config, key, val)
    with pytest.raises(RuntimeError):
        config.validate()


def load_genai_generation_config_from_file(configs: list[tuple], temp_path):
    for json_file in temp_path.glob("*.json"):
        json_file.unlink()

    for config_json, config_name in configs:
        with (temp_path / config_name).open('w', encoding="utf-8") as f:
            json.dump(config_json, f)

    ov_generation_config = GenerationConfig(temp_path / "generation_config.json")

    for _, config_name in configs:
        os.remove(temp_path / config_name)

    return ov_generation_config

@pytest.mark.precommit
def test_multiple_eos_are_read_as_stop_token_ids(tmp_path):
    generation_config_json = {
        "eos_token_id": [
            2,
            32000,
            32007
        ]
    }
    configs = [
        (generation_config_json, "generation_config.json"),
    ]

    generation_config = load_genai_generation_config_from_file(configs, tmp_path)

    assert generation_config.eos_token_id == 2
    assert generation_config.stop_token_ids == { 2, 32000, 32007 }
