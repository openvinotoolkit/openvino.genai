# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino_genai import GenerationConfig, GenerationResult

from utils.comparation import compare_generation_results


@pytest.mark.parametrize("per_prompt_config", [False, True])
def test_compare_generation_results_checks_every_prompt(per_prompt_config):
    def result(text):
        value = GenerationResult()
        value.m_generation_ids = [text]
        return value

    config = GenerationConfig(max_new_tokens=1)
    configs = [config, config] if per_prompt_config else config
    with pytest.raises(AssertionError):
        compare_generation_results(
            ["first", "second"],
            [result("match"), result("expected")],
            [result("match"), result("different")],
            configs,
        )


def test_compare_generation_results_rejects_incomplete_config_list():
    result = GenerationResult()
    result.m_generation_ids = ["match"]
    with pytest.raises(AssertionError):
        compare_generation_results(
            ["first", "second"], [result, result], [result, result], [GenerationConfig(max_new_tokens=1)]
        )


def test_compare_generation_results_accepts_config_list_subclass():
    class ConfigList(list):
        pass

    result = GenerationResult()
    result.m_generation_ids = ["match"]
    config = GenerationConfig(max_new_tokens=1)
    compare_generation_results(["first", "second"], [result, result], [result, result], ConfigList([config, config]))
