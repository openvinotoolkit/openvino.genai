# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GenAI integration regressions: pinned GGUF -> Optimum reference IR -> pipeline comparisons.

See README.md, "GGUF Frontend Tests", for the steps, CI runtime and model-coverage checklist.
"""

import shutil
import sys

import numpy as np
import pytest
import torch

import openvino as ov
import openvino_genai as genai

from data.models import TINY_GGUF_MODELS
from utils.gguf import load_tiny_gguf_model, reference_properties
from utils.comparation import compare_generation_results
from utils.constants import extra_generate_kwargs
from utils.hugging_face import (
    generation_config_to_hf,
    run_hugging_face,
)
from utils.ov_genai_pipelines import (
    GGUF_PIPELINE_TYPES,
    PipelineType,
    convert_decoded_results_to_generation_result,
    create_ov_pipeline,
)


pytestmark = [
    pytest.mark.skipif(sys.platform == "darwin", reason="CVS-168882: sporadic segmentation fault"),
    pytest.mark.skipif(sys.platform == "win32", reason="CVS-174065"),
]

@pytest.fixture(scope="module", params=TINY_GGUF_MODELS)
def tiny_gguf_model(request, tmp_path_factory):
    return load_tiny_gguf_model(request.param, tmp_path_factory)


@pytest.fixture(scope="module", params=(*GGUF_PIPELINE_TYPES, PipelineType.AUTO))
def pipeline_type(request):
    return request.param


@pytest.fixture(scope="module")
def gguf_pipeline(tiny_gguf_model, pipeline_type):
    return create_ov_pipeline(
        tiny_gguf_model.gguf_path,
        pipeline_type=pipeline_type,
        ov_config=reference_properties(),
        gguf_reader="FRONTEND",
    )


@pytest.mark.parametrize(
    "prompts",
    [
        ["Why is the Sun yellow?"],
        ["Hello", "table is made of", "Difference between Jupiter and Mars is that"],
    ],
)
def test_tiny_gguf_string_inputs(tiny_gguf_model, gguf_pipeline, pipeline_type, prompts):
    if pipeline_type == PipelineType.STATEFUL and len(prompts) > 1:
        pytest.skip("Native GGUF SDPA attention masks support batch one; PA covers batched generation")
    reference = tiny_gguf_model.reference
    generation_config = genai.GenerationConfig(
        max_new_tokens=16,
        min_new_tokens=16,
        apply_chat_template=False,
    )
    for prompt in prompts:
        assert gguf_pipeline.get_tokenizer().encode(prompt).input_ids.data.tolist() == [
            reference.hf_tokenizer.encode(prompt)
        ]
    expected = run_hugging_face(reference.opt_model, reference.hf_tokenizer, prompts, generation_config)
    actual = gguf_pipeline.generate(prompts, generation_config)
    actual = convert_decoded_results_to_generation_result(
        actual,
        len(prompts),
        generation_config.num_return_sequences,
        generation_config.is_beam_search(),
    )
    # Supply a config per prompt so the shared comparison checks every batch element.
    compare_generation_results(prompts, expected, actual, [generation_config] * len(prompts))


@pytest.mark.parametrize("text", [" hello", "  hello"])
def test_tiny_gguf_detokenization(tiny_gguf_model, gguf_pipeline, text):
    tokenizer = tiny_gguf_model.reference.hf_tokenizer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    assert gguf_pipeline.get_tokenizer().decode(tokens) == tokenizer.decode(tokens, skip_special_tokens=True)


def test_tiny_gguf_standalone_tokenizer(tiny_gguf_model):
    tokenizer = genai.GGUFTokenizer(tiny_gguf_model.gguf_path, GGUF_READER="FRONTEND")
    assert isinstance(tokenizer, genai.Tokenizer)
    reference = tiny_gguf_model.reference.hf_tokenizer
    prompt = "Why is the Sun yellow?"
    tokens = reference.encode(prompt)
    assert tokenizer.encode(prompt).input_ids.data.tolist() == [tokens]
    assert tokenizer.decode(tokens) == reference.decode(tokens, skip_special_tokens=True)


@pytest.mark.parametrize(
    "prompts",
    [
        ["Why is the Sun yellow?"],
        ["Hello", "table is made of", "Difference between Jupiter and Mars is that"],
    ],
)
def test_tiny_gguf_beam_search(tiny_gguf_model, gguf_pipeline, pipeline_type, prompts):
    if pipeline_type == PipelineType.STATEFUL:
        pytest.skip("Native GGUF SDPA attention masks support batch one; PA covers beam expansion")
    reference = tiny_gguf_model.reference
    inputs = reference.hf_tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
    config = genai.GenerationConfig(max_new_tokens=16, min_new_tokens=16, num_beams=3, num_return_sequences=2)
    hf_config = generation_config_to_hf(reference.opt_model.generation_config, config)
    expected = reference.opt_model.generate(
        **inputs,
        generation_config=hf_config,
        **extra_generate_kwargs(hf_config),
    )
    actual = gguf_pipeline.generate(
        genai.TokenizedInputs(
            ov.Tensor(inputs.input_ids.numpy()),
            ov.Tensor(inputs.attention_mask.numpy()),
        ),
        config,
    )
    # Random fixtures can generate reserved tokens whose display policy differs across tokenizers.
    # Compare token IDs and beam scores directly, including every returned sequence in the batch.
    assert actual.tokens == expected.sequences[:, inputs.input_ids.shape[1] :].tolist()
    np.testing.assert_allclose(actual.scores, expected.sequences_scores.numpy(), atol=0.02, rtol=0)


@pytest.mark.parametrize("with_attention_mask", [False, True])
def test_tiny_gguf_encoded_inputs(tiny_gguf_model, gguf_pipeline, with_attention_mask):
    reference = tiny_gguf_model.reference
    input_ids = np.array([[1, 4, 42]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    generation_config = genai.GenerationConfig(max_new_tokens=16, min_new_tokens=16)
    hf_config = generation_config_to_hf(reference.opt_model.generation_config, generation_config)
    hf_inputs = {"input_ids": torch.from_numpy(input_ids)}
    ov_inputs = ov.Tensor(input_ids)
    if with_attention_mask:
        hf_inputs["attention_mask"] = torch.from_numpy(attention_mask)
        ov_inputs = genai.TokenizedInputs(ov_inputs, ov.Tensor(attention_mask))
    expected = (
        reference.opt_model.generate(
            **hf_inputs,
            generation_config=hf_config,
            **extra_generate_kwargs(hf_config),
        )
        .sequences[:, input_ids.shape[1] :]
        .tolist()
    )
    assert gguf_pipeline.generate(ov_inputs, generation_config).tokens == expected


def test_tiny_gguf_streaming(tiny_gguf_model, gguf_pipeline):
    prompt = "Why is the Sun yellow?"
    generation_config = genai.GenerationConfig(max_new_tokens=16, min_new_tokens=16, apply_chat_template=False)
    reference = tiny_gguf_model.reference
    expected = run_hugging_face(reference.opt_model, reference.hf_tokenizer, [prompt], generation_config)
    chunks = []
    actual = gguf_pipeline.generate(prompt, generation_config, streamer=lambda text: chunks.append(text))
    assert actual == expected[0].m_generation_ids[0]
    assert "".join(chunks) == actual


def test_tiny_gguf_saved_ir(tiny_gguf_model, pipeline_type, tmp_path):
    # Serialization writes beside the GGUF; never modify the shared Hub cache.
    gguf_path = tmp_path / tiny_gguf_model.gguf_path.name
    shutil.copyfile(tiny_gguf_model.gguf_path, gguf_path)
    pipeline = create_ov_pipeline(
        gguf_path,
        pipeline_type=pipeline_type,
        ov_config=reference_properties(),
        gguf_reader="FRONTEND",
        enable_save_ov_model=True,
    )
    prompt = "table is made of"
    config = genai.GenerationConfig(max_new_tokens=16, apply_chat_template=False, ignore_eos=True)
    expected = pipeline.generate(prompt, config)
    del pipeline
    reloaded = create_ov_pipeline(tmp_path, pipeline_type=pipeline_type, ov_config=reference_properties())
    assert reloaded.generate(prompt, config) == expected
