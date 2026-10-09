# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiny GGUF frontend regressions against the regular Optimum export path.

Only the selected GGUF and tokenizer/configuration files are downloaded to the Hub cache.
Reference IRs and GGUF round-trip outputs are generated in pytest's temporary directories.
Real-checkpoint quality comparisons remain in the opt-in WWB GGUF suite.
"""

import gc
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import openvino as ov
import openvino_genai as genai

from utils.comparation import compare_generation_results
from utils.constants import extra_generate_kwargs, get_default_llm_properties
from utils.hugging_face import (
    OVConvertedModelSchema,
    convert_models,
    generation_config_to_hf,
    run_hugging_face,
)
from utils.network import retry_request
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

# Pinned GGUFs (1.8 / 25.5 / 4.9 / 47.2 MB) and their source tokenizer/configuration.
# The HF model is reconstructed from the GGUF weights, not independently downloaded weights.
TINY_GGUF_MODELS = [
    pytest.param(
        (
            "aladar/llama-2-tiny-random-GGUF",
            "8d5321916486e1d33c46b16990e8da6567785769",
            "llama-2-tiny-random.gguf",
            "yujiepan/llama-2-tiny-random",
            "74bb065d6381fdf9fcde5896cb32267ba2c0ee0a",
        ),
        id="llama",
    ),
    pytest.param(
        (
            "mradermacher/tiny-random-qwen3-GGUF",
            "058102830a84bb976597f4d8462bafe0ea319358",
            "tiny-random-qwen3.f16.gguf",
            "snake7gun/tiny-random-qwen3",
            "add6406ed2f82988fd5194e343572194f71b6f61",
        ),
        id="qwen3",
    ),
    pytest.param(
        (
            "mradermacher/tiny-random-Phi3ForCausalLM-GGUF",
            "ff12230de1587e824688cc6afef2af22dc2b7807",
            "tiny-random-Phi3ForCausalLM.f16.gguf",
            "Xenova/tiny-random-Phi3ForCausalLM",
            "bac53a980e9402513e0a5f8433656b4f845c1b9b",
        ),
        id="phi3",
    ),
    pytest.param(
        (
            "ggml-org/tinygemma3-GGUF",
            "c287502cd9e278dac8eed805c112cce5d0081e0b",
            "tinygemma3-Q8_0.gguf",
            "ngxson/tinygemma3_cifar",
            "da67f59d2195a0095711f1494f8ee5823cf54591",
        ),
        id="gemma3",
    ),
]


def reference_properties():
    return {**get_default_llm_properties(), "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}


@dataclass
class TinyGGUFModel:
    gguf_path: Path
    reference: OVConvertedModelSchema


@pytest.fixture(scope="module", params=TINY_GGUF_MODELS)
def tiny_gguf_model(request, tmp_path_factory):
    gguf_repo, gguf_revision, filename, hf_repo, hf_revision = request.param
    gguf_dir = retry_request(
        lambda: snapshot_download(
            gguf_repo,
            revision=gguf_revision,
            allow_patterns=[filename],
        )
    )
    hf_dir = retry_request(
        lambda: snapshot_download(
            hf_repo,
            revision=hf_revision,
            allow_patterns=["*.json", "tokenizer.model", "*.txt"],
        )
    )
    # Use the source configuration/tokenizer: HF's GGUF loader omits some configuration
    # fields (e.g. Qwen3 head_dim) and does not reconstruct all SentencePiece tokenizers.
    config = AutoConfig.from_pretrained(hf_dir).get_text_config()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_model, loading_info = AutoModelForCausalLM.from_pretrained(
        gguf_dir,
        gguf_file=filename,
        config=config,
        torch_dtype=torch.float32,
        output_loading_info=True,
    )
    assert not loading_info["missing_keys"], loading_info
    assert not loading_info["unexpected_keys"], loading_info
    assert not loading_info["mismatched_keys"], loading_info
    hf_model.eval()
    hf_model.generation_config.pad_token_id = hf_tokenizer.pad_token_id
    reference_dir = tmp_path_factory.mktemp("gguf_optimum_reference")
    hf_model.save_pretrained(reference_dir)
    hf_tokenizer.save_pretrained(reference_dir)
    del hf_model
    gc.collect()

    opt_model = OVModelForCausalLM.from_pretrained(
        reference_dir,
        export=True,
        load_in_8bit=False,
        ov_config=reference_properties(),
    )
    convert_models(opt_model, hf_tokenizer, reference_dir)
    yield TinyGGUFModel(
        Path(gguf_dir) / filename,
        OVConvertedModelSchema(hf_repo, opt_model, hf_tokenizer, reference_dir),
    )
    del opt_model
    gc.collect()


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
