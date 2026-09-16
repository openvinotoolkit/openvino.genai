# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate tiny GGUFs and CPU references with the same pinned llama.cpp build."""

import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

import openvino as ov
import openvino_genai as genai
from openvino_tokenizers import convert_tokenizer

from utils.constants import get_default_llm_properties


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="CPU reference build is validated on Linux")

# Native frontend architectures that the pinned llama.cpp generator can export.
# Gemma3, Gemma4, DeepSeek2-OCR, Mellum and Muse-Glimmer need upstream fixture/saver support.
ARCHITECTURES = [
    "bailingmoe2",
    "ernie4_5-moe",
    "exaone4",
    "gemma",
    "gemma2",
    "gpt-oss",
    "hunyuan-dense",
    "llama",
    "maincoder",
    "minicpm",
    "mistral3",
    "olmoe",
    "phi3",
    "qwen2",
    "qwen3",
    "qwen35",
    "qwen3moe",
    "smollm3",
]
MOE_ARCHITECTURES = {"bailingmoe2", "ernie4_5-moe", "gpt-oss", "olmoe", "qwen3moe"}
MODEL_CASES = [
    pytest.param((arch, "moe" if arch in MOE_ARCHITECTURES else "dense"), id=arch) for arch in ARCHITECTURES
] + [pytest.param((arch, "moe"), id=f"{arch}-moe") for arch in ("llama", "minicpm", "mistral3")]
PROMPTS = [[1, 2, 3], [7, 4, 9, 5, 11], list(range(1, 25))]
NEW_TOKENS = 16


def run_reference_command(command, timeout=180, env=None):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    assert completed.returncode == 0, f"{command}\n{completed.stdout}\n{completed.stderr}"
    return completed


@pytest.fixture(scope="session")
def llama_reference_tools(tmp_path_factory):
    source = Path(__file__).parent / "gguf_reference"
    build = Path(os.environ.get("GGUF_LLAMA_BUILD_DIR") or tmp_path_factory.mktemp("llama_reference_build"))
    # Wheel-test containers inherit build-job launcher variables but need not have sccache.
    build_env = os.environ.copy()
    build_env.pop("CMAKE_C_COMPILER_LAUNCHER", None)
    build_env.pop("CMAKE_CXX_COMPILER_LAUNCHER", None)
    run_reference_command(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER_LAUNCHER=",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=",
        ],
        timeout=300,
        env=build_env,
    )
    run_reference_command(
        ["cmake", "--build", str(build), "--target", "genai-gguf-generate", "genai-gguf-reference", "-j", "4"],
        timeout=900,
        env=build_env,
    )
    return build / "bin" / "genai-gguf-generate", build / "bin" / "genai-gguf-reference"


@pytest.fixture(scope="module")
def numeric_tokenizer(tmp_path_factory):
    # llama.cpp fixtures have no text vocabulary. Supply one only to satisfy the GenAI
    # constructor; all model inputs and comparisons use token IDs, never decoded text.
    backend = HFTokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0"))
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0", pad_token="t0", eos_token="t127")
    tokenizer, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    directory = tmp_path_factory.mktemp("numeric_tokenizer")
    ov.save_model(tokenizer, directory / "openvino_tokenizer.xml")
    ov.save_model(detokenizer, directory / "openvino_detokenizer.xml")
    hf_tokenizer.save_pretrained(directory)
    return genai.Tokenizer(directory)


@pytest.fixture(scope="module", params=MODEL_CASES)
def generated_model(request, llama_reference_tools, tmp_path_factory):
    arch, variant = request.param
    generator, oracle = llama_reference_tools
    directory = tmp_path_factory.mktemp(f"gguf_{arch}")
    run_reference_command([str(generator), arch, str(directory), variant])
    model = directory / f"{arch}-{variant}.gguf"
    assert model.is_file(), f"Pinned llama.cpp did not export {model.name}; do not silently drop architecture coverage"
    spec = directory / "inputs.json"
    spec.write_text(json.dumps({"prompts": PROMPTS, "max_new_tokens": NEW_TOKENS}))
    reference_path = model.with_suffix(".json")
    run_reference_command([str(oracle), str(model), str(spec), str(reference_path)])
    references = json.loads(reference_path.read_text())
    assert [ref["input_ids"] for ref in references] == PROMPTS
    for ref in references:
        assert len(ref["generated_ids"]) == len(ref["log_probs"]) == NEW_TOKENS
        assert all(0 <= token < 128 for token in ref["generated_ids"])
        assert np.isfinite(ref["log_probs"]).all()
    return arch, variant, model, references


def properties():
    return {
        **get_default_llm_properties(),
        "KV_CACHE_PRECISION": "f32",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
        "GGUF_READER": "FRONTEND",
    }


def generation_config():
    return genai.GenerationConfig(max_new_tokens=NEW_TOKENS, ignore_eos=True, apply_chat_template=False)


def mark_backend_limitation(request, arch, variant, backend):
    if arch == "qwen35" and backend != "SDPA":
        pytest.skip("Qwen3.5 recurrent state currently supports SDPA only")
    if arch == "gemma2" and backend == "PA":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="Soft-capped Gemma2 attention has no SDPA node for explicit PA conversion",
            )
        )
    elif variant == "moe" and backend != "SDPA" and arch in {"bailingmoe2", "ernie4_5-moe"}:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="GGUF hybrid MoE PA prefill disagrees with llama.cpp CPU; SDPA passes",
            )
        )
    elif (
        variant == "moe"
        and backend != "SDPA"
        and arch in {"llama", "minicpm", "mistral3", "olmoe", "qwen3moe", "gpt-oss"}
    ):
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="GGUF MoE PA prefill has token-axis shape errors in PagedAttention/Reshape",
            )
        )
    elif arch == "gpt-oss":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="Generated GPT-OSS F32 logits diverge from llama.cpp CPU before decoding",
            )
        )


@pytest.mark.parametrize("backend", ["SDPA", "PA", "AUTO"])
def test_generated_gguf_prefill_decode(request, generated_model, numeric_tokenizer, backend):
    arch, variant, model, references = generated_model
    mark_backend_limitation(request, arch, variant, backend)
    config = properties()
    if backend != "AUTO":
        config["ATTENTION_BACKEND"] = backend
    if backend == "PA":
        config["scheduler_config"] = genai.SchedulerConfig()
    pipeline = genai.LLMPipeline(model, numeric_tokenizer, "CPU", **config)
    for reference in references:
        inputs = ov.Tensor(np.array([reference["input_ids"]], dtype=np.int64))
        actual = pipeline.generate(inputs, generation_config()).tokens
        assert actual == [reference["generated_ids"]], f"{model.name}: {reference['input_ids']} ({backend})"


def test_generated_gguf_continuous_batching(request, generated_model, numeric_tokenizer):
    arch, variant, model, references = generated_model
    mark_backend_limitation(request, arch, variant, "PA")
    scheduler = genai.SchedulerConfig()
    scheduler.max_num_batched_tokens = 16
    scheduler.num_kv_blocks = 64
    config = generation_config()
    config.logprobs = 1
    pipeline = genai.ContinuousBatchingPipeline(model, numeric_tokenizer, scheduler, "CPU", **properties())
    handles = [
        pipeline.add_request(i, ov.Tensor(np.array([ref["input_ids"]], dtype=np.int64)), config)
        for i, ref in enumerate(references)
    ]
    steps = 0
    while pipeline.has_non_finished_requests():
        pipeline.step()
        steps += 1
        assert steps < 128, f"{model.name}: scheduler did not finish"
    for handle, reference in zip(handles, references):
        outputs = handle.read_all()
        assert len(outputs) == 1
        assert outputs[0].generated_ids == reference["generated_ids"], model.name
        np.testing.assert_allclose(outputs[0].generated_log_probs, reference["log_probs"], atol=1e-4, rtol=0)
