# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate tiny GGUFs and CPU references with the same pinned llama.cpp build."""

import hashlib
import io
import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import pytest
import requests
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

import openvino as ov
import openvino_genai as genai
from utils.constants import get_default_llm_properties
from utils.hugging_face import convert_and_save_tokenizer
from utils.network import retry_request


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="CPU reference build is validated on Linux")

# Native frontend architectures that the pinned llama.cpp generator can export.
ARCHITECTURES = [
    "bailingmoe2",
    "ernie4_5-moe",
    "exaone4",
    "gemma",
    "gemma2",
    "glm4moe",
    "gpt-oss",
    "hunyuan-dense",
    "hunyuan-moe",
    "jais2",
    "lfm2",
    "lfm2moe",
    "llama",
    "maincoder",
    "mamba2",
    "minicpm",
    "minimax-m2",
    "mistral3",
    "nemotron_h",
    "olmoe",
    "phi3",
    "qwen2",
    "qwen3",
    "qwen35",
    "qwen35moe",
    "qwen3moe",
    "smollm3",
]
MOE_ARCHITECTURES = {
    "bailingmoe2",
    "ernie4_5-moe",
    "glm4moe",
    "gpt-oss",
    "hunyuan-moe",
    "lfm2moe",
    "minimax-m2",
    "olmoe",
    "qwen35moe",
    "qwen3moe",
}
# The pinned llama.cpp saver cannot export these architectures. Use the OpenVINO GGUF frontend's
# accuracy fixtures instead: two- to four-layer F32 GGUFs with a 32-token vocabulary, pinned by
# OpenVINO revision and SHA-256. The pinned llama.cpp still produces their references.
OPENVINO_FIXTURE_REVISION = "0a3e3fb5e92b7c8bec70dece0bcaafe09703a1f6"
OPENVINO_FIXTURE_URL = (
    "https://raw.githubusercontent.com/openvinotoolkit/openvino/{revision}"
    "/src/frontends/gguf/tests/test_data/arch_accuracy/{name}.npz"
)
OPENVINO_FIXTURES = {
    "afmoe": ("afmoe", "64fdbb328ca492c34cc2f2c21491c9d0e5ce9ae96b34d25cd1910a203f5d8558"),
    "deepseek2-ocr": ("deepseek2-ocr", "43070fd56795f7f28e89e8c2023098b5aac802d8dc301ead2a81b1e03f3fb399"),
    "exaone-moe": ("exaone-moe", "4b380bbe5ebaede2f47f7f955822b4eab97049a9fcf92445b7aa1df6e9210fe2"),
    "gemma3": ("gemma3", "08381062a60f7191d1442ff7062a2a14699f351b771a4139f56253dde7bb9882"),
    "gemma4-moe": ("gemma4", "5d036080b72d1f6f2fb060678f59b4266399c2aee0a8002273cfc971b6d6ca50"),
    "gemma4-mqa": ("gemma4", "4080f624253e7b45d8c04b405ef4be6a2533fddc395d4e3508265f38149ecc42"),
    "gemma4-ple": ("gemma4", "a6bdc09e3b430d03720de18e8f6cdeadc3f44e0f72b8bf13811c2987bcdb4810"),
    "lfm3": ("lfm3", "953a987428a65a47e277f27ae94c5463cf6f8ac8780792982c26012b943459a7"),
    "mellum": ("mellum", "70c5df72cd62ecb42c3cbfc96cf1c468e6d74dcfe0b19e3e3e9fb1aa0ea4dc4b"),
    "muse-glimmer": ("muse-glimmer", "9fb2126e4f32c7d7d31f8e8cd6ef3d5395df71107185e438ccb1d07a327b29af"),
    "plamo3": ("plamo3", "12aa3e2f509109cdf69ed5959dfcdaeb89e239a16f0442da09dcdcca10390533"),
}
# The pinned llama.cpp predates LFM3. Its fixture stores llama.cpp logits after the token
# batches [1, 2, 3], [4], [5, 6]; with preserved state they score these prefixes.
STORED_LOGITS_PROMPTS = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
MODEL_CASES = (
    [pytest.param((arch, arch, "moe" if arch in MOE_ARCHITECTURES else "dense"), id=arch) for arch in ARCHITECTURES]
    + [pytest.param((arch, arch, "moe"), id=f"{arch}-moe") for arch in ("llama", "minicpm", "mistral3")]
    + [pytest.param((name, arch, "fixture"), id=name) for name, (arch, _) in OPENVINO_FIXTURES.items()]
)
# Token IDs stay below the 32-token fixture vocabulary. The last prompt exceeds a CPU KV block
# and shares its first 24 tokens with the previous prompt.
PROMPTS = [[1, 2, 3], [7, 4, 9, 5, 11], list(range(1, 25)), list(range(1, 25)) + list(range(24, 0, -1))[:24]]
NEW_TOKENS = 16
CPU_KV_BLOCK_SIZE = 32


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
    directory = tmp_path_factory.mktemp("numeric_tokenizer")
    convert_and_save_tokenizer(hf_tokenizer, directory)
    hf_tokenizer.save_pretrained(directory)
    return genai.Tokenizer(directory)


def download_openvino_fixture(name):
    _, sha256 = OPENVINO_FIXTURES[name]
    url = OPENVINO_FIXTURE_URL.format(revision=OPENVINO_FIXTURE_REVISION, name=name)

    def download():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content

    data = retry_request(download)
    assert hashlib.sha256(data).hexdigest() == sha256, f"{url}: SHA-256 mismatch"
    with np.load(io.BytesIO(data)) as fixture:
        return fixture["model"].tobytes(), fixture["logits"]


def stored_logits_references(logits):
    assert logits.shape == (len(STORED_LOGITS_PROMPTS), logits.shape[1])
    references = []
    for prompt, row in zip(STORED_LOGITS_PROMPTS, logits.astype(np.float64)):
        token = int(np.argmax(row))
        log_prob = -np.log(np.exp(row - row[token]).sum())
        references.append({"input_ids": prompt, "generated_ids": [token], "log_probs": [log_prob]})
    return references


@pytest.fixture(scope="module", params=MODEL_CASES)
def generated_model(request, llama_reference_tools, tmp_path_factory):
    name, arch, variant = request.param
    generator, oracle = llama_reference_tools
    directory = tmp_path_factory.mktemp(f"gguf_{name}")
    if variant == "fixture":
        weights, logits = download_openvino_fixture(name)
        model = directory / f"{name}.gguf"
        model.write_bytes(weights)
        if arch == "lfm3":
            return arch, variant, model, stored_logits_references(logits)
    else:
        run_reference_command([str(generator), arch, str(directory), variant])
        model = directory / f"{arch}-{variant}.gguf"
        assert model.is_file(), (
            f"Pinned llama.cpp did not export {model.name}; do not silently drop architecture coverage"
        )
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


def generation_config(max_new_tokens=NEW_TOKENS):
    return genai.GenerationConfig(max_new_tokens=max_new_tokens, ignore_eos=True, apply_chat_template=False)


# (predicate, raises, reason) applied in order; the first match marks the test xfail.
BACKEND_LIMITATIONS = [
    (
        lambda arch, variant, backend: arch == "mamba2" and backend == "PA",
        RuntimeError,
        "Pure Mamba2 has no SDPA node for explicit PA conversion",
    ),
    (
        lambda arch, variant, backend: arch == "nemotron_h" and backend == "PA",
        RuntimeError,
        "GGUF Nemotron-H PA conversion fails the SelectiveSSM fusion-count check",
    ),
    (
        lambda arch, variant, backend: arch == "gemma2" and backend == "PA",
        RuntimeError,
        "Soft-capped Gemma2 attention has no SDPA node for explicit PA conversion",
    ),
]


def mark_backend_limitation(request, arch, variant, backend):
    for matches, raises, reason in BACKEND_LIMITATIONS:
        if matches(arch, variant, backend):
            request.node.add_marker(pytest.mark.xfail(strict=False, raises=raises, reason=reason))
            return


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
        actual = pipeline.generate(inputs, generation_config(len(reference["generated_ids"]))).tokens
        assert actual == [reference["generated_ids"]], f"{model.name}: {reference['input_ids']} ({backend})"


CB_SCHEDULERS = {
    "chunked_prefill": {"max_num_batched_tokens": 16, "num_kv_blocks": 64},
    "whole_prompts": {"dynamic_split_fuse": False, "num_kv_blocks": 64},
    # Fits the longest request but not all requests together, forcing preemption.
    "preemption": {"max_num_batched_tokens": 16, "num_kv_blocks": 3},
    # The second round reuses cached blocks of the prompt exceeding one KV block.
    "prefix_caching": {
        "max_num_batched_tokens": 16,
        "num_kv_blocks": 64,
        "enable_prefix_caching": True,
        "cache_interval_multiplier": 1,
    },
}


def run_continuous_batching(pipeline, references, model_name, rtol):
    handles = []
    for i, reference in enumerate(references):
        config = generation_config(len(reference["generated_ids"]))
        config.logprobs = 1
        inputs = ov.Tensor(np.array([reference["input_ids"]], dtype=np.int64))
        handles.append(pipeline.add_request(i, inputs, config))
    steps = 0
    while pipeline.has_non_finished_requests():
        pipeline.step()
        steps += 1
        assert steps < 512, f"{model_name}: scheduler did not finish"
    for handle, reference in zip(handles, references):
        outputs = handle.read_all()
        assert len(outputs) == 1
        assert outputs[0].generated_ids == reference["generated_ids"], f"{model_name}: {reference['input_ids']}"
        np.testing.assert_allclose(outputs[0].generated_log_probs, reference["log_probs"], atol=1e-4, rtol=rtol)
    return handles


def kv_blocks_required(reference):
    return -(-(len(reference["input_ids"]) + len(reference["generated_ids"])) // CPU_KV_BLOCK_SIZE)


@pytest.mark.parametrize("scenario", CB_SCHEDULERS)
def test_generated_gguf_continuous_batching(request, generated_model, numeric_tokenizer, scenario):
    arch, variant, model, references = generated_model
    mark_backend_limitation(request, arch, variant, "PA")
    if scenario in {"preemption", "prefix_caching"} and references[-1]["input_ids"] != PROMPTS[-1]:
        pytest.skip("Stored-logits references are too short to fill a KV block")
    if scenario == "prefix_caching" and arch in {"qwen35", "qwen35moe"}:
        reason = "GatedDeltaNet state restored from a prefix-cache checkpoint drifts by about 3e-3 in log probs"
        request.node.add_marker(pytest.mark.xfail(strict=False, raises=AssertionError, reason=reason))
    scheduler = genai.SchedulerConfig()
    for name, value in CB_SCHEDULERS[scenario].items():
        setattr(scheduler, name, value)
    pipeline = genai.ContinuousBatchingPipeline(model, numeric_tokenizer, scheduler, "CPU", **properties())
    # F32 reduction order differs from llama.cpp. OpenVINO fixtures drift more (up to about 4e-3
    # relative); the frontend accepts their logits up to an NMSE of 1e-5.
    rtol = 5e-3 if variant == "fixture" else 1e-4
    scenario_references = references
    if scenario == "preemption":
        scenario_references = sorted(
            references,
            key=lambda reference: (
                len(reference["input_ids"]) < len(PROMPTS[2]),
                len(reference["input_ids"]),
            ),
        )
        assert max(map(kv_blocks_required, references)) <= scheduler.num_kv_blocks
        assert sum(map(kv_blocks_required, references)) > scheduler.num_kv_blocks
    run_continuous_batching(pipeline, scenario_references, model.name, rtol)
    if scenario == "prefix_caching":
        handles = run_continuous_batching(pipeline, references, model.name, rtol)
        hits = handles[-1].get_perf_metrics().get_num_prefix_cache_hit_tokens()
        assert hits >= CPU_KV_BLOCK_SIZE, f"{model.name}: prefix cache reused {hits} tokens"
    if scenario == "preemption":
        assert pipeline.get_metrics().max_cache_usage >= 99.0, f"{model.name}: KV cache was not under pressure"


def test_generated_gguf_recurrent_chat_cancel(generated_model, numeric_tokenizer):
    arch, variant, model, references = generated_model
    if arch not in {"mamba2", "nemotron_h"}:
        pytest.skip("Recurrent GGUF chat regression")
    config = {**properties(), "ATTENTION_BACKEND": "SDPA"}
    pipeline = genai.LLMPipeline(model, numeric_tokenizer, "CPU", **config)
    inputs = lambda ids: ov.Tensor(np.array([ids], dtype=np.int64))
    prompt = references[0]["input_ids"]
    pipeline.start_chat()
    try:
        first = pipeline.generate(inputs(prompt), generation_config()).tokens[0]
        pipeline.generate(
            inputs([4, 5]),
            generation_config(),
            streamer=lambda text: genai.StreamingStatus.CANCEL,
        )
        actual = pipeline.generate(inputs([6, 7]), generation_config()).tokens
    finally:
        pipeline.finish_chat()

    reference = genai.LLMPipeline(model, numeric_tokenizer, "CPU", **config)
    expected = reference.generate(inputs(prompt + first + [6, 7]), generation_config()).tokens
    assert actual == expected
