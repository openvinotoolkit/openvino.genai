# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Accuracy tests for GGUF models loaded through the OpenVINO GGUF frontend on the GenAI backend.
# One real GGUF per verified native architecture in OpenVINO's supported_models.md.
# Vision/audio components of multimodal checkpoints are not exercised here.
#
# Reference = llama.cpp (via llama-cpp-python, --llamacpp) running the same .gguf natively.
# Target = OpenVINO GenAI loading the .gguf through the frontend (--genai). Asserts WWB text
# similarity between the two is above a threshold.
# The reference_precision cases use OpenVINO's documented reference-check settings:
# FP32 inference, FP16 KV cache, no activation quantization and floating-point Q4_K zero points.
# Existing cases retain their production settings, except TinyLlama's strict Q4_K decoding.
#
# Downloads real GGUF checkpoints and requires llama-cpp-python; opt-in via WWB_GGUF_TESTS=1
# so the default CI text suite stays fast.

import os
import sys
import logging
import json

import pytest
import pandas as pd

from test_cli_image import get_similarity
from conftest import run_wwb, get_ov_cache_converted_models_dir
from ov_utils import download_hf_files_to_cache


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Minimum WWB text similarity (cosine over sentence embeddings) between llama.cpp and GenAI.
# Cases without reference_precision retain the CPU plugin's dynamic-quantization default;
# FP32 inference alone does not disable activation quantization. This can contribute to token
# drift, but disabling it does not eliminate differences in weight decoding, KV-cache precision
# or floating-point arithmetic. Autoregressive decoding can amplify those differences.
# This threshold is an end-to-end regression check, not proof of conversion correctness.
SIMILARITY_THRESHOLD = 0.8

# Bound generated length so late-token greedy drift between the two runtimes doesn't dominate
# the similarity score, and to keep runtime reasonable.
MAX_NEW_TOKENS = 32

# Samples per model. Kept small to bound runtime; raise locally for a stricter check.
NUM_SAMPLES = 4


# (architecture, Hub repository, filename, reference_precision).
# Catalog: openvinotoolkit/openvino at 38286c35d9, src/frontends/gguf/docs/supported_models.md.
# The `gguf_small` mark selects downloads under ~1 GB for an opt-in subset run.
# Precommit uses tests/python_tests/test_gguf_frontend.py with tiny models instead.
GGUF_MODELS = [
    pytest.param(
        "llama",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        False,
        id="llama-tinyllama-1.1b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "qwen2",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q8_0.gguf",
        False,
        id="qwen2-qwen2.5-0.5b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "qwen3",
        "Qwen/Qwen3-0.6B-GGUF",
        "Qwen3-0.6B-Q8_0.gguf",
        False,
        id="qwen3-qwen3-0.6b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "phi3",
        "microsoft/Phi-3-mini-4k-instruct-gguf",
        "Phi-3-mini-4k-instruct-q4.gguf",
        False,
        id="phi3-phi3-mini",
    ),
    pytest.param(
        "minicpm",
        "runfuture/MiniCPM-2B-dpo-q4km-gguf",
        "MiniCPM-2B-dpo-q4km-gguf.gguf",
        False,
        id="minicpm-2b",
    ),
    pytest.param(
        "hunyuan-dense",
        "gabriellarson/Hunyuan-0.5B-Instruct-GGUF",
        "Hunyuan-0.5B-Instruct-Q8_0.gguf",
        False,
        id="hunyuan-0.5b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "olmoe",
        "allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
        "olmoe-1b-7b-0924-instruct-q4_0.gguf",
        False,
        id="olmoe-1b-7b",
    ),
    pytest.param(
        "gpt-oss",
        "ggml-org/gpt-oss-20b-GGUF",
        "gpt-oss-20b-mxfp4.gguf",
        False,
        id="gpt-oss-20b",
    ),
    pytest.param(
        "gemma",
        "MaziyarPanahi/gemma-2b-it-GGUF",
        "gemma-2b-it.Q4_K_M.gguf",
        False,
        id="gemma-2b",
    ),
    pytest.param(
        "gemma2",
        "bartowski/gemma-2-2b-it-GGUF",
        "gemma-2-2b-it-Q4_K_M.gguf",
        False,
        id="gemma2-gemma-2-2b",
    ),
    pytest.param(
        "gemma4",
        "ggml-org/gemma-4-E4B-it-GGUF",
        "gemma-4-E4B-it-Q4_K_M.gguf",
        False,
        id="gemma4-e4b",
    ),
    pytest.param(
        "bailingmoe2",
        "bartowski/inclusionAI_Ling-mini-2.0-GGUF",
        "inclusionAI_Ling-mini-2.0-Q2_K.gguf",
        True,
        id="bailingmoe2-ling-mini-2.0",
    ),
    pytest.param(
        "deepseek2-ocr",
        "aditya00196/DeepSeek-OCR-2-Q4_K_M.gguf",
        "DeepSeek-OCR-2-Q4_K_M.gguf",
        True,
        id="deepseek2-ocr-language-backbone",
    ),
    pytest.param(
        "ernie4_5-moe",
        "bartowski/baidu_ERNIE-4.5-21B-A3B-PT-GGUF",
        "baidu_ERNIE-4.5-21B-A3B-PT-Q4_K_M.gguf",
        True,
        id="ernie4_5-moe-21b-a3b",
    ),
    pytest.param(
        "exaone4",
        "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF",
        "EXAONE-4.0-1.2B-Q4_K_M.gguf",
        True,
        id="exaone4-1.2b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "gemma3",
        "unsloth/gemma-3-1b-it-GGUF",
        "gemma-3-1b-it-Q4_K_M.gguf",
        True,
        id="gemma3-1b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "maincoder",
        "mradermacher/Maincoder-1B-GGUF",
        "Maincoder-1B.Q4_K_M.gguf",
        True,
        id="maincoder-1b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "mellum",
        "bartowski/Mellum2-12B-A2.5B-Instruct-GGUF",
        "Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf",
        True,
        id="mellum2-12b-a2.5b",
    ),
    pytest.param(
        "mistral3",
        "bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF",
        "mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
        True,
        id="mistral3-ministral-3b",
    ),
    pytest.param(
        "muse-glimmer",
        "bartowski/Muse-Glimmer-30B-GGUF",
        "Muse-Glimmer-30B-Q4_0.gguf",
        True,
        id="muse-glimmer-30b",
    ),
    pytest.param(
        "qwen35",
        "ggml-org/Qwen3.5-0.8B-GGUF",
        "Qwen3.5-0.8B-Q8_0.gguf",
        True,
        id="qwen35-0.8b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "qwen3moe",
        "mradermacher/Qwen3-0.9B-A0.6B-GGUF",
        "Qwen3-0.9B-A0.6B.Q4_K_M.gguf",
        True,
        id="qwen3moe-0.9b-a0.6b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "smollm3",
        "bartowski/HuggingFaceTB_SmolLM3-3B-GGUF",
        "HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf",
        True,
        id="smollm3-3b",
    ),
]


pytestmark = pytest.mark.skipif(
    os.environ.get("WWB_GGUF_TESTS", "0") != "1",
    reason="GGUF accuracy tests download large models and need llama-cpp-python "
    "(pip install 'whowhatbench[gguf]'); set WWB_GGUF_TESTS=1 to enable.",
)


def _download_gguf(repo_id, gguf):
    """Fetch the .gguf into the shared converted-models cache; return its directory.

    The GGUF tokenizer for both backends is built from the file itself, so a single local
    copy serves both the llama.cpp reference and the GenAI target.
    """
    dest = get_ov_cache_converted_models_dir() / ("gguf_" + repo_id.replace("/", "_"))
    download_hf_files_to_cache(repo_id, dest, [gguf])
    return str(dest)


@pytest.mark.skipif(sys.platform == "darwin", reason="CVS-168882: sporadic segfault on macOS")
@pytest.mark.skipif(sys.platform == "win32", reason="CVS-174065")
@pytest.mark.parametrize(("arch", "hf_id", "gguf", "reference_precision"), GGUF_MODELS)
def test_text_gguf_genai_vs_llamacpp(arch, hf_id, gguf, reference_precision, tmp_path):
    """Reference = llama.cpp running the .gguf; target = GenAI loading the same .gguf through
    the OpenVINO frontend. Assert their generations are similar."""
    pytest.importorskip("llama_cpp", reason="llama-cpp-python is required for the reference")

    gguf_dir = _download_gguf(hf_id, gguf)
    gt_data = tmp_path / "gt.csv"

    # 1) Ground truth from llama.cpp running the .gguf directly; --tokenizer points at the hub
    #    repo so WWB can build a tokenizer for its own prompt bookkeeping.
    # --base-model must be the directory: --gguf-file is joined onto it (same convention as the
    # --genai run below), not the full file path -- load_text_llamacpp_pipeline joins the two
    # itself and a full path here would append the filename twice.
    run_wwb(
        [
            "--base-model",
            gguf_dir,
            "--gt-data",
            gt_data,
            "--tokenizer",
            hf_id,
            "--gguf-file",
            gguf,
            "--num-samples",
            str(NUM_SAMPLES),
            "--max_new_tokens",
            str(MAX_NEW_TOKENS),
            "--device",
            "CPU",
            "--short-prompt",
            # Compare raw continuations: llama.cpp's WWB path can't apply a chat template (it runs
            # with tokenizer=None), so disable it on both sides to keep the comparison apples-to-apples.
            "--omit-chat-template",
            "--llamacpp",
        ]
    )
    data = pd.read_csv(gt_data)
    assert len(data["prompts"].values) == NUM_SAMPLES

    # 2) Target = GenAI loading the same .gguf through the frontend; compare to ground truth.
    # GGUF_READER defaults to the legacy reader (llama/qwen2/qwen3 only; see llm_pipeline.hpp),
    # so force the frontend explicitly -- this test is about the frontend's whole architecture
    # range, most of which the legacy reader can't load at all.
    # TinyLlama still falls below the threshold with production Q4_K requantization.
    is_tinyllama = hf_id == "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    strict_q4_k = is_tinyllama or reference_precision
    # Keep inference precision independent of the runner's CPU capabilities.
    ov_config = {"GGUF_READER": "FRONTEND", "INFERENCE_PRECISION_HINT": "f32"}
    if reference_precision:
        ov_config.update(KV_CACHE_PRECISION="f16", DYNAMIC_QUANTIZATION_GROUP_SIZE=0)
    # Qwen3.5's recurrent state supports only greedy, batch-one SDPA decoding.
    # This Qwen3-MoE checkpoint currently hits a PA KV-cache shape mismatch at inference.
    if arch in ("qwen35", "qwen3moe"):
        ov_config["ATTENTION_BACKEND"] = "SDPA"
    output = run_wwb(
        [
            "--target-model",
            gguf_dir,
            "--gguf-file",
            gguf,
            "--tokenizer",
            hf_id,
            "--gt-data",
            gt_data,
            "--num-samples",
            str(NUM_SAMPLES),
            "--max_new_tokens",
            str(MAX_NEW_TOKENS),
            "--device",
            "CPU",
            "--short-prompt",
            "--omit-chat-template",
            "--genai",
            "--ov-config",
            json.dumps(ov_config),
            "--output",
            str(tmp_path / "target"),
        ],
        env={"OV_GGUF_Q4_K_ZP_F16": "1" if strict_q4_k else "0"},
    )

    similarity = get_similarity(output)
    logger.info("[%s] %s genai-vs-llamacpp similarity = %.4f", arch, gguf, similarity)
    if is_tinyllama and similarity < SIMILARITY_THRESHOLD:
        pytest.xfail(
            f"TinyLlama Q4_K_M frontend similarity {similarity:.4f} < {SIMILARITY_THRESHOLD} "
            "on CI despite OV_GGUF_Q4_K_ZP_F16=1 and FP32 inference"
        )
    assert similarity >= SIMILARITY_THRESHOLD, (
        f"{arch} ({gguf}) genai-vs-llamacpp similarity {similarity:.4f} < {SIMILARITY_THRESHOLD}"
    )
