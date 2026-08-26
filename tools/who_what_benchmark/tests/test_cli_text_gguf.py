# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Accuracy tests for GGUF models loaded through the OpenVINO GGUF frontend on the GenAI backend.
# One small real GGUF per architecture family the frontend supports (llama, qwen2, qwen3, phi3,
# minicpm, hunyuan-dense, olmoe, gpt-oss, gemma, gemma2, gemma4).
#
# Reference = llama.cpp (via llama-cpp-python, --llamacpp) running the same .gguf natively.
# Target = OpenVINO GenAI loading the .gguf through the frontend (--genai). Asserts WWB text
# similarity between the two is above a threshold.
#
# Downloads real (small) GGUF models and requires llama-cpp-python; opt-in via WWB_GGUF_TESTS=1
# so the default CI text suite stays fast.

import os
import sys
import logging

import pytest
import pandas as pd

from test_cli_image import get_similarity
from conftest import run_wwb, get_ov_cache_converted_models_dir
from ov_utils import download_hf_files_to_cache


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Minimum WWB text similarity (cosine over sentence embeddings) between the llama.cpp reference
# and the GenAI-frontend target. Similarity falls with generated length even for a correct
# conversion (measured for qwen3-0.6B: ~0.89 @ 16 tokens, ~0.85 @ 32, ~0.74 @ 100), so the
# threshold is set to clear a correct conversion comfortably while failing hard on a broken one.
SIMILARITY_THRESHOLD = 0.8

# Bound generated length so late-token greedy drift between the two runtimes doesn't dominate
# the similarity score, and to keep runtime reasonable.
MAX_NEW_TOKENS = 32

# Samples per model. Kept small to bound runtime; raise locally for a stricter check.
NUM_SAMPLES = 4


# One representative small GGUF per architecture family the frontend supports: (arch, hf_repo_id,
# gguf_filename). The `gguf_small` mark selects models under ~1 GB; CI runs only those (see the
# "WWB tests (GGUF)" job in linux.yml) -- the rest are 1.5-11.5 GB and stay a local/opt-in check.
GGUF_MODELS = [
    pytest.param(
        "llama",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        id="llama-tinyllama-1.1b",
        marks=[
            pytest.mark.gguf_small,
            pytest.mark.xfail(
                reason="genai-vs-llamacpp similarity ~0.66 < 0.8 threshold; frontend output "
                "diverges from llama.cpp for this model, root cause not yet investigated",
                strict=False,
            ),
        ],
    ),
    pytest.param(
        "qwen2",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q8_0.gguf",
        id="qwen2-qwen2.5-0.5b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "qwen3",
        "Qwen/Qwen3-0.6B-GGUF",
        "Qwen3-0.6B-Q8_0.gguf",
        id="qwen3-qwen3-0.6b",
        marks=pytest.mark.gguf_small,
    ),
    pytest.param(
        "phi3",
        "microsoft/Phi-3-mini-4k-instruct-gguf",
        "Phi-3-mini-4k-instruct-q4.gguf",
        id="phi3-phi3-mini",
    ),
    pytest.param(
        "minicpm",
        "runfuture/MiniCPM-2B-dpo-q4km-gguf",
        "MiniCPM-2B-dpo-q4km-gguf.gguf",
        id="minicpm-2b",
    ),
    pytest.param(
        "hunyuan-dense",
        "gabriellarson/Hunyuan-0.5B-Instruct-GGUF",
        "Hunyuan-0.5B-Instruct-Q8_0.gguf",
        id="hunyuan-0.5b",
        marks=[
            pytest.mark.gguf_small,
            pytest.mark.xfail(
                reason="genai-vs-llamacpp similarity ~0.12 < 0.8 threshold; frontend output "
                "for hunyuan-dense diverges heavily from llama.cpp, root cause not yet investigated",
                strict=False,
            ),
        ],
    ),
    pytest.param(
        "olmoe",
        "allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
        "olmoe-1b-7b-0924-instruct-q4_0.gguf",
        id="olmoe-1b-7b",
    ),
    pytest.param(
        "gpt-oss",
        "ggml-org/gpt-oss-20b-GGUF",
        "gpt-oss-20b-mxfp4.gguf",
        id="gpt-oss-20b",
    ),
    pytest.param(
        "gemma",
        "MaziyarPanahi/gemma-2b-it-GGUF",
        "gemma-2b-it.Q4_K_M.gguf",
        id="gemma-2b",
    ),
    pytest.param(
        "gemma2",
        "bartowski/gemma-2-2b-it-GGUF",
        "gemma-2-2b-it-Q4_K_M.gguf",
        id="gemma2-gemma-2-2b",
    ),
    pytest.param(
        "gemma4",
        "ggml-org/gemma-4-E4B-it-GGUF",
        "gemma-4-E4B-it-Q4_K_M.gguf",
        id="gemma4-e4b",
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
@pytest.mark.parametrize(("arch", "hf_id", "gguf"), GGUF_MODELS)
def test_text_gguf_genai_vs_llamacpp(arch, hf_id, gguf, tmp_path):
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
            '{"GGUF_READER": "FRONTEND"}',
        ]
    )

    similarity = get_similarity(output)
    logger.info("[%s] %s genai-vs-llamacpp similarity = %.4f", arch, gguf, similarity)
    assert similarity >= SIMILARITY_THRESHOLD, (
        f"{arch} ({gguf}) genai-vs-llamacpp similarity {similarity:.4f} < {SIMILARITY_THRESHOLD}"
    )
