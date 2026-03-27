# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging
import sys
import re
from pathlib import Path

from conftest import convert_model, run_wwb
from ov_utils import get_ov_cache_dir
from huggingface_hub import hf_hub_download


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_speaker_embedding():
    """Download speaker embedding file for speech generation tests."""
    speaker_embeddings_cache_dir = get_ov_cache_dir() / "test_data" / "speaker_embeddings"
    speaker_embeddings_cache_dir.mkdir(parents=True, exist_ok=True)

    filename = "cmu_us_slt_arctic-wav-arctic_a0508.bin"
    embedding_file = Path(
        hf_hub_download(
            repo_id="Xenova/cmu-arctic-xvectors-extracted",
            filename=filename,
            repo_type="dataset",
            local_dir=speaker_embeddings_cache_dir,
        )
    )

    assert embedding_file.exists(), f"Speaker embedding file wasn't downloaded: {embedding_file}"
    return str(embedding_file)


def get_overall_score(output: str) -> float:
    metric_pattern = r"INFO:whowhatbench\.wwb:.*overall score"
    m = re.search(metric_pattern, output, re.DOTALL)
    assert m, "Could not find metrics header in output"

    substr = output[m.end() :]
    float_pattern = r"[-+]?\d*\.\d+"
    matches = re.findall(float_pattern, substr)
    return float(matches[-1])


def run_test(model_id, model_type, speaker_embeddings, optimum_threshold, genai_threshold, tmp_path):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    # Collect reference with HF model
    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--speaker_embeddings",
            speaker_embeddings,
            "--hf",
        ]
    )

    # test Optimum
    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--speaker_embeddings",
            speaker_embeddings,
        ]
    )

    optimum_score = get_overall_score(output)
    if optimum_threshold is not None:
        assert optimum_score >= optimum_threshold

    # test GenAI
    output = run_wwb(
        [
            "--target-model",
            MODEL_PATH,
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--speaker_embeddings",
            speaker_embeddings,
            "--genai",
            "--output",
            tmp_path,
        ]
    )

    genai_score = get_overall_score(output)
    if genai_threshold is not None:
        assert genai_score >= genai_threshold

    # test w/o models (only compute metrics on pre-generated audio)
    output = run_wwb(
        [
            "--target-data",
            tmp_path / "target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--model-type",
            model_type,
        ]
    )
    genai_score_no_gen = get_overall_score(output)
    assert genai_score_no_gen == genai_score


@pytest.mark.speech_generation
@pytest.mark.speecht5
@pytest.mark.parametrize(
    ("model_id", "model_type", "optimum_threshold", "genai_threshold"),
    [
        ("microsoft/speecht5_tts", "speech-generation", 0.90, 0.90),
    ],
)
def test_tts_speecht5(model_id, model_type, optimum_threshold, genai_threshold, tmp_path):
    speaker_embeddings = get_speaker_embedding()
    run_test(model_id, model_type, speaker_embeddings, optimum_threshold, genai_threshold, tmp_path)
