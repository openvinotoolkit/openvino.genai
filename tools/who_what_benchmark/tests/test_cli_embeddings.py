import sys
import pytest
import shutil
from pathlib import Path
from test_cli_image import get_similarity
from conftest import convert_model, run_wwb
from profile_utils import _log, _stage


def remove_artifacts(artifacts_path: Path):
    shutil.rmtree(artifacts_path)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        pytest.param(
            "BAAI/bge-small-en-v1.5",
            "text-embedding",
            marks=pytest.mark.xfail(sys.platform == "darwin", reason="Hangs. Ticket 175534", run=False),
        ),
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding"),
    ],
)
@pytest.mark.xfail(sys.platform == "win32", reason="Ticket 178790", run=False)
def test_embeddings_basic(model_id, model_type, tmp_path):
    _log(f"test_embeddings_basic: model_id={model_id} model_type={model_type}")
    GT_FILE = tmp_path / "gt.csv"
    with _stage("convert_model"):
        MODEL_PATH = convert_model(model_id)
    _log(f"Model converted to: {MODEL_PATH}")
    SIMILARITY_THRESHOLD = 0.99

    # Collect reference with HF model
    with _stage("run_wwb_hf_reference"):
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
            "--hf",
        ]
    )

    outputs_path = tmp_path / "optimum"
    # test Optimum
    with _stage("run_wwb_optimum"):
        outputs = run_wwb(
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
            "--output",
            outputs_path,
        ]
    )

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs

    similarity = get_similarity(outputs)
    assert similarity >= SIMILARITY_THRESHOLD

    remove_artifacts(outputs_path)

    outputs_path = tmp_path / "genai"
    # test GenAI
    with _stage("run_wwb_genai"):
        outputs = run_wwb(
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
            "--genai",
            "--output",
            outputs_path,
        ]
    )

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs

    similarity = get_similarity(outputs)
    assert similarity >= SIMILARITY_THRESHOLD

    # test w/o models
    with _stage("run_wwb_metrics_without_models"):
        run_wwb(
        [
            "--target-data",
            outputs_path / "target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
        ]
    )

    remove_artifacts(outputs_path)


@pytest.mark.parametrize(
    ("model_id", "model_type", "batch_size"),
    [
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding", 1),
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding", 12),
    ],
)
@pytest.mark.xfail(sys.platform == "win32", reason="Ticket 178790", run=False)
def test_embeddings_with_batch(model_id, model_type, batch_size, tmp_path):
    _log(f"test_embeddings_with_batch: model_id={model_id} model_type={model_type} batch_size={batch_size}")
    GT_FILE = tmp_path / f"gt_batch_{batch_size}.csv"
    with _stage("convert_model"):
        MODEL_PATH = convert_model(model_id)
    _log(f"Model converted to: {MODEL_PATH}")
    SIMILARITY_THRESHOLD = 0.99

    # Collect reference with HF model
    with _stage("run_wwb_hf_reference"):
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
            "--embeds_batch_size",
            str(batch_size),
            "--hf",
        ]
    )

    # test Optimum
    outputs_path = tmp_path / "optimum"
    with _stage("run_wwb_optimum"):
        outputs = run_wwb(
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
            "--output",
            outputs_path,
            "--embeds_batch_size",
            str(batch_size),
        ]
    )
    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs

    similarity = get_similarity(outputs)
    assert similarity >= SIMILARITY_THRESHOLD

    remove_artifacts(outputs_path)

    # test GenAI
    outputs_path = tmp_path / "genai"
    with _stage("run_wwb_genai"):
        outputs = run_wwb(
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
            "--genai",
            "--output",
            outputs_path,
            "--embeds_batch_size",
            str(batch_size),
        ]
    )

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs

    similarity = get_similarity(outputs)
    assert similarity >= SIMILARITY_THRESHOLD

    # test w/o models
    with _stage("run_wwb_metrics_without_models"):
        run_wwb(
        [
            "--target-data",
            outputs_path / "target.csv",
            "--num-samples",
            "1",
            "--gt-data",
            GT_FILE,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--embeds_batch_size",
            str(batch_size),
        ]
    )

    remove_artifacts(outputs_path)
